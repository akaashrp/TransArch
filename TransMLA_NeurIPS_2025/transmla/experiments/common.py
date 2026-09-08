"""Local artifact identity, atomic results, and model loading."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def environment():
    packages = ("torch", "transformers", "datasets", "safetensors", "lm_eval", "math-verify", "accelerate",
                "peft", "evaluate", "latex2sympy2_extended", "numpy", "sympy", "tokenizers",
                "huggingface-hub", "antlr4-python3-runtime")
    return {p: importlib.metadata.version(p) for p in packages}


def source_hash():
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))
                   if "transformers" not in p.relative_to(root).parts})


class RunStore:
    """Resume only identical runs; each completed item is an atomic file."""
    def __init__(self, root, manifest):
        self.root = Path(root)
        path = self.root / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError(f"Run identity changed; choose a new output directory: {root}")
        atomic_json(path, manifest)

    def get(self, key):
        path = self.root / "items" / f"{key}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, key, value):
        atomic_json(self.root / "items" / f"{key}.json", value)


def load_model(path, device="cuda:0", dtype="bf16", native_mimo=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..convert_pretrained import load_source_model
    path = Path(path).resolve()
    raw = json.loads((path / "config.json").read_text())
    if raw["model_type"] in ("qwen3", "mimo", "qwen2") and not native_mimo:
        model, _ = load_source_model(SimpleNamespace(model_path=str(path), revision=None,
                                      local_files_only=True, dtype=dtype, device=device), raw)
    else:
        model, info = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, local_files_only=True,
            torch_dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[dtype],
            attn_implementation="sdpa", output_loading_info=True)
        if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
            raise ValueError(f"Checkpoint did not load exactly: {info}")
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model.eval().requires_grad_(False), tokenizer


def checkpoint_identity(path):
    path = Path(path).resolve()
    files = [p for p in path.iterdir() if p.suffix in (".json", ".py", ".jinja") or p.name == "validation_reference.pt"]
    # Source safetensor bytes are checked during staging. Track size/mtime too,
    # so an overwritten local checkpoint cannot silently resume old results.
    return {"path": str(path), "metadata": {p.name: file_hash(p) for p in sorted(files)},
            "weights": {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                        for p in sorted(path.glob("*.safetensors"))}}
