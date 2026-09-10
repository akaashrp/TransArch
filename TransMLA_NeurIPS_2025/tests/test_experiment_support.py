import json
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from test_pretrained_conversion import source_model, batches
from transmla.convert_pretrained import convert_model
from transmla.modeling_transmla import TransMLAAttention, TransMLAConfig, apply_mla_rope
from transmla.experiments.common import RunStore, atomic_json, digest
from transmla.experiments.hf_backend import HFGenerator, Sampling, decode_completion
from transmla.experiments.evaluate import summarize_math, summarize_ruler, run_likelihood, run_reasoning
from transmla.experiments.campaign import build_jobs, launch, evaluation_args


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(12)
    torch.set_num_threads(2)


def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    raw = Tokenizer(models.WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2,
                                      "hello": 3, "world": 4}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    return tok


@pytest.fixture
def harness_task_subset():
    import lm_eval
    from lm_eval.tasks import TaskManager
    task_root = Path(lm_eval.__file__).parent / "tasks"
    manager = TaskManager(include_defaults=False,
                          include_path=[task_root / "piqa", task_root / "gsm8k"])
    # Retain the real upstream definitions without indexing unrelated suites
    # on the shared filesystem for every small integration test.
    with patch("lm_eval.evaluator.TaskManager", return_value=manager):
        yield


@pytest.mark.parametrize("family", ["qwen3", "mimo"])
@torch.inference_mode()
def test_tiled_attention_never_builds_full_model_mask(family):
    model = convert_model(source_model(family), batches(), kv_lora_rank=12, qk_mqa_dim=4,
                          freqfold=2, source_model_type=family)
    tokens = torch.randint(2, 41, (2, 43))
    mask = torch.ones_like(tokens)
    mask[0, :7] = 0
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    expected = model(tokens, attention_mask=mask, position_ids=positions, use_cache=False).logits
    model.config.mla_query_chunk_size = 3
    model.config.mla_score_budget_mb = 0.001
    with patch("transformers.models.qwen3.modeling_qwen3.create_causal_mask", side_effect=AssertionError("dense mask")):
        actual = model(tokens, attention_mask=mask, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for layer in model.model.layers:
        shape = layer.self_attn.last_score_tile_shape
        assert shape[-2] <= 3 and shape[-2] < shape[-1]
    # Respect arbitrary 4D masks as well as causality.
    allowed = torch.ones(2, 1, 43, 43, dtype=torch.bool)
    allowed[0, :, :, :7] = False
    actual_4d = model(tokens, attention_mask=allowed, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(actual_4d, expected, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_real_attention_dimensions_expanded_equals_absorbed():
    cfg = TransMLAConfig(hidden_size=64, intermediate_size=96, num_hidden_layers=1,
                         num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                         kv_lora_rank=512, qk_mqa_dim=64, mla_query_chunk_size=2)
    cfg._attn_implementation = "sdpa"
    attn = TransMLAAttention(cfg, 0).eval()
    attn.q_rope_weight.normal_(std=0.03)
    hidden = torch.randn(1, 13, 64)
    phases = torch.randn(1, 13, 128)
    cos, sin = phases.cos(), phases.sin()
    actual, _ = attn(hidden, (cos, sin))
    q, qr, latent, kr = attn.project(hidden)
    qr = apply_mla_rope(qr, cos[..., ::2], sin[..., ::2])
    kr = apply_mla_rope(kr, cos[..., ::2], sin[..., ::2])
    ku, vu = attn.kv_b_proj.weight.view(32, 256, 512).split(128, 1)
    query, key, value = attn._expanded_qkv(q, qr, latent, kr, ku, vu)
    assert query.shape[-1] == key.shape[-1] == value.shape[-1] == 192
    output = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True,
                                                              scale=attn.scaling)[..., :128]
    expected = attn.o_proj(output.transpose(1, 2).flatten(-2))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_completion_eos_and_length_are_distinct():
    tok = tokenizer()
    stopped = decode_completion(torch.tensor([3, 1, 0, 0]), tok, {1}, 4)
    assert stopped["token_ids"] == [3, 1] and stopped["finish_reason"] == "stop"
    limited = decode_completion(torch.tensor([3, 4]), tok, {1}, 2)
    assert limited["truncated"] and limited["gen_tokens"] == 2
    with pytest.raises(RuntimeError):
        decode_completion(torch.tensor([3]), tok, {1}, 2)


@torch.inference_mode()
def test_real_hf_generation_and_context_guard():
    model = convert_model(source_model("qwen3"), batches(), kv_lora_rank=12, qk_mqa_dim=4, freqfold=2)
    gen = HFGenerator(model, tokenizer())
    sampling = Sampling(n=2, temperature=0.6, top_p=0.95, top_k=20, max_tokens=3)
    first = gen.generate("hello world", sampling, 7)
    assert first == gen.generate("hello world", sampling, 7)
    assert len(first["outputs"]) == 2 and first["prompt_tokens"] == 2
    with pytest.raises(ValueError, match="native context"):
        gen.generate("hello", Sampling(max_tokens=129), 0)


def test_atomic_resume_rejects_changed_protocol(tmp_path):
    run = RunStore(tmp_path, {"model": "example", "cap": 128})
    run.put("0", {"complete": True})
    assert RunStore(tmp_path, {"model": "example", "cap": 128}).get("0") == {"complete": True}
    with pytest.raises(ValueError, match="identity changed"):
        RunStore(tmp_path, {"model": "example", "cap": 4096})


def test_math_sample_metrics_and_fractional_retrieval():
    completion = {"truncated": False, "gen_tokens": 4}
    items = [{"correct": [True] + [False] * 7, "outputs": [completion] * 8},
             {"correct": [False] * 8, "outputs": [completion] * 8}]
    score = summarize_math(items)
    assert score["pass@1"] == 1/16 and score["pass@8"] == 0.5
    assert summarize_ruler([{"score": 0.25, "outputs": [completion]},
                            {"score": 0.75, "outputs": [completion]}])["accuracy"] == 0.5


def test_campaign_has_complete_disjoint_shards_and_no_implicit_submission(tmp_path):
    sources = {"qwen3": {"path": "/qwen"}, "mimo": {"path": "/mimo"}}
    rows, jobs = build_jobs(tmp_path, sources, tmp_path / "data", [512])
    assert len(rows) == 4 and [len(jobs[x]) for x in ("source", "convert", "diagnostic", "full")] == [2, 2, 4, 476]
    row = rows[0]
    for thinking in (True, False):
        shards = [j for j in jobs["full"] if j["row"] == row and j["task"] == "math500" and j["thinking"] == thinking]
        assert sorted(i for j in shards for i in range(j["offset"], j["offset"] + j["limit"])) == list(range(500))
    niah = [j for j in jobs["full"] if j["row"] == row and j["task"] == "niah_multikey" and j["length"] == 32768]
    assert sum(j["limit"] for j in niah) == 500 and len({j["seed"] for j in niah}) == 5
    with patch("transmla.experiments.campaign.preflight"), patch("subprocess.check_output", side_effect=AssertionError("submitted")):
        launch({"root": str(tmp_path)}, execute=False)
    assert not (tmp_path / "submission.json").exists()


@torch.inference_mode()
def test_actual_lm_eval_hf_backend_with_offline_staged_dataset(tmp_path, harness_task_subset):
    from datasets import Dataset, DatasetDict
    data = tmp_path / "data"
    rows = {"goal": ["hello", "world"], "sol1": ["hello", "world"],
            "sol2": ["world", "hello"], "label": [0, 1]}
    DatasetDict({"train": Dataset.from_dict(rows), "validation": Dataset.from_dict(rows)}).save_to_disk(str(data / "piqa"))
    atomic_json(data / "harness_registry.json", {digest(["baber/piqa", None]): {"directory": "piqa"}})
    model = convert_model(source_model("qwen3"), batches(), kv_lora_rank=12, qk_mqa_dim=4, freqfold=2)
    args = SimpleNamespace(out=str(tmp_path / "results"), data=str(data), task="piqa", limit=2, seed=0)
    run_likelihood(args, model, tokenizer())
    result = json.loads((Path(args.out) / "results.json").read_text())
    assert result["status"] == "complete" and "piqa" in result["lm_eval"]["results"]
    assert len(result["lm_eval"]["samples"]["piqa"]) == 2


@pytest.mark.parametrize("family", ["qwen3", "mimo"])
@pytest.mark.parametrize("converted", [False, True], ids=["teacher", "transmla"])
@torch.inference_mode()
def test_gsm8k_harness_uses_native_no_thinking_prompts(tmp_path, family, converted, harness_task_subset):
    from datasets import Dataset, DatasetDict
    from huggingface_hub import snapshot_download
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer
    from transmla.experiments.campaign import SOURCES

    source = SOURCES[family]
    directory = snapshot_download(source["repo"], revision=source["revision"], local_files_only=True)
    tok = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=True)
    model = source_model(family)
    if converted:
        model = convert_model(model, batches(), kv_lora_rank=12, qk_mqa_dim=4,
                              freqfold=2, source_model_type=family)
        tok.save_pretrained(tmp_path / "exported-tokenizer")
        tok = AutoTokenizer.from_pretrained(tmp_path / "exported-tokenizer", local_files_only=True)
    model.config.max_position_embeddings = 8192
    data = tmp_path / "data"
    train = {"question": [f"Training question {i}: what is {i} + 1?" for i in range(7)],
             "answer": [f"Add one.\n#### {i + 1}" for i in range(7)]}
    test = {"question": ["Held-out question: what is 40 + 2?"], "answer": ["Add two.\n#### 42"]}
    DatasetDict({"train": Dataset.from_dict(train), "test": Dataset.from_dict(test)}).save_to_disk(str(data / "gsm8k"))
    atomic_json(data / "harness_registry.json", {digest(["openai/gsm8k", "main"]): {"directory": "gsm8k"}})
    args = SimpleNamespace(out=str(tmp_path / "results"), data=str(data), task="gsm8k", limit=1, seed=0)
    prompts = []

    # Exercise real harness sampling, chat rendering, tokenization and scoring.
    # Replace only model inference so this regression requires no full weights.
    def controlled_generation(wrapper, context, max_length, stop, **kwargs):
        prompt = wrapper.tokenizer.decode(context[0], skip_special_tokens=False)
        assert wrapper.enable_thinking is False
        assert max_length - context.shape[1] == 1024 and kwargs["do_sample"] is False
        assert "Question:" in stop
        assert prompt.count("Question:") == 6 and prompt.count("Training question") == 5
        assert prompt.count("#### ") == 5 and "Held-out question: what is 40 + 2?" in prompt
        assert prompt.count("<|im_start|>user\n") == 1
        assert prompt.count("<|im_start|>assistant\n") == 1
        assert re.search(r"<\|im_start\|>assistant\n<think>\s*</think>\s*$", prompt)
        prompts.append(prompt)
        answer = wrapper.tokenizer.encode("Add two.\n#### 42", add_special_tokens=False)
        return torch.cat((context, torch.tensor([answer], device=context.device)), dim=1)

    with patch.object(HFLM, "_model_generate", controlled_generation):
        run_likelihood(args, model, tok)
    report = json.loads((Path(args.out) / "results.json").read_text())["lm_eval"]
    assert len(prompts) == 1 and report["configs"]["gsm8k"]["num_fewshot"] == 5
    assert report["results"]["gsm8k"]["exact_match,strict-match"] == 1.0
    assert report["results"]["gsm8k"]["exact_match,flexible-extract"] == 1.0


def test_gsm8k_rejects_thinking_mode_before_loading():
    from transmla.experiments.evaluate import main
    with pytest.raises(ValueError, match="no-thinking"):
        main(["--task", "gsm8k", "--model", "/unused", "--data", "/unused", "--out", "/unused", "--thinking"])


def test_math_runner_reuses_results_without_regeneration(tmp_path):
    atomic_json(tmp_path / "unused.json", {})
    (tmp_path / "math500.jsonl").write_text(json.dumps({"problem": "What is 2+2?", "answer": "4"}) + "\n")
    args = SimpleNamespace(task="math500", data=str(tmp_path), offset=0, limit=1, thinking=True,
                           n=1, max_tokens=32, seed=0, out=str(tmp_path / "run"))
    store = RunStore(args.out, {"protocol": "test"})
    completion = {"outputs": [{"text": "\\boxed{4}", "token_ids": [3, 1], "gen_tokens": 2,
                                 "finish_reason": "stop", "truncated": False}], "seed": 0}
    with patch.object(HFGenerator, "generate", return_value=completion) as generate:
        run_reasoning(args, None, tokenizer(), store)
        run_reasoning(args, None, tokenizer(), store)
        assert generate.call_count == 1
    assert json.loads((Path(args.out) / "results.json").read_text())["summary"]["accuracy"] == 1.0


@torch.inference_mode()
def test_native_mimo_decoder_parity_from_pinned_source():
    from huggingface_hub import snapshot_download
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    directory = snapshot_download("XiaomiMiMo/MiMo-7B-RL-0530",
                                  revision="323400599af3903adc2a536d6340a23fee88d2e0", local_files_only=True)
    native_cls = get_class_from_dynamic_module("modeling_mimo.MiMoForCausalLM", directory, local_files_only=True)
    config_cls = get_class_from_dynamic_module("configuration_mimo.MiMoConfig", directory, local_files_only=True)
    source = source_model("mimo")
    config = config_cls(**{**source.config.to_dict(), "num_nextn_predict_layers": 1})
    config._attn_implementation = "sdpa"
    native = native_cls(config).eval()
    info = native.load_state_dict(source.state_dict(), strict=False)
    assert not info.unexpected_keys and all(k.startswith("model.mtp_layers.") for k in info.missing_keys)
    ids = torch.randint(2, 41, (1, 13))
    expected = source(ids, use_cache=False).logits
    torch.testing.assert_close(native(ids, use_cache=False).logits, expected, atol=1e-6, rtol=1e-5)
    prefix = native(ids[:, :5], use_cache=True)
    suffix = native(ids[:, 5:], past_key_values=prefix.past_key_values, use_cache=True)
    torch.testing.assert_close(suffix.logits, expected[:, 5:], atol=1e-6, rtol=1e-5)


def test_source_protocol_extraction_is_verbatim():
    import ast
    import subprocess
    from transmla.experiments import protocol
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / "experiments/protocol_source.json").read_text())
    source = subprocess.check_output(["git", "show", f"{metadata['commit']}:{metadata['path']}"],
                                     cwd=root.parents[1] / "Linearization").decode()
    current = Path(protocol.__file__).read_text()
    def spans(text):
        result = {}
        for node in ast.parse(text).body:
            name = node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None
            if name in metadata["extracted_names"]:
                result[name] = ast.get_source_segment(text, node)
        return result
    assert spans(source) == spans(current)


def test_failed_gpu_gate_cannot_release_an_evaluation(tmp_path):
    from transmla.experiments.validate import require_gate
    path = tmp_path / "failed.json"
    atomic_json(path, {"status": "failed"})
    with pytest.raises(ValueError, match="GPU validation"):
        require_gate(path, tmp_path / "nonexistent-model")


def test_collection_never_scores_missing_shards_as_zero(tmp_path):
    from transmla.experiments.collect import collect
    _, jobs = build_jobs(tmp_path, {"qwen3": {"path": "/qwen"}}, tmp_path / "data", [512])
    report = collect({"jobs": jobs, "external_comparators": {}})
    assert report["status"] == "partial" and len(report["missing_job_indices"]) == 238
    assert all("accuracy" not in r for r in report["rows"])


def test_collector_matched_teacher_delta_and_coverage(tmp_path):
    from transmla.experiments.collect import collect
    from transmla.experiments.evaluate import parser
    jobs = []
    plan = {"jobs": {"full": jobs}, "external_comparators": {}, "code_sha256": "code",
            "data_manifest_sha256": "data", "environment": {}, "data": "/data", "item_tokenizer": "/qwen"}
    for index, method in enumerate(("teacher", "transmla")):
        row = {"id": f"qwen3-{method}", "model": f"/models/{method}"}
        job = {"row": row, "task": "math500", "thinking": True, "max_tokens": 32,
               "limit": 1, "offset": 0, "n": 1, "seed": 0, "index": index, "output": str(tmp_path / method)}
        jobs.append(job)
        arguments = vars(parser().parse_args(evaluation_args(plan, job, job["output"])))
        atomic_json(Path(job["output"]) / "manifest.json", {"arguments": arguments, "code_sha256": "code",
                    "data_manifest_sha256": "data", "environment": {}, "backend": "hf",
                    "checkpoint": {"path": row["model"]}})
        atomic_json(Path(job["output"]) / "results.json", {"status": "complete", "items": [
            {"idx": 0, "correct": [method == "teacher"], "outputs": [{"truncated": False, "gen_tokens": 3}]}]})
    report = collect(plan)
    assert report["status"] == "complete"
    assert {r["row"]: r["delta_from_teacher"] for r in report["rows"]} == {"qwen3-teacher": 0.0, "qwen3-transmla": -1.0}
    atomic_json(tmp_path / "teacher/results.json", {"status": "complete", "items": []})
    with pytest.raises(ValueError, match="sample coverage"):
        collect(plan)


@pytest.mark.parametrize("mismatch", ["context", "checkpoint"])
def test_collector_rejects_wrong_context_or_mixed_checkpoints(tmp_path, mismatch):
    from transmla.experiments.collect import collect
    from transmla.experiments.evaluate import parser
    jobs = []
    plan = {"jobs": {"full": jobs}, "external_comparators": {}, "code_sha256": "code",
            "data_manifest_sha256": "data", "environment": {}, "data": "/data", "item_tokenizer": "/qwen"}
    for index in range(2):
        row = {"id": "qwen3-teacher", "model": "/models/teacher"}
        job = {"row": row, "task": "niah_single", "thinking": False, "max_tokens": 128, "length": 4096,
               "limit": 1, "offset": 0, "n": 1, "seed": index, "index": index, "output": str(tmp_path / str(index))}
        jobs.append(job)
        arguments = vars(parser().parse_args(evaluation_args(plan, job, job["output"])))
        identity = {"path": row["model"], "weights": {"shard": [100, 1]}}
        if index == 1:
            if mismatch == "context":
                arguments["length"] = 8192
            else:
                identity["weights"]["shard"][1] = 2
        atomic_json(Path(job["output"]) / "manifest.json", {"arguments": arguments, "code_sha256": "code",
                    "data_manifest_sha256": "data", "environment": {}, "backend": "hf", "checkpoint": identity})
        atomic_json(Path(job["output"]) / "results.json", {"status": "complete", "items": [
            {"idx": 0, "score": 1.0, "expected": ["needle"], "input_sha256": str(index),
             "outputs": [{"truncated": False, "gen_tokens": 3}]}]})
    with pytest.raises(ValueError, match="identity mismatch|Mixed checkpoints"):
        collect(plan)


def test_all_generated_evaluation_commands_parse(tmp_path):
    from transmla.experiments.evaluate import parser
    _, jobs = build_jobs(tmp_path, {"qwen3": {"path": "/qwen"}}, tmp_path / "data", [512, 1024])
    plan = {"data": str(tmp_path / "data"), "item_tokenizer": "/qwen"}
    for job in jobs["full"]:
        args = parser().parse_args(evaluation_args(plan, job, job["output"]))
        assert args.model == job["row"]["model"] and args.task == job["task"]
        assert args.max_tokens == job["max_tokens"] and args.thinking == job["thinking"]


@torch.inference_mode()
def test_conversion_stage_diagnostics_are_separate_from_calibration():
    from transmla.utils import evaluate_ppl
    probes = [{"input_ids": torch.tensor([[3,4,5,6]]), "attention_mask": torch.ones(1,4,dtype=torch.long)}]
    seen = {}
    convert_model(source_model("qwen3"), batches(), kv_lora_rank=12, qk_mqa_dim=4, freqfold=2,
                  stage_callback=lambda name, model: seen.update({name: evaluate_ppl(model, 0, probes)}))
    assert list(seen) == ["source", "partial_rope", "converted"]
    assert all(0 < value < float("inf") for value in seen.values())
