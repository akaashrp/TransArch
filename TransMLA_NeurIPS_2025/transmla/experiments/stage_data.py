"""Download and freeze calibration/evaluation data on a networked login node."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from .common import atomic_json, digest, file_hash
from .protocol import MATH_TASKS

TASKS = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "mmlu", "gsm8k"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    import datasets
    from huggingface_hub import HfApi
    from lm_eval.tasks import TaskManager
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    revision_path = root / "revisions.json"
    revisions = json.loads(revision_path.read_text()) if revision_path.exists() else {}
    def revision(repo):
        if repo not in revisions:
            revisions[repo] = HfApi().dataset_info(repo).sha
            atomic_json(revision_path, revisions)
        return revisions[repo]

    registry_path = root / "harness_registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    original_load = datasets.load_dataset
    def staged_load(path, name=None, **kwargs):
        key = digest([path, name])
        directory = f"harness/{key}"
        if key in registry:
            return datasets.load_from_disk(str(root / directory))
        kwargs["revision"] = revision(path)
        ds = original_load(path, name, **kwargs)
        ds.save_to_disk(str(root / directory))
        registry[key] = {"repository": path, "name": name, "revision": kwargs["revision"],
                         "directory": directory, "splits": {s: {"rows": len(d), "fingerprint": d._fingerprint}
                                                                   for s, d in ds.items()}}
        atomic_json(registry_path, registry)
        return ds
    with patch.object(datasets, "load_dataset", staged_load):
        loaded = TaskManager().load(TASKS)
    task_configs = {name: task.dump_config() for name, task in loaded["tasks"].items()}
    atomic_json(root / "harness_tasks.json", task_configs)

    sources = {}
    for name in ("math500", "aime24", "aime25"):
        repo, split, _ = MATH_TASKS[name]
        destination = root / f"{name}.jsonl"
        ds = original_load(repo, split=split, revision=revision(repo))
        destination.write_text("".join(json.dumps(row) + "\n" for row in ds))
        sources[name] = {"repository": repo, "revision": revision(repo), "split": split,
                         "rows": len(ds), "fingerprint": ds._fingerprint}
    repo = "Salesforce/wikitext"
    ds = original_load(repo, "wikitext-2-raw-v1", revision=revision(repo))
    for split, name in (("train", "calibration"), ("validation", "validation")):
        (root / f"{name}.jsonl").write_text("".join(json.dumps({"text": r["text"]}) + "\n" for r in ds[split]))
        sources[name] = {"repository": repo, "revision": revision(repo), "split": split,
                         "rows": len(ds[split]), "fingerprint": ds[split]._fingerprint}
    # Hash all staged files. No benchmark labels are used for calibration.
    files = {str(f.relative_to(root)): file_hash(f) for f in sorted(root.rglob("*"))
             if f.is_file() and f.name != "manifest.json" and not f.name.startswith(".")}
    atomic_json(root / "manifest.json", {"sources": sources, "files": files,
                "harness_tasks": TASKS, "lm_eval": "0.4.12", "calibration_split": "train"})
    print(f"Staged {len(loaded['tasks'])} harness tasks and three math datasets at {root}")


if __name__ == "__main__":
    main()
