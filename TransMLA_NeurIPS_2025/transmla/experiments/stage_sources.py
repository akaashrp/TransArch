"""Stage pinned source snapshots; no model execution or scheduler submission."""
from huggingface_hub import snapshot_download
from .campaign import SOURCES


def main():
    for entry in SOURCES.values():
        directory = snapshot_download(entry["repo"], revision=entry["revision"],
                                      allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.model", "*.jinja"],
                                      max_workers=2)
        print(f"{entry['repo']} @ {entry['revision']}: {directory}", flush=True)


if __name__ == "__main__":
    main()
