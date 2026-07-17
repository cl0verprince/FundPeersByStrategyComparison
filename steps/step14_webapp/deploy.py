"""Deploy webapp/ (including the current extract) to a Hugging Face Space. Manual and
human-gated - never called by advance.py. Requires `hf auth login` (or HF_TOKEN)."""
import argparse
import logging

from huggingface_hub import HfApi

log = logging.getLogger(__name__)


def deploy(space_id: str) -> str:
    api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path="webapp", repo_id=space_id, repo_type="space",
                      ignore_patterns=["__pycache__", "*.pyc", "data/.gitkeep"])
    url = f"https://huggingface.co/spaces/{space_id}"
    log.info("deployed: %s", url)
    return url


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="e.g. <user>/fundspeers")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    deploy(args.space)
