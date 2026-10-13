"""Download the XTTS-v2 model files (~2 GB) to a local directory.

XTTS-v2 is licensed under the Coqui Public Model License (non-commercial).
Read https://coqui.ai/cpml and pass --accept-license to confirm you comply.

    python scripts/download_xtts.py --accept-license --dest models/XTTS-v2
"""

import argparse
import sys
from pathlib import Path

FILES = [
    "config.json",
    "model.pth",
    "vocab.json",
    "speakers_xtts.pth",
    "dvae.pth",
    "mel_stats.pth",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="models/XTTS-v2")
    parser.add_argument("--accept-license", action="store_true")
    args = parser.parse_args()

    if not args.accept_license:
        print("Refusing to download: read https://coqui.ai/cpml and re-run with --accept-license.")
        return 2
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is missing. Run: pip install -r requirements-xtts.txt")
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="coqui/XTTS-v2", local_dir=str(dest), allow_patterns=FILES)
    missing = [f for f in ("config.json", "model.pth", "vocab.json") if not (dest / f).is_file()]
    if missing:
        print(f"Download incomplete, missing: {', '.join(missing)}")
        return 1
    print(f"Model ready in {dest}. Set XTTS_MODEL_DIR={dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
