from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST_PATH = ROOT / "openenv.yaml"
GENERATED_MANIFEST_PATH = ROOT / "environments" / "openenv.yaml"


def sync_manifest() -> None:
    GENERATED_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CANONICAL_MANIFEST_PATH, GENERATED_MANIFEST_PATH)


if __name__ == "__main__":
    sync_manifest()
