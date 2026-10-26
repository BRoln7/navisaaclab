#!/usr/bin/env python3
"""Download the small official TailorNet female T-shirt sample.

TailorNet data is governed by its own non-commercial research license.  This
script downloads it from the official Hugging Face dataset instead of
redistributing it with socnav_sim.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESTINATION = PROJECT_ROOT / "data/usd/assets/people/source"
BASE_URL = "https://huggingface.co/datasets/zycliao/TailorNet_dataset/resolve/main"
FILES = {
    "dataset_meta.zip": "56707f0ceeeb035f14ebcf4d183c4dc84e59f93667d0aa48fdd43aa85554aef3",
    "t-shirt_female_sample.zip": "e5a43bd7d1e317198ca28be25ffec61ff152b833319e7ef46927064ecff40bb6",
}
LEGACY_FILENAMES = {
    "dataset_meta.zip": "tailornet_dataset_meta.zip",
    "t-shirt_female_sample.zip": "tailornet_tshirt_female_sample.zip",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"[TailorNet] downloading {url}")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    temporary.replace(destination)


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe path in {archive.name}: {member.filename}")
        bundle.extractall(destination)


def install(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    source = destination / "tailornet"
    source.mkdir(parents=True, exist_ok=True)

    for filename, expected_hash in FILES.items():
        archive = destination / filename
        legacy_archive = destination / LEGACY_FILENAMES[filename]
        if not archive.is_file() and legacy_archive.is_file():
            archive = legacy_archive
        if not archive.is_file() or _sha256(archive) != expected_hash:
            archive = destination / filename
            _download(f"{BASE_URL}/{filename}?download=true", archive)
        actual_hash = _sha256(archive)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {archive}: expected {expected_hash}, got {actual_hash}"
            )
        print(f"[TailorNet] verified {filename}")
        _safe_extract(archive, source)

    license_path = destination / "LICENSE-tailornet.md"
    if not license_path.is_file():
        _download(f"{BASE_URL}/LICENSE.md", license_path)

    required = (
        source / "garment_class_info.pkl",
        source / "t-shirt_female/shape/beta_000.npy",
        source / "t-shirt_female/style_shape/beta000_gamma023.npy",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("TailorNet extraction is incomplete:\n  " + "\n  ".join(missing))
    print(f"[TailorNet] sample installed under {source}")
    print(f"[TailorNet] license: {license_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    install(args.destination.expanduser().resolve())


if __name__ == "__main__":
    main()
