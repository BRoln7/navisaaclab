"""Install only the official SMPL asset folder from the MimicKit data ZIP."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_URL = "https://1sfu-my.sharepoint.com/:u:/g/personal/xbpeng_sfu_ca/EclKq9pwdOBAl-17SogfMW0Bved4sodZBQ_5eZCiz9O--w?e=bqXBaa"


def install(archive, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Asset folder already exists; it was left unchanged: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "smpl"
        staging.mkdir()
        files = {}
        for member in bundle.infolist():
            parts = PurePosixPath(member.filename).parts
            if ".." in parts or PurePosixPath(member.filename).is_absolute():
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
            matches = [i for i in range(len(parts)-1) if parts[i:i+2] == ("assets", "smpl")]
            if not matches or member.is_dir():
                continue
            if len(matches) != 1 or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unexpected archive member: {member.filename}")
            relative = PurePosixPath(*parts[matches[0]+2:])
            target = staging / relative
            if target.exists():
                raise ValueError(f"Duplicate asset member: {relative}")
            data = bundle.read(member)  # ZipFile verifies the member CRC.
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files[str(relative)] = hashlib.sha256(data).hexdigest()
        for required in ("smpl.xml", "smpl.usd", "LICENSE.txt"):
            if required not in files:
                raise ValueError(f"Official asset pack is missing assets/smpl/{required}")
        manifest = {"source": OFFICIAL_URL, "archive_folder": "assets/smpl", "sha256": files}
        (staging / "asset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(destination)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Official MimicKit_Data.zip from the upstream README")
    parser.add_argument("--destination", type=Path, default=ROOT / "third_party/MimicKit/data/assets/smpl")
    args = parser.parse_args()
    manifest = install(args.archive, args.destination)
    print(f"Installed {len(manifest['sha256'])} official SMPL asset files into {args.destination}")
