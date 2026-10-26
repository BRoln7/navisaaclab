"""Install an official MimicKit SMPL motion from its published data ZIP."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_URL = "https://1sfu-my.sharepoint.com/:u:/g/personal/xbpeng_sfu_ca/EclKq9pwdOBAl-17SogfMW0Bved4sodZBQ_5eZCiz9O--w?e=bqXBaa"
MOTION_DIRECTORY = ROOT / "third_party/MimicKit/data/motions/smpl"
DEFAULT_MOTION_NAME = "smpl_walk.pkl"
DESTINATION = MOTION_DIRECTORY / DEFAULT_MOTION_NAME


def _validate_motion_name(motion_name):
    path = PurePosixPath(motion_name)
    if path.name != motion_name or path.suffix != ".pkl":
        raise ValueError(f"Expected a plain .pkl filename, got: {motion_name}")


def _motion_member(bundle, motion_name=DEFAULT_MOTION_NAME):
    _validate_motion_name(motion_name)
    matches = []
    suffixes = (
        ("data", "motions", "smpl", motion_name),
        ("motions", "smpl", motion_name),
    )
    for member in bundle.infolist():
        parts = PurePosixPath(member.filename).parts
        if ".." in parts or PurePosixPath(member.filename).is_absolute():
            raise ValueError(f"Unsafe ZIP path: {member.filename}")
        if member.is_dir() or stat.S_ISLNK(member.external_attr >> 16):
            continue
        if any(tuple(parts[-len(suffix):]) == suffix for suffix in suffixes):
            matches.append(member)
    if len(matches) != 1:
        names = [member.filename for member in matches]
        raise ValueError(f"Expected exactly one official {motion_name}, found {names}")
    return matches[0]


def install(archive, destination=DESTINATION, motion_name=DEFAULT_MOTION_NAME):
    archive = Path(archive).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    _validate_motion_name(motion_name)
    if not archive.is_file():
        raise FileNotFoundError(f"MimicKit data ZIP does not exist: {archive}")

    with zipfile.ZipFile(archive) as bundle:
        member = _motion_member(bundle, motion_name)
        data = bundle.read(member)  # Also verifies the member CRC.
    digest = hashlib.sha256(data).hexdigest()
    if motion_name == DEFAULT_MOTION_NAME:
        manifest_name = "official-install.json"
    else:
        manifest_name = f"official-{Path(motion_name).stem}-install.json"
    manifest_path = destination.with_name(manifest_name)

    if destination.exists():
        existing = hashlib.sha256(destination.read_bytes()).hexdigest()
        if existing != digest:
            raise FileExistsError(f"Existing motion differs and was left unchanged: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(destination)

    manifest = {
        "source": OFFICIAL_URL,
        "archive": str(archive),
        "archive_member": member.filename,
        "destination": str(destination),
        "size_bytes": len(data),
        "sha256": digest,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("sha256") != digest:
            raise FileExistsError(f"Existing manifest differs and was left unchanged: {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive", type=Path, help="Official MimicKit_Data.zip linked by the upstream README"
    )
    parser.add_argument(
        "--motion-name", default=DEFAULT_MOTION_NAME, help="Filename below motions/smpl in the archive"
    )
    parser.add_argument(
        "--destination", type=Path, help="Destination path (defaults to MimicKit's SMPL motion directory)"
    )
    args = parser.parse_args()
    destination = args.destination or (MOTION_DIRECTORY / args.motion_name)
    result = install(args.archive, destination, args.motion_name)
    print(f"Official SMPL motion installed: {result['destination']}")
    print(f"SHA-256: {result['sha256']}")
