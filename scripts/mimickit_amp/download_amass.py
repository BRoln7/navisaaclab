"""Download original AMASS SMPL+H G subsets used by PACER; never import the simulator.

Requires an AMASS account with access to the selected datasets. Credentials are
prompted locally and sent in a HTTPS POST, never in argv, URLs, or output files.
"""

import argparse
import bz2
from collections import Counter
import csv
from datetime import datetime, timezone
from getpass import getpass
import hashlib
from http.client import IncompleteRead
from http.cookiejar import CookieJar
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import build_opener, HTTPRedirectHandler, HTTPCookieProcessor, Request
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD = "https://download.is.tue.mpg.de/download.php"
# Archive names differ from the legacy AMASS folder names used in PACER keys.
DATASETS = {
    "ACCAD": "ACCAD",
    "BMLrub": "BioMotionLab_NTroje",
    "EyesJapanDataset": "Eyes_Japan_Dataset",
    "HumanEva": "HumanEva",
    "KIT": "KIT",
    "SFU": "SFU",
    "TotalCapture": "TotalCapture",
    "Transitions": "Transitions_mocap",
}
BLOCK = 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def download_url(dataset):
    file = f"amass_per_dataset/smplh/gender_specific/mosh_results/{dataset}.tar.bz2"
    return DOWNLOAD + "?" + urlencode({"domain": "amass", "resume": 1, "sfile": file})


class OfficialRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        if parsed.scheme != "https" or not (parsed.hostname or "").endswith(".is.tue.mpg.de"):
            raise RuntimeError("AMASS redirected outside its HTTPS hosts; stopped.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def archive_header(path):
    with path.open("rb") as stream:
        return stream.read(3) == b"BZh"


def download_once(opener, url, destination, credentials, timeout):
    partial = destination.with_name(destination.name + ".part")
    metadata = partial.with_name(partial.name + ".json")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset and not archive_header(partial):
        raise RuntimeError(f"Invalid partial archive: {partial}. Move it aside before retrying.")
    headers = {"User-Agent": "socnav-amass-downloader/1.0", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved.get("etag") or saved.get("last_modified"):
                headers["If-Range"] = saved.get("etag") or saved["last_modified"]
    request = Request(url, data=urlencode(credentials()).encode(), headers=headers)
    try:
        response = opener.open(request, timeout=timeout)
    except HTTPError as error:
        if error.code == 416 and offset:
            # Some servers do not support range POSTs; safely restart this file.
            headers.pop("Range", None)
            headers.pop("If-Range", None)
            response = opener.open(Request(url, data=urlencode(credentials()).encode(), headers=headers), timeout=timeout)
        else:
            raise
    with response:
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/" in content_type or "html" in content_type or "json" in content_type:
            raise RuntimeError("AMASS returned a web page instead of data. Check your AMASS account and "
                               "dataset access at https://amass.is.tue.mpg.de/download.php .")
        status = response.status
        if status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != offset:
                raise RuntimeError("Invalid Content-Range; partial file was left unchanged.")
            expected_size = int(match[3])
            append = True
        elif status == 200:
            offset, append = 0, False
            size = response.headers.get("Content-Length")
            expected_size = int(size) if size else None
        else:
            raise RuntimeError(f"Unexpected HTTP status {status}")
        first = response.read(BLOCK)
        if (not first or (offset == 0 and not first.startswith(b"BZh"))
                or first.lstrip().lower().startswith((b"<!doctype", b"<html", b"<?xml"))):
            raise RuntimeError("Response is not a bzip2 archive. Check account access and the official download page.")
        if expected_size and expected_size - offset > shutil.disk_usage(destination.parent).free:
            raise RuntimeError("Insufficient disk space for the remaining archive download.")
        write_json(metadata, {"url": url, "etag": response.headers.get("ETag"),
                              "last_modified": response.headers.get("Last-Modified")})
        written, last_print = offset, 0.
        with partial.open("ab" if append else "wb") as stream:
            block = first
            while block:
                stream.write(block)
                written += len(block)
                if time.monotonic() - last_print >= 3:
                    progress = f" / {expected_size / 2**30:.2f} GiB" if expected_size else " GiB"
                    print(f"  {destination.name}: {written / 2**30:.2f}{progress}", flush=True)
                    last_print = time.monotonic()
                block = response.read(BLOCK)
        if expected_size is not None and written != expected_size:
            raise URLError(f"Incomplete download: {written} / {expected_size} bytes; retry to resume")
    partial.replace(destination)
    metadata.unlink(missing_ok=True)


def download(opener, dataset, destination, credentials, timeout, retries):
    if destination.exists():
        if not archive_header(destination):
            raise RuntimeError(f"Not a bzip2 archive: {destination}")
        print(f"[AMASS] Reusing archive: {destination.name}", flush=True)
    else:
        for attempt in range(retries + 1):
            try:
                download_once(opener, download_url(dataset), destination, credentials, timeout)
                break
            except (HTTPError, URLError, TimeoutError, ConnectionError, IncompleteRead) as error:
                if isinstance(error, HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {error.code}: check AMASS login/dataset access and download availability.") from None
                if attempt == retries:
                    raise RuntimeError(f"Download interrupted: {error}. Re-run the same command to resume.") from None
                print(f"[AMASS] Network interruption; retry {attempt + 1}/{retries}", flush=True)
                time.sleep(min(2 ** attempt, 15))
    digest = sha256(destination)
    record = destination.with_name(destination.name + ".sha256.json")
    if record.exists() and json.loads(record.read_text())["sha256"] != digest:
        raise RuntimeError(f"Archive changed since its checksum was recorded: {destination}")
    write_json(record, {"url": download_url(dataset), "bytes": destination.stat().st_size, "sha256": digest,
                        "note": "Local provenance checksum; not a publisher-provided checksum."})
    return digest


def unpack(archive, staging):
    # Drain the bzip2 stream through its trailer to validate CRC, including padding
    # after tar's end marker. Only regular files/directories may be extracted.
    with bz2.open(archive, "rb") as compressed, tarfile.open(fileobj=compressed, mode="r|") as bundle:
        for member in bundle:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member.name:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            target = staging.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(f"Unsupported archive member (link or special file): {member.name}")
            if member.size + 64 * 2**20 > shutil.disk_usage(staging).free:
                raise RuntimeError("Insufficient disk space for extraction; archive was retained.")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Unreadable archive member: {member.name}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, BLOCK)
            if target.stat().st_size != member.size:
                raise RuntimeError(f"Truncated archive member: {member.name}")
        while compressed.read(BLOCK):
            pass


def numpy_shape(bundle, key):
    import numpy as np

    with bundle.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(stream)[0]
        if version == (2, 0):
            return np.lib.format.read_array_header_2_0(stream)[0]
        raise RuntimeError(f"Unsupported NPY header version: {version}")


def inventory(folder, output):
    """Read array headers and scalar FPS only; leave motion samples untouched."""
    import numpy as np

    fps_counts, frames, count, skipped = Counter(), 0, 0, 0
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "fps", "frames", "pose_dimensions", "bytes"])
        for path in sorted(folder.rglob("*.npz")):
            with zipfile.ZipFile(path) as bundle:
                names = set(bundle.namelist())
                if "poses.npy" not in names:
                    skipped += 1
                    continue
                if not {"mocap_framerate.npy", "trans.npy", "betas.npy", "gender.npy"} <= names:
                    raise RuntimeError(f"Missing AMASS SMPL+H fields: {path}")
                pose = numpy_shape(bundle, "poses")
                trans = numpy_shape(bundle, "trans")
            with np.load(path, allow_pickle=False) as clip:
                fps = float(clip["mocap_framerate"].item())
            if (len(pose) != 2 or pose[1] != 156 or pose[0] < 1 or trans != (pose[0], 3)
                    or not np.isfinite(fps) or fps <= 0):
                raise RuntimeError(f"Unexpected SMPL+H shape/FPS: {path}; poses={pose}, trans={trans}, fps={fps}")
            writer.writerow([path.relative_to(folder).as_posix(), fps, pose[0], pose[1], path.stat().st_size])
            count += 1
            frames += pose[0]
            fps_counts[str(fps)] += 1
    if count == 0:
        raise RuntimeError(f"No original SMPL+H motions found in {folder}")
    return {"clips": count, "frames": frames, "source_fps_clip_counts": dict(fps_counts),
            "non_motion_npz": skipped, "inventory_sha256": sha256(output)}


def extract(archive, digest, dataset, root):
    canonical = DATASETS[dataset]
    raw, metadata = root / "raw", root / "metadata"
    raw.mkdir(exist_ok=True)
    metadata.mkdir(exist_ok=True)
    destination = raw / canonical
    marker = metadata / (canonical + ".json")
    if destination.exists():
        if marker.exists() and json.loads(marker.read_text()).get("archive_sha256") == digest:
            print(f"[AMASS] Already extracted: {destination} (use inventory CSV to inspect source FPS)", flush=True)
            return
        raise RuntimeError(f"Existing extraction has no matching provenance: {destination}. Use a new --output directory.")
    print(f"[AMASS] Extracting and inspecting {dataset} ...", flush=True)
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-", dir=raw) as temporary:
        staging = Path(temporary) / "contents"
        staging.mkdir()
        unpack(archive, staging)
        npz = list(staging.rglob("*.npz"))
        # Archives may have either the canonical dataset root or just subject folders.
        candidates = [staging / name for name in {dataset, canonical} if (staging / name).is_dir()]
        source = staging
        if len(candidates) == 1 and all(path.is_relative_to(candidates[0]) for path in npz):
            source = candidates[0]
        csv_temporary = Path(temporary) / "inventory.csv"
        info = inventory(source, csv_temporary)
        info.update(dataset=canonical, archive=archive.name, archive_sha256=digest,
                    format="AMASS SMPL+H gender-specific", source_url=download_url(dataset),
                    created_utc=datetime.now(timezone.utc).isoformat(), resampled=False)
        source.rename(destination)
        csv_temporary.replace(metadata / (canonical + ".csv"))
        write_json(marker, info)
        print(f"[AMASS] {canonical}: {info['clips']} clips; source FPS {info['source_fps_clip_counts']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/amass_raw/pacer_smplh_g")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        help="Default: PACER's 8 source subsets; accepts archive or legacy AMASS folder names")
    parser.add_argument("--plan", action="store_true", help="List official URLs without network requests or credentials")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--no-extract", action="store_true", help="Download archives only")
    modes.add_argument("--extract-only", action="store_true", help="Use locally downloaded archives; no login/network")
    parser.add_argument("--timeout", type=float, default=120, help="Socket timeout, seconds")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.timeout <= 0 or args.retries < 0:
        parser.error("--timeout must be positive and --retries nonnegative")
    aliases = {name: name for name in DATASETS} | {legacy: name for name, legacy in DATASETS.items()}
    if any(name not in aliases for name in args.datasets):
        parser.error("Unknown dataset. Choices: " + ", ".join(aliases))
    datasets = list(dict.fromkeys(aliases[name] for name in args.datasets))
    print("[AMASS] Original SMPL+H G data, using the official AMASS download service.", flush=True)
    print("[AMASS] Account registration and dataset access: https://amass.is.tue.mpg.de/download.php", flush=True)
    if args.plan:
        for dataset in datasets:
            print(f"{dataset} -> raw/{DATASETS[dataset]}\n  {download_url(dataset)}")
        return
    # Import only NumPy before downloading, so a missing dependency fails early.
    if not args.no_extract:
        import numpy  # noqa: F401
    root = args.output.expanduser().resolve()
    archives = root / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    secret = {}

    def credentials():
        if not secret:
            secret["username"] = input("AMASS username/email: ").strip()
            secret["password"] = getpass("AMASS password (hidden): ")
            if not secret["username"] or not secret["password"]:
                raise RuntimeError("AMASS username and password must be nonempty.")
        return secret

    opener = build_opener(OfficialRedirect(), HTTPCookieProcessor(CookieJar()))
    for dataset in datasets:
        archive = archives / (dataset + ".tar.bz2")
        if args.extract_only and not archive.is_file():
            raise RuntimeError(f"Missing archive: {archive}")
        digest = download(opener, dataset, archive, credentials, args.timeout, args.retries)
        if not args.no_extract:
            extract(archive, digest, dataset, root)
    secret.clear()
    print(f"[AMASS] Complete: {root}\n[AMASS] Raw samples/FPS retained; training data configuration unchanged.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[AMASS] Interrupted; completed archives and .part downloads retained. Re-run to resume.", file=sys.stderr)
        sys.exit(130)
    except (OSError, RuntimeError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"[AMASS] Error: {error}", file=sys.stderr)
        sys.exit(1)
