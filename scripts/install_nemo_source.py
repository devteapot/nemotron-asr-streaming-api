from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a pinned NeMo source tree from GitHub.")
    parser.add_argument("ref", help="Git ref, tag, or commit SHA to install")
    parser.add_argument("target", help="Target source directory, for example /opt/NeMo")
    args = parser.parse_args()

    target = Path(args.target)
    url = f"https://github.com/NVIDIA/NeMo/archive/{args.ref}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "nemo.tar.gz"
        extract_path = tmp_path / "extract"
        extract_path.mkdir()

        print(f"downloading NeMo source {args.ref}")
        urllib.request.urlretrieve(url, archive_path)

        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if not members:
                raise RuntimeError("NeMo source archive is empty")
            _safe_extract(archive, members, extract_path)

        source_root = _single_child_dir(extract_path)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source_root, target)
        print(f"installed NeMo source {args.ref} to {target}")


def _safe_extract(archive: tarfile.TarFile, members: list[tarfile.TarInfo], target: Path) -> None:
    target_root = target.resolve()
    for member in members:
        member_path = (target / member.name).resolve()
        try:
            member_path.relative_to(target_root)
        except ValueError as exc:
            raise RuntimeError(f"unsafe path in NeMo source archive: {member.name}")
    archive.extractall(target, members=members)


def _single_child_dir(path: Path) -> Path:
    children = [child for child in path.iterdir() if child.is_dir()]
    if len(children) != 1:
        raise RuntimeError(f"expected one source directory in archive, found {len(children)}")
    return children[0]


if __name__ == "__main__":
    main()
