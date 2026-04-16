#!/usr/bin/env python3
"""Check that version has been bumped and PKGBUILD matches pyproject.toml."""

import re
import subprocess
import sys

try:
    import tomli as tomllib
except ImportError:
    import tomllib


def get_git_file_content(ref, path):
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else None


def get_pyproject_version(content):
    data = tomllib.loads(content)
    return data.get("project", {}).get("version")


def get_pkgbuild_version(content):
    for line in content.splitlines():
        if line.strip().startswith("pkgver="):
            return line.split("=")[1].strip()
    return None


def main():
    with open("pyproject.toml", "rb") as f:
        local_version = get_pyproject_version(f.read().decode("utf-8"))

    try:
        with open("PKGBUILD") as f:
            pkgbuild_version = get_pkgbuild_version(f.read())
        if local_version != pkgbuild_version:
            print(
                f"❌ Version mismatch: pyproject.toml={local_version} PKGBUILD={pkgbuild_version}",
                file=sys.stderr,
            )
            return 1
    except FileNotFoundError:
        pass

    current_branch = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True
    ).stdout.strip()

    if current_branch == "master":
        prev_content = get_git_file_content("HEAD^", "pyproject.toml")
        if not prev_content:
            print("✅ No previous version found")
            return 0
        prev_version = get_pyproject_version(prev_content)
    else:
        subprocess.run(["git", "fetch", "origin", "master"], capture_output=True)
        master_content = get_git_file_content("origin/master", "pyproject.toml")
        if not master_content:
            print("✅ Master version not found")
            return 0
        prev_version = get_pyproject_version(master_content)

    print(f"Current: {local_version}, Previous: {prev_version}")
    if local_version == prev_version:
        print("❌ VERSION BUMP REQUIRED", file=sys.stderr)
        return 1
    print(f"✅ Version bumped: {prev_version} → {local_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
