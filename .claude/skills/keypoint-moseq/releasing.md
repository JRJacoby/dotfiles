# Releasing keypoint-moseq

## Overview

Releases are fully manual — no CI/CD. The process involves merging to main, creating a GitHub release (which creates a git tag), building an sdist, and uploading to PyPI with twine.

## Repo location

The repo is at `/n/groups/datta/john/repos/keypoint-moseq`.

## Versioning

Uses **versioneer** — the version is derived automatically from git tags. There is no hardcoded version number anywhere. When you create a git tag (e.g. via a GitHub release), versioneer picks it up at build time.

Tags have no prefix (e.g. `0.6.8`, not `v0.6.8`).

## Step-by-step release process

### 1. Merge PR to main

```bash
gh pr merge <PR_NUMBER> --merge
```

### 2. Create a GitHub release

This creates the git tag and a release page in one step. Check previous releases for style:

```bash
gh release list --limit 5   # see recent releases for style reference
gh release view <VERSION>    # see a specific release's notes
```

Release title format: `Keypoint MoSeq X.Y.Z`

Small releases get a one-liner. Medium releases get a summary paragraph + "What's Changed" section with PR links + "Full Changelog" link. See examples:

```bash
# Small (one-liner)
gh release create 0.6.8 --title "Keypoint MoSeq 0.6.8" --notes "Brief description. commit_hash"

# Medium (with PR links)
gh release create 0.6.8 --title "Keypoint MoSeq 0.6.8" --notes "$(cat <<'EOF'
Summary paragraph here.

## What's Changed
* Feature description by @JRJacoby in https://github.com/dattalab/keypoint-moseq/pull/NNN

**Full Changelog**: https://github.com/dattalab/keypoint-moseq/compare/PREV_TAG...NEW_TAG
EOF
)"
```

### 3. Pull the tag locally and build

```bash
cd /n/groups/datta/john/repos/keypoint-moseq
git checkout main
git pull --tags
python -c "import versioneer; print(versioneer.get_version())"  # verify version
```

### 4. Build and upload to PyPI

There is an `update_pypi.sh` script in the repo root (untracked). As of Feb 2026 it should contain:

```bash
rm -rf dist/
python -m build --sdist
twine upload dist/*
```

**IMPORTANT**: `python -m build` may not be installed in the system Python. Use `uvx`:

```bash
rm -rf dist/
uvx --from build pyproject-build --sdist
uvx twine upload dist/*
```

### 5. Verify

```bash
curl -s https://pypi.org/pypi/keypoint-moseq/json | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['version'])"
```

## PyPI credentials

The PyPI API token is stored in `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-...
```

File permissions must be `600`. If the file doesn't exist, ask John for the token and create it:

```bash
chmod 600 ~/.pypirc
```

## Sharp edges

### PyPI now requires PEP 625 normalized filenames (as of Oct 23, 2025)

The old `python setup.py sdist` produces `keypoint-moseq-X.Y.Z.tar.gz` (hyphens). PyPI now rejects this — it requires `keypoint_moseq-X.Y.Z.tar.gz` (underscores).

**Use `python -m build --sdist`** (or `uvx --from build pyproject-build --sdist`) instead of `python setup.py sdist`. This produces the correct filename.

The display name on PyPI is unaffected — `keypoint-moseq` and `keypoint_moseq` are treated as the same project. This is only about the tarball filename.

Background: PyPI merged [warehouse PR #18924](https://github.com/pypi/warehouse/pull/18924) on Oct 23, 2025, enforcing [PEP 625](https://peps.python.org/pep-0625/) sdist filename normalization. The 0.6.7 release on that same day was the last to use the old format.

### PyPI filename reuse is forbidden

Once a filename+hash has been uploaded to PyPI for a given version, you cannot upload a different file with the same name. If a build goes through but you realize it was wrong, you must bump the version number. You also cannot re-upload the exact same file — PyPI will say "File already exists."

### twine is not installed globally

`twine` is not in the system PATH. Use `uvx twine` to run it without installing. Same for `build` — use `uvx --from build pyproject-build`.

### Non-interactive environment

twine cannot prompt for credentials in non-interactive environments (like Claude Code). It needs `~/.pypirc` to be set up in advance. The keyring/secretstorage backend will fail on headless servers — that's fine as long as `.pypirc` exists.

### Docs auto-update on merge

ReadTheDocs is configured to build automatically when main is updated. No manual step needed for docs — just merge the PR.

## Related files

- `update_pypi.sh` — build+upload script (untracked, lives in repo root)
- `setup.py` — minimal, delegates to versioneer
- `setup.cfg` — package metadata, dependencies, versioneer config
- `versioneer.py` — version management from git tags
- `.readthedocs.yml` — docs build config
