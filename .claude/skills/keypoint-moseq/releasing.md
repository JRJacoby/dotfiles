# Releasing keypoint-moseq

## Overview

A release is a GitHub release. Publishing one creates the git tag, and the `release` workflow (`.github/workflows/release.yml`) then builds the sdist from that tag and uploads it to PyPI through trusted publishing. Read the Docs moves its `stable` version, which is the docs site's default, to the new tag on its own.

The `tests` workflow runs the full test suite with one job per Python version: 3.10 and 3.14 on every pull request, and all of 3.10 through 3.14 on every push to main, nightly at 4am America/New_York, and on manual runs. A gate job named `tests` passes only when every version's job passed. Merging into main requires a pull request whose `tests` check passes (ruleset on main, no bypass). The workflow never builds or publishes anything.

## Versioning

Uses **versioneer**: the version is derived from git tags at build time, and there is no hardcoded version number anywhere. Tags have no prefix (`0.6.14`, not `v0.6.14`).

### Choosing the version number

The version number tells a user what changes when they upgrade. "Outputs" means anything a user could compare across versions: syllables and fitted parameters, PCA, dataframes, saved files and figures. The promise is for a fixed environment: with the same dependency versions, a patch release produces the same outputs. Dependencies are unpinned, so upstream releases can shift results at rounding level with no keypoint-moseq release at all; reproducibility comes from the user's recorded environment, not from the keypoint-moseq version.

- **Patch** (`0.7.0` → `0.7.1`): outputs unchanged for the same environment. Crash fixes, including something that errored or produced nothing and now works; documentation; performance improvements with identical output; tightening a dependency requirement to exclude a broken upstream release.
- **Minor** (`0.7.0` → `0.8.0`): anything that changes outputs, including bug fixes that change results people already have; new features; loosening, removing or raising a dependency requirement, since that changes what a fresh install gets; deprecations. While the version is 0.x, API changes and loss of the ability to open older projects (config, checkpoints, `results.h5`) are also minor, and the release notes call them out.
- **Major**, from 1.0 on: removing or renaming public functions, or older projects no longer loading.

A release that needed any stored snapshot regenerated with `--force-regen` is at least minor. Fitted syllables are not snapshotted, so changes to model fitting need judgment. The release notes for a minor release say which outputs change.

0.7.0 is the worked example: keypoint-moseq's own code changes left outputs unchanged, but removing the dependency pins means a fresh install gets NumPy 2, pandas 3 and newer jax, which changes dataframe dtypes and fitted results, so it is minor rather than 0.6.15.

## Step-by-step release process

### 1. Merge the pull requests

Each pull request merges once its `tests` check passes:

```bash
gh pr checks <PR_NUMBER>
gh pr merge <PR_NUMBER> --merge
```

### 2. Publish a GitHub release

Target the commit on main to release (usually the latest merge) and check previous releases for style:

```bash
gh release list --limit 5
gh release view <VERSION>
```

Release title format: `Keypoint MoSeq X.Y.Z`. Small releases get a one-liner. Medium releases get a summary paragraph, a "What's Changed" section with PR links, and a "Full Changelog" link:

```bash
gh release create 0.6.14 --target <MERGE_SHA> --title "Keypoint MoSeq 0.6.14" --notes "$(cat <<'EOF'
Summary paragraph here.

## What's Changed
* Feature description by @JRJacoby in https://github.com/dattalab/keypoint-moseq/pull/NNN

**Full Changelog**: https://github.com/dattalab/keypoint-moseq/compare/PREV_TAG...NEW_TAG
EOF
)"
```

Changes that do not alter the installed package (tests, CI, docs-only) do not need a release on their own. User-facing fixes get a prompt patch release.

### 3. Watch the release workflow

```bash
gh run list --workflow release.yml --limit 1
gh run watch <RUN_ID> --exit-status
```

It takes about a minute: checkout of the tag with full history, `uv build --sdist`, then `pypa/gh-action-pypi-publish`, which also uploads a signed attestation.

### 4. Verify

```bash
curl -s https://pypi.org/pypi/keypoint-moseq/<VERSION>/json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['info']['version'], [(f['filename'], f['digests']['sha256']) for f in d['urls']])"
curl -s https://pypi.org/integrity/keypoint-moseq/<VERSION>/keypoint_moseq-<VERSION>.tar.gz/provenance
curl -s https://readthedocs.org/api/v3/projects/keypoint-moseq/versions/stable/ | python3 -c "import sys,json; print(json.load(sys.stdin)['identifier'])"
```

The sha256 on PyPI should match the one printed in the workflow's attestation step, the provenance publisher should be GitHub `dattalab/keypoint-moseq` `release.yml`, and Read the Docs `stable` should point at the release commit. PyPI's project-wide `/pypi/keypoint-moseq/json` is cached and can lag the version-specific endpoint by a few minutes.

When the workflow's log is unavailable (during a GitHub Actions incident a finished run can show as queued for an hour while PyPI already has the file), rebuild the sdist from a fresh clone of the tag with `uv build --sdist` and compare the extracted contents with PyPI's file (`diff -rq` on the two unpacked trees). The sha256 will differ even when the contents are identical, because the tarball records each file's owner and modification time.

## PyPI permissions and trusted publishing

PyPI accepts uploads from the `release` workflow because the project has a trusted publisher: GitHub, owner `dattalab`, repository `keypoint-moseq`, workflow `release.yml`, no environment. No API token is involved.

PyPI has two project roles. Maintainers (John) can upload releases. Owners (Caleb) can also manage the project, including trusted publishers and collaborators. Renaming `release.yml` requires an owner to update the trusted publisher.

## If the release workflow fails

The GitHub release and tag still exist. Fix the cause and use "Re-run jobs" on the failed run, which rebuilds from the same tag. For an upload rejected as unauthorized, check the trusted publisher settings on PyPI.

A manual upload is the fallback. Build from a fresh clone of the tag so nothing from a local checkout leaks into the sdist, then upload with a PyPI API token (create one under PyPI account settings if none exists):

```bash
git clone --branch <VERSION> git@github.com:dattalab/keypoint-moseq.git <SCRATCH_DIR>/release-<VERSION>
cd <SCRATCH_DIR>/release-<VERSION>
uv build --sdist
uvx twine check dist/*
uvx twine upload dist/*
```

twine cannot prompt for credentials in a non-interactive environment such as Claude Code, so it needs the token in `~/.pypirc` (`username = __token__`, `password = pypi-...`, file mode `600`).

## Sharp edges

### PyPI filename reuse is forbidden

Once a filename has been uploaded for a version, a different file with that name cannot be uploaded, and re-uploading the same file fails with "File already exists". A wrong build means bumping the version.

### sdist filenames must be PEP 625 normalized

PyPI rejects `keypoint-moseq-X.Y.Z.tar.gz` (hyphen) and requires `keypoint_moseq-X.Y.Z.tar.gz`. `uv build` and `python -m build` produce the normalized name; `python setup.py sdist` does not. The project name on PyPI is unaffected.

### The stable docs follow releases, not main

Docs changes merged to main appear under `latest` immediately and under `stable` (the default) only after the next release.

## Related files

- `.github/workflows/release.yml`: builds and publishes on a published GitHub release
- `.github/workflows/tests.yml`: the test suite (pull requests, main, nightly)
- `setup.py`, `setup.cfg`, `versioneer.py`: packaging and version from tags
- `.readthedocs.yml`: docs build config
