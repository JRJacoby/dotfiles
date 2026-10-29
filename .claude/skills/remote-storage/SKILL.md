---
name: remote-storage
description: Move files to and from remote storage — Google Drive, Box, OneDrive via rclone; Google Cloud Storage via gsutil; S3 via aws. Use when asked to upload, download, share, sync, or fetch data from any cloud location, including bare phrasings like "upload this to temp" or "put it on Drive".
---

# Remote Storage

Three tools cover everything: **rclone** for consumer cloud drives, **gsutil** for
Google Cloud Storage buckets, **aws** for S3.

## "Upload to temp" means Google Drive `work/temp`

When the user says *upload to temp*, *drop it in temp*, *put this in /work/temp*,
or names `/work/temp` with no remote, they mean:

```bash
rclone copy -v <local-file> drive:work/temp/
```

This is a scratch/handoff folder on their Google Drive. Do not create a README or
notes file alongside the upload — ship exactly the artifacts named.

Confirm the destination only if the phrasing is genuinely ambiguous (a different
remote is named, or the path does not look like `work/temp`).

## rclone

Binary at `~/bin/rclone`. Three remotes are configured:

| Remote | Service | Notes |
|---|---|---|
| `drive:` | Google Drive | the default target; holds `work/temp` |
| `box:` | Box | token expires often, see below |
| `onedrive:` | OneDrive | |

### Core commands

```bash
rclone copy -v LOCAL remote:path/      # copy, never deletes
rclone copy -v remote:path/file .      # download
rclone lsd remote:path                 # list directories only
rclone lsl remote:path                 # list files with size + mtime
rclone ls  remote:path                 # list files, size + name
rclone size remote:path                # total bytes and object count
rclone mkdir remote:path/new
```

**Prefer `copy` over `sync`.** `rclone sync` makes the destination match the
source, which means it **deletes** files at the destination that are not in the
source. Use it only when the user explicitly asks to mirror, and pass
`--dry-run` first.

### Options worth knowing

- `-v` / `-vv` — progress and per-file logging. `-P` gives a live transfer bar,
  which is noisy in captured output; prefer `-v` when redirecting to a log.
- `--dry-run` — mandatory rehearsal for anything with `sync` or `delete`.
- `--transfers N` (default 4) and `--checkers N` (default 8) — raise for many
  small files, lower if the remote rate-limits.
- `--drive-shared-with-me` — Google Drive items shared *with* the user do not
  appear in the normal tree; this flag switches to that namespace.
- `--max-age 7d`, `--include '*.pdf'`, `--exclude '*.tmp'` — filtering.
- `--progress-terminal-title` and interactive flags are pointless in captured
  output; skip them.

### Expired tokens

Any remote can return `couldn't fetch token: invalid_grant: maybe token expired?`
(Box does this frequently). Fix:

```bash
rclone config reconnect box:
```

**This opens a browser and that works fine — do not reach for a two-machine
token-paste workaround, `--no-browser`, or an ssh tunnel.** The user's machine
has a real browser and localhost ports are reachable. Answer *yes* to the
auto-authenticate prompt. If the user is on O2, the same applies: it has a
desktop environment and a working browser.

### Google Drive quirks

- `rclone lsd drive:` may print `NOTICE: Dangling shortcut "..."`. Harmless.
- Listing the Drive root is slow and long; filter with `rg` rather than paging.
- Google-native files (Docs, Sheets) are exported on download, not copied
  byte-for-byte; use `--drive-export-formats pdf,docx` when that matters.

## gsutil / gcloud — Google Cloud Storage

Installed under a user-local Google Cloud SDK, so it may not be on a bare
`PATH`; `command -v gsutil` first.

```bash
gsutil ls gs://bucket/                 # list a prefix
gsutil ls -l gs://bucket/prefix/       # with sizes and mtimes
gsutil ls -lh "gs://bucket/*.mp4"      # human-readable, globbed
gsutil du -sh gs://bucket/prefix       # total size of a prefix
gsutil cp gs://bucket/obj ./local      # download one object
gsutil -m cp -r gs://bucket/prefix ./  # parallel recursive download
gsutil cat gs://bucket/small.csv       # stream without downloading
```

- **`-m` is the big lever** for many objects; it parallelizes and turns an hour
  into minutes. Always use it for recursive transfers.
- **Bucket *listing* at the project level is often denied** even when object
  access works: a bare `gsutil ls` returns nothing while
  `gsutil ls gs://known-bucket/` works fine. This is normal IAM scoping, not a
  broken install. It means you cannot discover buckets — ask the user for the
  bucket name.
- To survey a bucket's layout cheaply, list one level at a time with a trailing
  glob (`gsutil ls "gs://bucket/*/"`) rather than a full recursive `-r`, which
  can return millions of lines.
- Check identity and project with `gcloud config list`. Re-auth with
  `gcloud auth login` — again, the normal browser flow, no device-code variant.

## aws — S3

Binary at `~/bin/aws` (AWS CLI v2). Confirm identity with
`aws sts get-caller-identity`.

```bash
aws s3 ls s3://bucket/prefix/
aws s3 cp local s3://bucket/prefix/
aws s3 cp s3://bucket/prefix/obj ./local
aws s3 sync ./dir s3://bucket/prefix/   # deletes only with --delete
aws s3 ls --summarize --human-readable --recursive s3://bucket/prefix/
```

- `aws s3 sync` does **not** delete by default (unlike `rclone sync`); it deletes
  only with an explicit `--delete`. The two tools' `sync` semantics differ — do
  not carry an assumption from one to the other.
- `--profile NAME` when more than one profile is configured; check with
  `aws configure list-profiles`.
- `--dryrun` (one word, no hyphen) is the rehearsal flag, unlike rclone's
  `--dry-run`.

## Choosing a tool

| Destination | Tool |
|---|---|
| Google Drive, Box, OneDrive, Dropbox | `rclone` |
| `gs://` bucket | `gsutil` (rclone can do GCS but the native tool is faster and better authenticated) |
| `s3://` bucket | `aws s3` |
| Sharing a file with a person | `rclone` to Drive — see the temp convention above |

## General cautions

- Uploading publishes data to a third party. It may be cached or indexed even
  after deletion. Confirm before uploading anything the user has not explicitly
  pointed at, and never upload a file you have not read.
- Redirect verbose transfers to a log file rather than piping through `tail`, so
  a failure is diagnosable.
- Verify after uploading (`rclone lsl`, `gsutil ls -l`, `aws s3 ls`) rather than
  trusting exit code 0 alone.
