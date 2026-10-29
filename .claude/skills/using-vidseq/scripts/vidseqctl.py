#!/usr/bin/env python3
"""Safe VidSeq project setup and read-only monitoring helper."""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import os
import socket
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = os.environ.get("VIDSEQ_URL", "http://127.0.0.1:8000")
ARRAY_FILES = (
    "tracker_masks.h5",
    "tracker_logits.h5",
    "detector_masks.h5",
    "final_masks.h5",
    "cropped_masks.h5",
    "aligned_masks.h5",
    "alignment_keypoints.h5",
    "pca_scores.h5",
)
MIN_HDF5_CONTAINER_BYTES = 512


class VidSeqCtlError(RuntimeError):
    pass


def dump(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str), flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def api_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/api"):
        return root
    return root + "/api"


def request_json(
    base_url: str,
    method: str,
    endpoint: str,
    body: Any | None = None,
    timeout: float | None = 30.0,
) -> Any:
    url = api_root(base_url) + "/" + endpoint.lstrip("/")
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise VidSeqCtlError(f"VidSeq API {method} {url} returned {error.code}: {detail}") from error
    except URLError as error:
        raise VidSeqCtlError(f"cannot reach VidSeq API at {url}: {error.reason}") from error
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        raise VidSeqCtlError(f"VidSeq API returned non-JSON data from {url}") from error


def canonical_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise VidSeqCtlError(f"video paths must be absolute: {raw}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise VidSeqCtlError(f"path does not exist: {path}") from error
    if not resolved.is_file():
        raise VidSeqCtlError(f"path is not a regular file: {resolved}")
    return resolved


def validate_project_name(name: str) -> str:
    if not name or not name.strip():
        raise VidSeqCtlError("project name must not be empty or whitespace")
    if name in (".", "..") or Path(name).name != name or Path(name).is_absolute():
        raise VidSeqCtlError(f"project name must be one simple directory name: {name!r}")
    if "\x00" in name or "/" in name or "\\" in name:
        raise VidSeqCtlError(f"project name contains a path separator or NUL: {name!r}")
    if len(name.encode("utf-8")) > 240:
        raise VidSeqCtlError("project name is too long (maximum 240 UTF-8 bytes)")
    return name


def validate_missing_tolerance(
    frame_counts: Iterable[int],
    allow_missing: int,
    allow_large_missing: bool,
) -> None:
    counts = [int(value) for value in frame_counts]
    if any(allow_missing >= count for count in counts):
        raise VidSeqCtlError("--allow-missing must be smaller than every selected video's frame count")
    safe_limits = [min(10, max(1, int(count * 0.001))) for count in counts]
    if not allow_large_missing and any(allow_missing > limit for limit in safe_limits):
        raise VidSeqCtlError(
            f"--allow-missing={allow_missing} exceeds the conservative per-video limits "
            f"{safe_limits}; investigate first or add --allow-large-missing explicitly"
        )


@contextmanager
def project_apply_lock(project_dir: Path, payload: dict[str, Any]):
    """Hold an atomic advisory lock across one helper-driven apply sequence."""
    if not project_dir.is_dir():
        raise VidSeqCtlError(f"project directory is not accessible from this client: {project_dir}")
    lock_path = project_dir / ".vidseqctl-apply.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        try:
            existing = lock_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            existing = "<unreadable>"
        raise VidSeqCtlError(
            f"another helper apply may be active; lock exists at {lock_path}: {existing}. "
            "Never remove it until the recorded process/job and backend request are proven dead."
        ) from error
    inode = os.fstat(descriptor).st_ino
    try:
        record = {
            **payload,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": utc_now(),
        }
        os.write(descriptor, (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(descriptor)
        yield lock_path
    finally:
        os.close(descriptor)
        try:
            if lock_path.stat().st_ino == inode:
                lock_path.unlink()
        except FileNotFoundError:
            pass


def read_path_list(path: Path) -> list[str]:
    if not path.is_file():
        raise VidSeqCtlError(f"path-list file does not exist: {path}")
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            result.append(value)
    return result


def collect_video_paths(args: argparse.Namespace) -> list[Path]:
    raw: list[str] = list(args.paths)
    for pattern in args.glob_patterns:
        matches = sorted(glob_module.glob(pattern, recursive=True))
        if not matches:
            raise VidSeqCtlError(f"glob matched no files: {pattern}")
        raw.extend(matches)
    for list_path in args.path_lists:
        raw.extend(read_path_list(Path(list_path).expanduser()))
    if not raw:
        raise VidSeqCtlError("provide video paths, --glob, or --path-list")
    result: list[Path] = []
    seen: set[str] = set()
    for value in raw:
        path = canonical_path(value)
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def resolve_video_ids(args: argparse.Namespace) -> tuple[list[int], list[dict[str, Any]]]:
    videos = request_json(args.base_url, "GET", f"projects/{args.project_id}/videos")
    available = {int(video["id"]): video for video in videos}
    if args.all_videos:
        ids = sorted(available)
    else:
        ids = sorted(set(args.video_ids))
    if not ids:
        raise VidSeqCtlError("select --all-videos or at least one --video-id")
    unknown = [video_id for video_id in ids if video_id not in available]
    if unknown:
        raise VidSeqCtlError(f"unknown project video IDs: {unknown}")
    return ids, [available[video_id] for video_id in ids]


def command_list_projects(args: argparse.Namespace) -> None:
    dump(request_json(args.base_url, "GET", "projects"))


def command_create_project(args: argparse.Namespace) -> None:
    validate_project_name(args.name)
    parent = Path(args.parent).expanduser()
    if not parent.is_absolute():
        raise VidSeqCtlError(f"project parent must be absolute: {parent}")
    if not parent.is_dir():
        raise VidSeqCtlError(f"project parent is not an existing directory: {parent}")
    parent = parent.resolve()
    target = parent / args.name
    projects = request_json(args.base_url, "GET", "projects")
    collisions = [
        project for project in projects
        if project.get("name") == args.name or Path(project.get("path", "")) == target
    ]
    plan = {
        "action": "create-project",
        "execute": args.execute,
        "request": {"name": args.name, "path": str(parent)},
        "target": str(target),
        "registry_collisions": collisions,
        "target_exists": target.exists(),
    }
    if collisions or target.exists():
        dump(plan)
        raise VidSeqCtlError("project name/path collision; inspect instead of creating over it")
    if not args.execute:
        dump(plan)
        return
    try:
        plan["result"] = request_json(
            args.base_url, "POST", "projects", {"name": args.name, "path": str(parent)}
        )
    except VidSeqCtlError as error:
        plan["api_error"] = str(error)
        plan["target_exists_after_error"] = target.exists()
        try:
            plan["registry_after_error"] = request_json(args.base_url, "GET", "projects")
        except VidSeqCtlError as followup_error:
            plan["registry_after_error"] = {"unavailable": str(followup_error)}
        dump(plan)
        raise VidSeqCtlError(
            "project creation failed; inspect the target directory and registry before retrying"
        ) from error
    dump(plan)


def command_list_videos(args: argparse.Namespace) -> None:
    dump(request_json(args.base_url, "GET", f"projects/{args.project_id}/videos"))


def import_state(base_url: str, project_id: int, video_ids: Iterable[int] | None = None) -> dict[str, Any]:
    project = request_json(base_url, "GET", f"projects/{project_id}")
    project_dir = Path(project["path"])
    videos = request_json(base_url, "GET", f"projects/{project_id}/videos")
    selected = set(video_ids or [])
    if selected:
        videos = [video for video in videos if int(video["id"]) in selected]
    results = []
    for video in videos:
        video_dir = project_dir / "array_data" / str(video["id"])
        arrays = [
            file_metadata(video_dir / name)
            for name in ARRAY_FILES[:4]
        ]
        item = {
            "id": int(video["id"]),
            "name": video["name"],
            "source_path": video["path"],
            "source_exists": Path(video["path"]).is_file(),
            "metadata_valid": all(float(video[key]) > 0 for key in ("fps", "width", "height", "num_frames")),
            "arrays": arrays,
        }
        item["complete"] = (
            item["source_exists"]
            and item["metadata_valid"]
            and all(
                array["exists"]
                and not array.get("locked", False)
                and int(array.get("bytes", 0)) >= MIN_HDF5_CONTAINER_BYTES
                for array in arrays
            )
        )
        results.append(item)
    return {
        "project_id": project_id,
        "project_dir": str(project_dir),
        "video_count": len(results),
        "complete": bool(results) and all(item["complete"] for item in results),
        "videos": results,
        "note": (
            f"Checks source metadata plus base HDF5 presence, locks, and minimum "
            f"container size ({MIN_HDF5_CONTAINER_BYTES} bytes) without opening datasets."
        ),
    }


def command_verify_import(args: argparse.Namespace) -> None:
    state = import_state(args.base_url, args.project_id, args.video_ids or None)
    dump(state)
    if not state["complete"]:
        raise VidSeqCtlError("one or more imported videos failed the metadata/array-container postcheck")


def command_add_videos(args: argparse.Namespace) -> None:
    candidates = collect_video_paths(args)
    existing = request_json(args.base_url, "GET", f"projects/{args.project_id}/videos")
    existing_by_canonical: dict[str, dict[str, Any]] = {}
    for video in existing:
        try:
            key = str(Path(video["path"]).expanduser().resolve(strict=False))
        except (KeyError, OSError):
            continue
        existing_by_canonical[key] = video
    add = [path for path in candidates if str(path) not in existing_by_canonical]
    skipped = [
        {"path": str(path), "existing": existing_by_canonical[str(path)]}
        for path in candidates if str(path) in existing_by_canonical
    ]
    by_basename: dict[str, set[str]] = {}
    for path in candidates:
        by_basename.setdefault(path.name, set()).add(str(path))
    for video in existing:
        existing_path = Path(video["path"]).expanduser().resolve(strict=False)
        by_basename.setdefault(existing_path.name, set()).add(str(existing_path))
    basename_collisions = {
        name: sorted(paths) for name, paths in by_basename.items() if len(paths) > 1
    }
    plan: dict[str, Any] = {
        "action": "add-videos",
        "project_id": args.project_id,
        "execute": args.execute,
        "candidate_count": len(candidates),
        "add_count": len(add),
        "paths_to_add": [str(path) for path in add],
        "already_present": skipped,
        "basename_collisions": basename_collisions,
    }
    if basename_collisions and not args.allow_basename_collisions:
        dump(plan)
        raise VidSeqCtlError(
            "distinct videos share a basename; resolve the manifest ambiguity or pass "
            "--allow-basename-collisions after review"
        )
    if not args.execute or not add:
        dump(plan)
        return
    try:
        result = request_json(
            args.base_url,
            "POST",
            f"projects/{args.project_id}/videos",
            {"paths": [str(path) for path in add]},
            timeout=None if args.timeout == 0 else args.timeout,
        )
    except VidSeqCtlError as error:
        # VidSeq commits video rows before it creates the HDF5 containers. A
        # post-commit container failure can therefore return HTTP 500 while
        # leaving rows behind; surface their state before anyone retries.
        try:
            after = request_json(args.base_url, "GET", f"projects/{args.project_id}/videos")
        except VidSeqCtlError as followup_error:
            after = []
            plan["post_error_video_listing"] = {"unavailable": str(followup_error)}
        wanted = {str(path) for path in add}
        matching_ids = [
            int(video["id"])
            for video in after
            if str(Path(video["path"]).expanduser().resolve(strict=False)) in wanted
        ]
        plan["api_error"] = str(error)
        plan["matching_rows_after_error"] = matching_ids
        if matching_ids:
            plan["postcheck_after_error"] = import_state(
                args.base_url, args.project_id, matching_ids
            )
        dump(plan)
        raise VidSeqCtlError(
            "video import failed; inspect matching_rows_after_error/postcheck_after_error before retrying"
        ) from error
    plan["result"] = result
    plan["postcheck"] = import_state(
        args.base_url, args.project_id, [int(video["id"]) for video in result]
    )
    dump(plan)
    if not plan["postcheck"]["complete"]:
        raise VidSeqCtlError(
            "API returned after video rows were committed, but one or more array containers failed postcheck; "
            "inspect before retrying because a retry can create duplicate video rows"
        )


def command_worker_status(args: argparse.Namespace) -> None:
    dump(request_json(args.base_url, "GET", "segmentation/status"))


def command_load_worker(args: argparse.Namespace) -> None:
    current = request_json(args.base_url, "GET", "segmentation/status")
    plan: dict[str, Any] = {
        "action": "load-segmentation-worker",
        "execute": args.execute,
        "current_status": current,
    }
    if not args.execute:
        dump(plan)
        return
    if current.get("status") not in ("ready", "loading_model"):
        plan["result"] = request_json(args.base_url, "POST", "segmentation/loaded-model")
    dump(plan)
    if not args.wait or current.get("status") == "ready":
        return
    started = time.monotonic()
    last: Any = object()
    while True:
        status = request_json(args.base_url, "GET", "segmentation/status")
        if status != last:
            dump({"at": utc_now(), "worker_status": status})
            last = status
        if status.get("status") == "ready":
            return
        if status.get("status") == "error":
            raise VidSeqCtlError(f"segmentation worker failed to load: {status}")
        if args.timeout and time.monotonic() - started >= args.timeout:
            raise VidSeqCtlError(f"worker did not become ready within {args.timeout} seconds")
        time.sleep(args.interval)


def training_paths(project_id: int, kind: str) -> tuple[str, str]:
    if kind == "bbox":
        return (
            f"projects/{project_id}/detection/status",
            f"projects/{project_id}/detection/training",
        )
    return (
        f"projects/{project_id}/detection/seg/status",
        f"projects/{project_id}/detection/seg/training",
    )


def command_detector_status(args: argparse.Namespace) -> None:
    status_path, progress_path = training_paths(args.project_id, args.kind)
    dump({
        "kind": args.kind,
        "model_status": request_json(args.base_url, "GET", status_path),
        "training_progress": request_json(args.base_url, "GET", progress_path),
    })


def command_start_training(args: argparse.Namespace) -> None:
    ids, videos = resolve_video_ids(args)
    total_training_frames = sum(int(video.get("training_frame_count") or 0) for video in videos)
    status_path, training_path = training_paths(args.project_id, args.kind)
    plan: dict[str, Any] = {
        "action": "start-training",
        "execute": args.execute,
        "kind": args.kind,
        "project_id": args.project_id,
        "video_ids": ids,
        "training_frames_reported": total_training_frames,
        "videos_without_training_frames": [
            int(video["id"]) for video in videos if not video.get("training_frame_count")
        ],
        "current_status": request_json(args.base_url, "GET", status_path),
        "current_progress": request_json(args.base_url, "GET", training_path),
        "request": {
            "video_ids": ids,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "early_stop_patience": args.early_stop_patience,
            "from_checkpoint": args.from_checkpoint,
        },
    }
    if total_training_frames <= 0:
        dump(plan)
        raise VidSeqCtlError("selected videos report no training frames; wait for GUI labeling")
    if plan["videos_without_training_frames"] and not args.allow_unlabeled_videos:
        dump(plan)
        raise VidSeqCtlError(
            "one or more selected videos have no training frames; select only labeled video IDs "
            "or pass --allow-unlabeled-videos after review"
        )
    if not args.execute:
        dump(plan)
        return
    if (
        plan["current_status"].get("is_training")
        or plan["current_progress"].get("is_training")
        or plan["current_progress"].get("status") in ("training", "applying")
    ):
        dump(plan)
        raise VidSeqCtlError("a detector training/apply operation is already active in this backend process")
    if args.kind == "bbox" and args.detector_type:
        plan["detector_config"] = request_json(
            args.base_url,
            "PUT",
            f"projects/{args.project_id}/detection/config",
            {"detector_type": args.detector_type},
        )
    plan["result"] = request_json(
        args.base_url, "POST", training_path, plan["request"]
    )
    dump(plan)


def command_wait_training(args: argparse.Namespace) -> None:
    _, progress_path = training_paths(args.project_id, args.kind)
    started = time.monotonic()
    last: Any = object()
    observed_active = False
    while True:
        progress = request_json(args.base_url, "GET", progress_path)
        if progress != last:
            dump({"at": utc_now(), "kind": args.kind, "progress": progress})
            last = progress
        status = progress.get("status")
        is_training = bool(progress.get("is_training"))
        observed_active = observed_active or is_training or status in ("training", "applying")
        # The service can leave is_training stale after an exception. Terminal
        # status is authoritative.
        if status == "completed":
            if observed_active or args.accept_existing_terminal:
                return
            raise VidSeqCtlError(
                "the first observed status was already completed, so it may belong to a stale run; "
                "inspect model/log timestamps or pass --accept-existing-terminal"
            )
        if status in ("failed", "stopped"):
            raise VidSeqCtlError(f"training ended with status {status}: {progress}")
        if status == "idle" and not is_training:
            if observed_active:
                raise VidSeqCtlError("training became idle without a terminal result; backend may have restarted")
            raise VidSeqCtlError("training is idle; no active run was observed")
        if args.timeout and time.monotonic() - started >= args.timeout:
            raise VidSeqCtlError(f"timed out after {args.timeout} seconds")
        time.sleep(args.interval)


def command_apply(args: argparse.Namespace) -> None:
    ids, videos = resolve_video_ids(args)
    validate_missing_tolerance(
        (int(video["num_frames"]) for video in videos),
        args.allow_missing,
        args.allow_large_missing,
    )
    project = request_json(args.base_url, "GET", f"projects/{args.project_id}")
    project_dir = Path(project["path"])
    status_path, progress_path = training_paths(args.project_id, args.kind)
    model_status = request_json(args.base_url, "GET", status_path)
    current_progress = request_json(args.base_url, "GET", progress_path)
    worker_status = request_json(args.base_url, "GET", "segmentation/status")
    endpoint = (
        f"projects/{args.project_id}/videos/detection"
        if args.kind == "bbox"
        else f"projects/{args.project_id}/videos/seg-detection"
    )
    existing_h5_locks = [
        str(project_dir / "array_data" / str(video_id) / "detector_masks.h5.lock")
        for video_id in ids
        if (project_dir / "array_data" / str(video_id) / "detector_masks.h5.lock").exists()
    ]
    helper_lock_path = project_dir / ".vidseqctl-apply.lock"
    plan: dict[str, Any] = {
        "action": "apply-detector",
        "execute": args.execute,
        "blocking_request": True,
        "kind": args.kind,
        "project_id": args.project_id,
        "video_ids": ids,
        "request_mode": "one-video-per-request",
        "model_status": model_status,
        "current_progress": current_progress,
        "worker_status": worker_status,
        "project_dir": str(project_dir),
        "allow_missing": args.allow_missing,
        "allow_large_missing": args.allow_large_missing,
        "helper_apply_lock": str(helper_lock_path),
        "helper_apply_lock_exists": helper_lock_path.exists(),
        "existing_detector_h5_locks": existing_h5_locks,
        "external_apply_caveat": (
            "The helper lock excludes other vidseqctl applies, but VidSeq has no API that can "
            "prove a GUI bbox apply is absent; confirm the live request/log before execution."
        ),
    }
    if not args.execute:
        dump(plan)
        return
    if not model_status.get("model_exists"):
        dump(plan)
        raise VidSeqCtlError(f"no trained {args.kind} detector model exists in this project")
    if (
        model_status.get("is_training")
        or current_progress.get("is_training")
        or current_progress.get("status") in ("training", "applying")
    ):
        dump(plan)
        raise VidSeqCtlError("a detector training/apply operation is already active in this backend process")
    if worker_status.get("status") != "ready":
        dump(plan)
        raise VidSeqCtlError(
            "segmentation GPU worker is not ready; load it in the GUI or POST "
            "/api/segmentation/loaded-model and wait for status=ready"
        )
    if existing_h5_locks:
        dump(plan)
        raise VidSeqCtlError("one or more selected detector HDF5 files are already write-locked")
    lock_payload = {
        "action": "apply-detector",
        "kind": args.kind,
        "project_id": args.project_id,
        "video_ids": ids,
        "base_url": api_root(args.base_url),
    }
    with project_apply_lock(project_dir, lock_payload):
        completed: list[dict[str, Any]] = []
        for position, video_id in enumerate(ids, start=1):
            dump({
                **plan,
                "status": "request-starting",
                "at": utc_now(),
                "position": position,
                "total": len(ids),
                "current_video_id": video_id,
            })
            result = request_json(
                args.base_url,
                "POST",
                endpoint,
                {"video_ids": [video_id]},
                timeout=None if args.timeout == 0 else args.timeout,
            )
            item = {"video_id": video_id, "result": result, "finished_at": utc_now()}
            db_path = project_dir / "vidseq.db"
            if db_path.is_file():
                item["postcheck"] = detector_postcheck(
                    project_dir, video_id, args.kind, args.allow_missing, args.busy_timeout
                )
            else:
                item["postcheck"] = {
                    "available": False,
                    "reason": f"project database is not accessible from this client: {db_path}",
                }
            completed.append(item)
            dump({**plan, "status": "video-finished", "position": position, "total": len(ids), **item})
            if item["postcheck"].get("available") and not item["postcheck"].get("complete"):
                raise VidSeqCtlError(
                    f"video {video_id} returned from apply but failed the local database/HDF5 postcheck"
                )
        dump({**plan, "status": "all-finished", "at": utc_now(), "completed": completed})


def readonly_connection(db_path: Path, busy_timeout: float) -> sqlite3.Connection:
    if not db_path.is_file():
        raise VidSeqCtlError(f"VidSeq database does not exist: {db_path}")
    uri = f"file:{quote(str(db_path.resolve()), safe='/')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=busy_timeout)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout={max(1, round(busy_timeout * 1000))}")
    return connection


def aggregate_rows(connection: sqlite3.Connection, video_ids: Iterable[int] | None = None) -> list[dict[str, Any]]:
    ids = list(video_ids or [])
    where = ""
    parameters: list[Any] = []
    if ids:
        where = "WHERE v.id IN (" + ",".join("?" for _ in ids) + ")"
        parameters.extend(ids)
    query = f"""
        SELECT v.id, v.name, v.path, v.fps, v.width, v.height, v.num_frames,
               v.segmentation_status,
               COUNT(fd.id) AS frame_rows,
               COALESCE(SUM(CASE WHEN fd.frame_type='train' THEN 1 ELSE 0 END), 0) AS training_frames,
               COALESCE(SUM(CASE WHEN fd.frame_type='apply' THEN 1 ELSE 0 END), 0) AS apply_frames,
               COALESCE(SUM(CASE WHEN fd.bbox_x1 IS NOT NULL AND fd.bbox_y1 IS NOT NULL
                                      AND fd.bbox_x2 IS NOT NULL AND fd.bbox_y2 IS NOT NULL
                                 THEN 1 ELSE 0 END), 0) AS tracker_bbox_frames,
               COALESCE(SUM(CASE WHEN fd.frame_idx >= 0 AND fd.frame_idx < v.num_frames
                                      AND fd.detector_score >= 0 THEN 1 ELSE 0 END), 0) AS detector_scored_frames,
               COALESCE(SUM(CASE WHEN fd.frame_idx >= 0 AND fd.frame_idx < v.num_frames
                                      AND fd.detector_score > 0 THEN 1 ELSE 0 END), 0) AS detector_positive_frames,
               COALESCE(SUM(CASE WHEN fd.frame_idx >= 0 AND fd.frame_idx < v.num_frames
                                      AND fd.detector_bbox_x1 IS NOT NULL AND fd.detector_bbox_y1 IS NOT NULL
                                      AND fd.detector_bbox_x2 IS NOT NULL AND fd.detector_bbox_y2 IS NOT NULL
                                 THEN 1 ELSE 0 END), 0) AS detector_bbox_frames,
               COALESCE(SUM(CASE WHEN fd.frame_idx >= 0 AND fd.frame_idx < v.num_frames
                                      AND fd.detector_score > 0
                                      AND (fd.detector_bbox_x1 IS NULL OR fd.detector_bbox_y1 IS NULL
                                           OR fd.detector_bbox_x2 IS NULL OR fd.detector_bbox_y2 IS NULL)
                                 THEN 1 ELSE 0 END), 0) AS positive_without_complete_bbox,
               COALESCE(SUM(CASE WHEN fd.frame_idx >= 0 AND fd.frame_idx < v.num_frames
                                      AND fd.detector_score <= 0
                                      AND (fd.detector_bbox_x1 IS NOT NULL OR fd.detector_bbox_y1 IS NOT NULL
                                           OR fd.detector_bbox_x2 IS NOT NULL OR fd.detector_bbox_y2 IS NOT NULL)
                                 THEN 1 ELSE 0 END), 0) AS nonpositive_with_any_bbox,
               COALESCE(SUM(CASE WHEN fd.frame_idx >= 0 AND fd.frame_idx < v.num_frames
                                      AND fd.detector_score > 0
                                      AND COALESCE(fd.has_detector_mask, 0) != 1
                                 THEN 1 ELSE 0 END), 0) AS positive_without_mask,
               COALESCE(SUM(CASE WHEN fd.frame_idx >= 0 AND fd.frame_idx < v.num_frames
                                      AND fd.detector_score <= 0 AND fd.has_detector_mask=1
                                 THEN 1 ELSE 0 END), 0) AS nonpositive_with_mask,
               COALESCE(SUM(CASE WHEN (fd.frame_idx < 0 OR fd.frame_idx >= v.num_frames)
                                      AND (fd.detector_score >= 0
                                           OR fd.detector_bbox_x1 IS NOT NULL
                                           OR fd.detector_bbox_y1 IS NOT NULL
                                           OR fd.detector_bbox_x2 IS NOT NULL
                                           OR fd.detector_bbox_y2 IS NOT NULL
                                           OR fd.has_detector_mask=1)
                                 THEN 1 ELSE 0 END), 0) AS detector_active_out_of_range,
               COALESCE(SUM(CASE WHEN fd.has_tracker_mask=1 THEN 1 ELSE 0 END), 0) AS tracker_mask_flags,
               COALESCE(SUM(CASE WHEN fd.frame_idx >= 0 AND fd.frame_idx < v.num_frames
                                      AND fd.has_detector_mask=1 THEN 1 ELSE 0 END), 0) AS detector_mask_flags,
               COALESCE(SUM(CASE WHEN fd.has_final_mask=1 THEN 1 ELSE 0 END), 0) AS final_mask_flags
        FROM videos AS v
        LEFT JOIN frame_data AS fd ON fd.video_id=v.id
        {where}
        GROUP BY v.id
        ORDER BY v.id
    """
    rows = [dict(row) for row in connection.execute(query, parameters)]
    for row in rows:
        frames = int(row["num_frames"] or 0)
        scored = int(row["detector_scored_frames"] or 0)
        row["detector_score_coverage"] = (scored / frames) if frames else None
        row["source_exists"] = Path(row["path"]).is_file()
    return rows


def detector_postcheck(
    project_dir: Path,
    video_id: int,
    kind: str,
    allow_missing: int,
    busy_timeout: float,
) -> dict[str, Any]:
    with readonly_connection(project_dir / "vidseq.db", busy_timeout) as connection:
        rows = aggregate_rows(connection, [video_id])
    if len(rows) != 1:
        return {
            "available": True,
            "complete": False,
            "reason": f"expected one video row, found {len(rows)}",
        }
    row = rows[0]
    frames = int(row["num_frames"])
    required = max(0, frames - allow_missing)
    lock_path = project_dir / "array_data" / str(video_id) / "detector_masks.h5.lock"
    checks = {
        "score_coverage": int(row["detector_scored_frames"]) >= required,
        "bbox_row_consistency": (
            int(row["positive_without_complete_bbox"]) == 0
            and int(row["nonpositive_with_any_bbox"]) == 0
        ),
        "no_active_out_of_range": int(row["detector_active_out_of_range"]) == 0,
        "no_detector_h5_lock": not lock_path.exists(),
    }
    if kind == "seg":
        checks["mask_row_consistency"] = (
            int(row["positive_without_mask"]) == 0
            and int(row["nonpositive_with_mask"]) == 0
        )
    result = {
        "available": True,
        "complete": all(checks.values()),
        "kind": kind,
        "video_id": video_id,
        "name": row["name"],
        "num_frames": frames,
        "required_scored_frames": required,
        "scored_frames": row["detector_scored_frames"],
        "positive_frames": row["detector_positive_frames"],
        "bbox_frames": row["detector_bbox_frames"],
        "mask_flags": row["detector_mask_flags"],
        "positive_without_complete_bbox": row["positive_without_complete_bbox"],
        "nonpositive_with_any_bbox": row["nonpositive_with_any_bbox"],
        "active_out_of_range": row["detector_active_out_of_range"],
        "checks": checks,
    }
    if kind == "seg":
        result["positive_without_mask"] = row["positive_without_mask"]
        result["nonpositive_with_mask"] = row["nonpositive_with_mask"]
    return result


def file_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        result.update({
            "bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        })
    lock = path.with_suffix(".h5.lock") if path.suffix == ".h5" else None
    if lock is not None:
        result["lock_path"] = str(lock)
        result["locked"] = lock.exists()
    return result


def command_inspect_project(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).expanduser().resolve()
    db_path = project_dir / "vidseq.db"
    with readonly_connection(db_path, args.busy_timeout) as connection:
        rows = aggregate_rows(connection, args.video_ids or None)
    if args.video_ids:
        found = {int(row["id"]) for row in rows}
        unknown = sorted(set(args.video_ids) - found)
        if unknown:
            raise VidSeqCtlError(f"unknown project video IDs: {unknown}")
    if args.summary:
        compact_videos = []
        for row in rows:
            video_dir = project_dir / "array_data" / str(row["id"])
            locked = sorted(path.name for path in video_dir.glob("*.h5.lock"))
            compact_videos.append({
                "id": row["id"],
                "name": row["name"],
                "path": row["path"],
                "num_frames": row["num_frames"],
                "training_frames": row["training_frames"],
                "detector_scored_frames": row["detector_scored_frames"],
                "detector_positive_frames": row["detector_positive_frames"],
                "detector_bbox_frames": row["detector_bbox_frames"],
                "detector_mask_flags": row["detector_mask_flags"],
                "positive_without_complete_bbox": row["positive_without_complete_bbox"],
                "nonpositive_with_any_bbox": row["nonpositive_with_any_bbox"],
                "detector_active_out_of_range": row["detector_active_out_of_range"],
                "detector_score_coverage": row["detector_score_coverage"],
                "source_exists": row["source_exists"],
                "h5_locks": locked,
            })
        dump({
            "project_dir": str(project_dir),
            "helper_apply_lock": file_metadata(project_dir / ".vidseqctl-apply.lock"),
            "summary": {
                "video_count": len(rows),
                "declared_frames": sum(int(row["num_frames"]) for row in rows),
                "training_frames": sum(int(row["training_frames"]) for row in rows),
                "detector_scored_frames": sum(int(row["detector_scored_frames"]) for row in rows),
                "detector_positive_frames": sum(int(row["detector_positive_frames"]) for row in rows),
                "videos_with_h5_locks": sum(bool(video["h5_locks"]) for video in compact_videos),
                "missing_sources": sum(not bool(row["source_exists"]) for row in rows),
            },
            "videos": compact_videos,
            "note": "Compact read-only SQLite/file summary; HDF5 content was not opened.",
        })
        return
    arrays: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        video_dir = project_dir / "array_data" / str(row["id"])
        arrays[str(row["id"])] = [
            file_metadata(video_dir / name)
            for name in ARRAY_FILES
            if (video_dir / name).exists() or (video_dir / name).with_suffix(".h5.lock").exists()
        ]
    model_paths = (
        project_dir / "models" / "detector.pt",
        project_dir / "models" / "detector_config.json",
        project_dir / "models" / "seg_detector.pt",
        project_dir / "models" / "yolo_train" / "weights" / "best.pt",
        project_dir / "models" / "rtdetr_train" / "weights" / "best.pt",
        project_dir / "models" / "seg_train" / "weights" / "best.pt",
    )
    dump({
        "project_dir": str(project_dir),
        "helper_apply_lock": file_metadata(project_dir / ".vidseqctl-apply.lock"),
        "database": file_metadata(db_path),
        "videos": rows,
        "arrays": arrays,
        "models": [file_metadata(path) for path in model_paths if path.exists()],
        "note": "HDF5 content was not opened; lock presence is reported from sidecars.",
    })


def command_wait_apply(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).expanduser().resolve()
    db_path = project_dir / "vidseq.db"
    started = time.monotonic()
    last_signature: Any = None
    while True:
        with readonly_connection(db_path, args.busy_timeout) as connection:
            rows = aggregate_rows(connection, args.video_ids or None)
        if not rows:
            raise VidSeqCtlError("no matching videos in project")
        if args.video_ids:
            found = {int(row["id"]) for row in rows}
            unknown = sorted(set(args.video_ids) - found)
            if unknown:
                raise VidSeqCtlError(f"unknown project video IDs: {unknown}")
        validate_missing_tolerance(
            (int(row["num_frames"]) for row in rows),
            args.allow_missing,
            args.allow_large_missing,
        )
        lock_state = []
        for row in rows:
            h5_path = project_dir / "array_data" / str(row["id"]) / "detector_masks.h5"
            lock_path = h5_path.with_suffix(".h5.lock")
            if lock_path.exists():
                lock_state.append({
                    "video_id": row["id"],
                    "name": row["name"],
                    "lock_path": str(lock_path),
                    "h5_bytes": h5_path.stat().st_size if h5_path.exists() else None,
                    "h5_modified_utc": (
                        datetime.fromtimestamp(h5_path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
                        if h5_path.exists() else None
                    ),
                })
        helper_lock_path = project_dir / ".vidseqctl-apply.lock"
        helper_lock = file_metadata(helper_lock_path)
        signature = (tuple(
            (
                row["id"],
                row["detector_scored_frames"],
                row["detector_positive_frames"],
                row["positive_without_complete_bbox"],
                row["nonpositive_with_any_bbox"],
                row["positive_without_mask"],
                row["nonpositive_with_mask"],
                row["detector_active_out_of_range"],
            )
            for row in rows
        ), tuple((item["video_id"], item["h5_bytes"], item["h5_modified_utc"]) for item in lock_state),
            helper_lock.get("exists"), helper_lock.get("modified_utc"))
        complete = all(
            int(row["detector_scored_frames"]) >= max(0, int(row["num_frames"]) - args.allow_missing)
            and int(row["positive_without_complete_bbox"]) == 0
            and int(row["nonpositive_with_any_bbox"]) == 0
            and int(row["detector_active_out_of_range"]) == 0
            and (
                args.kind == "bbox"
                or (
                    int(row["positive_without_mask"]) == 0
                    and int(row["nonpositive_with_mask"]) == 0
                )
            )
            for row in rows
        ) and not lock_state and not helper_lock.get("exists")
        if signature != last_signature or complete:
            dump({
                "at": utc_now(),
                "allow_missing": args.allow_missing,
                "kind": args.kind,
                "complete": complete,
                "active_h5_locks": lock_state,
                "helper_apply_lock": helper_lock,
                "videos": [
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "num_frames": row["num_frames"],
                        "scored": row["detector_scored_frames"],
                        "positive": row["detector_positive_frames"],
                        "bboxes": row["detector_bbox_frames"],
                        "active_out_of_range": row["detector_active_out_of_range"],
                        "positive_without_complete_bbox": row["positive_without_complete_bbox"],
                        "nonpositive_with_any_bbox": row["nonpositive_with_any_bbox"],
                        "positive_without_mask": row["positive_without_mask"] if args.kind == "seg" else None,
                        "nonpositive_with_mask": row["nonpositive_with_mask"] if args.kind == "seg" else None,
                        "coverage": row["detector_score_coverage"],
                    }
                    for row in rows
                ],
            })
            last_signature = signature
        if complete:
            return
        if args.timeout and time.monotonic() - started >= args.timeout:
            raise VidSeqCtlError(f"timed out after {args.timeout} seconds")
        time.sleep(args.interval)


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-videos", action="store_true")
    group.add_argument("--video-id", dest="video_ids", type=int, action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="VidSeq server root; default %(default)s")
    subparsers = parser.add_subparsers(dest="command", required=True)

    item = subparsers.add_parser("list-projects", help="List the VidSeq registry")
    item.set_defaults(handler=command_list_projects)

    item = subparsers.add_parser("create-project", help="Preview or create a project")
    item.add_argument("--name", required=True)
    item.add_argument("--parent", required=True)
    item.add_argument("--execute", action="store_true", help="Perform the mutation; otherwise preview")
    item.set_defaults(handler=command_create_project)

    item = subparsers.add_parser("list-videos", help="List a project's videos")
    item.add_argument("--project-id", type=int, required=True)
    item.set_defaults(handler=command_list_videos)

    item = subparsers.add_parser("verify-import", help="Check source metadata and base HDF5 containers without opening them")
    item.add_argument("--project-id", type=int, required=True)
    item.add_argument("--video-id", dest="video_ids", type=int, action="append", default=[])
    item.set_defaults(handler=command_verify_import)

    item = subparsers.add_parser("add-videos", help="Preview or import videos")
    item.add_argument("paths", nargs="*")
    item.add_argument("--project-id", type=int, required=True)
    item.add_argument("--glob", dest="glob_patterns", action="append", default=[])
    item.add_argument("--path-list", dest="path_lists", action="append", default=[])
    item.add_argument(
        "--allow-basename-collisions",
        action="store_true",
        help="Proceed after explicitly reviewing distinct paths with the same basename",
    )
    item.add_argument("--timeout", type=float, default=0, help="API seconds; 0 means unlimited")
    item.add_argument("--execute", action="store_true", help="Perform the mutation; otherwise preview")
    item.set_defaults(handler=command_add_videos)

    item = subparsers.add_parser("worker-status", help="Show segmentation GPU-worker status")
    item.set_defaults(handler=command_worker_status)

    item = subparsers.add_parser("load-worker", help="Preview or trigger segmentation GPU-worker loading")
    item.add_argument("--execute", action="store_true", help="Perform the mutation; otherwise preview")
    item.add_argument("--wait", action="store_true", help="Wait until status is ready or error")
    item.add_argument("--interval", type=float, default=2.0)
    item.add_argument("--timeout", type=float, default=600.0, help="seconds; 0 means unlimited")
    item.set_defaults(handler=command_load_worker)

    item = subparsers.add_parser("detector-status", help="Show model and training status")
    item.add_argument("--project-id", type=int, required=True)
    item.add_argument("--kind", choices=("bbox", "seg"), required=True)
    item.set_defaults(handler=command_detector_status)

    item = subparsers.add_parser("start-training", help="Preview or start detector training")
    item.add_argument("--project-id", type=int, required=True)
    item.add_argument("--kind", choices=("bbox", "seg"), required=True)
    add_selection_arguments(item)
    item.add_argument("--detector-type", choices=("yolo", "rtdetr"), help="bbox only")
    item.add_argument("--max-epochs", type=int, default=100)
    item.add_argument("--batch-size", type=int, default=4)
    item.add_argument("--lr", type=float, default=1e-4)
    item.add_argument("--early-stop-patience", type=int, default=20)
    item.add_argument("--from-checkpoint", action="store_true")
    item.add_argument(
        "--allow-unlabeled-videos",
        action="store_true",
        help="Include selected videos with zero reported training frames",
    )
    item.add_argument("--execute", action="store_true", help="Perform the mutation; otherwise preview")
    item.set_defaults(handler=command_start_training)

    item = subparsers.add_parser("wait-training", help="Poll detector training progress")
    item.add_argument("--project-id", type=int, required=True)
    item.add_argument("--kind", choices=("bbox", "seg"), required=True)
    item.add_argument("--interval", type=float, default=5.0)
    item.add_argument("--timeout", type=float, default=0, help="seconds; 0 means unlimited")
    item.add_argument(
        "--accept-existing-terminal",
        action="store_true",
        help="Accept completed even if no active state was observed in this monitor",
    )
    item.set_defaults(handler=command_wait_training)

    item = subparsers.add_parser("apply", help="Preview or run blocking full-video detector apply")
    item.add_argument("--project-id", type=int, required=True)
    item.add_argument("--kind", choices=("bbox", "seg"), required=True)
    add_selection_arguments(item)
    item.add_argument("--timeout", type=float, default=0, help="API seconds; 0 means unlimited")
    item.add_argument("--allow-missing", type=int, default=0, help="Postcheck tolerance for unreadable tail frames")
    item.add_argument(
        "--allow-large-missing",
        action="store_true",
        help="Permit tolerance above the one-frame-minimum, then min(10, 0.1%%) cap",
    )
    item.add_argument("--busy-timeout", type=float, default=30.0, help="SQLite read-only postcheck timeout")
    item.add_argument("--execute", action="store_true", help="Perform the mutation; otherwise preview")
    item.set_defaults(handler=command_apply)

    item = subparsers.add_parser("inspect-project", help="Read SQLite and file metadata without opening HDF5")
    item.add_argument("--project-dir", required=True)
    item.add_argument("--video-id", dest="video_ids", type=int, action="append", default=[])
    item.add_argument("--busy-timeout", type=float, default=30.0)
    item.add_argument("--summary", action="store_true", help="Emit compact cohort/video counters without per-file metadata")
    item.set_defaults(handler=command_inspect_project)

    item = subparsers.add_parser("wait-apply", help="Watch detector score coverage in read-only SQLite")
    item.add_argument("--project-dir", required=True)
    item.add_argument("--kind", choices=("bbox", "seg"), required=True)
    item.add_argument("--video-id", dest="video_ids", type=int, action="append", default=[])
    item.add_argument("--allow-missing", type=int, default=0)
    item.add_argument(
        "--allow-large-missing",
        action="store_true",
        help="Permit tolerance above the one-frame-minimum, then min(10, 0.1%%) cap",
    )
    item.add_argument("--interval", type=float, default=30.0)
    item.add_argument("--timeout", type=float, default=0, help="seconds; 0 means unlimited")
    item.add_argument("--busy-timeout", type=float, default=30.0)
    item.set_defaults(handler=command_wait_apply)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "detector_type", None) and getattr(args, "kind", None) != "bbox":
        parser.error("--detector-type is only valid with --kind bbox")
    if args.command == "load-worker" and args.wait and not args.execute:
        parser.error("--wait requires --execute for load-worker")
    if getattr(args, "allow_missing", 0) < 0:
        parser.error("--allow-missing must be nonnegative")
    try:
        args.handler(args)
    except (VidSeqCtlError, sqlite3.Error, OSError) as error:
        print(f"vidseqctl: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
