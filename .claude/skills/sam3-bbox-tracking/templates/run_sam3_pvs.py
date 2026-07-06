"""SAM3 PVS streaming driver: bbox-prompted multi-object tracking from a role-labeled
``boxes.json``.

Reads a single session's ``boxes.json`` (produced by ``bbox_widget.py``), groups
the boxes by their ``role`` field to assign one ``obj_id`` per unique role, runs
streaming SAM3 forward through the video, and writes per-frame outputs.

This is a template — adapt the OUTPUT block at the bottom for your project's
storage convention. The default output is:

- ``<output_dir>/<session_id>.masks.h5`` — packed-bit per-frame masks, group per
  ``obj_id``, attrs include the role string.
- ``<output_dir>/<session_id>.objects.csv`` — per-frame `(obj_id, role, area, score)`.
- ``<output_dir>/<session_id>.overlay.mp4`` — visual QA video.

Usage:
    uv run python run_sam3_pvs.py --boxes-json PATH/boxes.json --output-dir DIR
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_model
from streaming_pvs import StreamingSam3Tracker


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_boxes(boxes_json: Path) -> tuple[str, Path, list[dict]]:
    """Return (session_id, video_path, boxes) from a boxes.json file."""
    data = json.loads(boxes_json.read_text())
    return data["session"], Path(data["video"]), data.get("boxes", [])


def open_ffmpeg_writer(out_path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    """Open an ffmpeg subprocess that writes raw RGB frames to an H.264 MP4.
    Uses libx264 + yuv420p + faststart for browser-friendly playback."""
    return subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-pix_fmt", "rgb24", "-i", "-",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )


def color_for_obj_id(obj_id: int, cache: dict) -> tuple[int, int, int]:
    """Stable BGR color per obj_id, drawn from a fixed palette."""
    palette = [
        (255, 64, 64), (64, 255, 64), (64, 128, 255), (255, 192, 64),
        (255, 64, 255), (64, 255, 255), (192, 192, 192),
    ]
    if obj_id not in cache:
        cache[obj_id] = palette[(obj_id - 1) % len(palette)]
    return cache[obj_id]


def draw_object_overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    """In-place: tint mask area + draw label dot at centroid."""
    if mask.sum() == 0:
        return
    overlay = rgb.copy()
    overlay[mask] = np.array(color, dtype=np.uint8)
    rgb[:] = ((0.6 * rgb + 0.4 * overlay)).astype(np.uint8)
    ys, xs = np.where(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    cv2.circle(rgb, (cx, cy), 6, color, -1)


class PackedMaskWriter:
    """Per-frame packed-bit mask storage, one group per obj_id."""

    def __init__(self, out_path: Path, h: int, w: int, role_by_obj_id: dict[int, str]):
        self.h, self.w = h, w
        self.f = h5py.File(out_path, "w")
        for obj_id, role in role_by_obj_id.items():
            g = self.f.create_group(f"obj_{obj_id}")
            g.attrs["role"] = role
            g.attrs["height"] = h
            g.attrs["width"] = w

    def append(self, obj_id: int, frame_idx: int, mask_bool: np.ndarray) -> None:
        packed = np.packbits(mask_bool.astype(np.uint8))
        ds = self.f[f"obj_{obj_id}"].create_dataset(
            f"frame_{frame_idx:06d}", data=packed, compression="gzip", compression_opts=4,
        )
        ds.attrs["n_pixels"] = int(mask_bool.sum())

    def close(self) -> None:
        self.f.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--boxes-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--keep", type=int, default=32, help="sliding-window memory size")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames for testing")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    session_id, video_path, boxes = load_boxes(args.boxes_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    positives = [b for b in boxes if b.get("label") == "positive"]
    negatives = [b for b in boxes if b.get("label") == "negative"]
    if not positives:
        raise ValueError(f"{args.boxes_json}: no positive boxes")

    # role -> obj_id (1-indexed, in first-seen order)
    role_to_obj_id: dict[str, int] = {}
    for b in positives:
        role = b.get("role", "")
        if role not in role_to_obj_id:
            role_to_obj_id[role] = len(role_to_obj_id) + 1
    obj_id_to_role = {v: k for k, v in role_to_obj_id.items()}
    print(f"session={session_id} video={video_path.name}")
    print(f"roles -> obj_ids: {role_to_obj_id}")

    cap = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = n_total if args.max_frames is None else min(n_total, args.max_frames)
    print(f"video: {w}x{h} @ {fps:.1f}fps, {n_total} frames, processing {n_frames}")

    # ---- model setup ----
    sam3_model = build_sam3_video_model(apply_temporal_disambiguation=False)
    predictor = sam3_model.tracker
    predictor.__class__ = StreamingSam3Tracker
    predictor.backbone = sam3_model.detector.backbone

    state = predictor.init_stream_state(h, w)
    mask_writer = PackedMaskWriter(args.output_dir / f"{session_id}.masks.h5", h, w, obj_id_to_role)
    overlay_writer = open_ffmpeg_writer(args.output_dir / f"{session_id}.overlay.mp4", w, h, fps)
    csv_f = open(args.output_dir / f"{session_id}.objects.csv", "w")
    csv_f.write("frame,obj_id,role,cx,cy,area,score\n")
    id_color: dict = {}

    def emit_frame(frame_idx: int, rgb: np.ndarray, masks: np.ndarray, scores: np.ndarray) -> None:
        # masks: (n_obj, H, W) bool; scores: (n_obj,) float in [0,1]
        for obj_id in obj_id_to_role:
            mask = masks[obj_id - 1]
            color = color_for_obj_id(obj_id, id_color)
            area = int(mask.sum())
            if area > 0:
                mask_writer.append(obj_id, frame_idx, mask)
                ys, xs = np.where(mask)
                cy, cx = float(ys.mean()), float(xs.mean())
                draw_object_overlay(rgb, mask, color)
                csv_f.write(
                    f"{frame_idx},{obj_id},{obj_id_to_role[obj_id]},"
                    f"{cx:.2f},{cy:.2f},{area},{scores[obj_id-1]:.4f}\n"
                )
        overlay_writer.stdin.write(rgb.tobytes())

    try:
        # ---- frame 0: prompt ----
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError("could not read frame 0")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img0 = predictor.preprocess_frame(rgb)
        predictor.seed_features(state, 0, img0)

        # One positive box per role on frame 0 (multiple positives sharing a role
        # are merged into one prompt by taking the union bbox).
        per_role_box: dict[str, tuple[float, float, float, float]] = {}
        for b in positives:
            role = b.get("role", "")
            x0, y0, x1, y1 = b["xyxy"]
            if role in per_role_box:
                ox0, oy0, ox1, oy1 = per_role_box[role]
                per_role_box[role] = (min(ox0, x0), min(oy0, y0), max(ox1, x1), max(oy1, y1))
            else:
                per_role_box[role] = (x0, y0, x1, y1)

        res0 = None
        for role, box in per_role_box.items():
            obj_id = role_to_obj_id[role]
            rel = np.array(
                [[box[0] / w, box[1] / h, box[2] / w, box[3] / h]], dtype=np.float32
            )
            res0 = predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=obj_id, box=rel
            )
        # Negative boxes attached per-role (refinement). SAM3 accepts box + extra
        # points/labels via the points API; negatives are represented as label=0
        # points at the box corners. For an initial template we just apply
        # negative boxes as separate "negative" point pairs on the same obj_id.
        for b in negatives:
            role = b.get("role", "")
            if role not in role_to_obj_id:
                continue  # negative for a role with no positive — skip
            obj_id = role_to_obj_id[role]
            x0, y0, x1, y1 = b["xyxy"]
            # SAM3 represents a box as two corner points with labels 2, 3
            # (positive corners). For a negative box, use the same corners with
            # label 0 (negative point), interpreted as "not here". This is a
            # heuristic — fine-tune if your data needs it.
            points = np.array([[(x0 + x1) / 2 / w, (y0 + y1) / 2 / h]], dtype=np.float32)
            labels = np.array([[0]], dtype=np.int32)
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=obj_id, points=points, labels=labels,
            )

        predictor.propagate_in_video_preflight(state)
        masks0 = (res0[3][:, 0] > 0).cpu().numpy()  # (n_obj, H, W) bool
        emit_frame(0, rgb, masks0, np.ones(len(role_to_obj_id), dtype=np.float32))

        # ---- per-frame loop ----
        torch.cuda.synchronize()
        window_start = time.perf_counter()
        for f in range(1, n_frames):
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = predictor.preprocess_frame(rgb)
            _, _, _, video_res, score_logits = predictor.stream_step(state, f, img)
            masks = (video_res[:, 0] > 0).cpu().numpy()
            scores = torch.sigmoid(score_logits.reshape(-1)).cpu().numpy()
            emit_frame(f, rgb, masks, scores)
            predictor.evict(state, f, args.keep)

            if f % args.log_every == 0:
                torch.cuda.synchronize()
                now = time.perf_counter()
                free, total = torch.cuda.mem_get_info()
                print(
                    f"  [{f:6d}/{n_frames}] {args.log_every / (now - window_start):.2f} fps  "
                    f"gpu={(total - free) / 1e9:.2f}GB",
                    flush=True,
                )
                window_start = now
    finally:
        cap.release()
        overlay_writer.stdin.close()
        overlay_writer.wait()
        mask_writer.close()
        csv_f.close()
        print(f"done: session={session_id} -> {args.output_dir}/")


if __name__ == "__main__":
    main()
