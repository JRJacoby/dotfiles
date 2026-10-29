"""Frame-accurate syllable scrubber: a video next to its per-frame label timeline.

All interaction runs client-side in assets/scrubber.js; this module lays out the
page and serves the video (with HTTP range support, needed for seeking) and the
per-frame labels.

Labels come from an .npz (one int array per key; pick one with --key) or a .npy
holding a single 1-D int array. Label index n must correspond to video frame n.

Usage:
    python app.py --video VIDEO.mp4 --labels LABELS.npz --key KEY \
        [--fps 30] [--title TITLE] [--subtitle SUBTITLE] [--port 8050]
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
from pathlib import Path

import numpy as np
from dash import Dash, html
from flask import send_file


def load_labels(path: Path, key: str | None) -> np.ndarray:
    if path.suffix == ".npz":
        archive = np.load(path)
        if key is None:
            if len(archive.files) != 1:
                raise SystemExit(f"{path} has {len(archive.files)} arrays; pass --key "
                                 f"(first few: {archive.files[:5]})")
            key = archive.files[0]
        labels = archive[key]
    else:
        labels = np.load(path)
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise SystemExit(f"labels must be 1-D, got shape {labels.shape}")
    return labels.astype(np.int64)


def video_frame_count(video_path: Path) -> int | None:
    if shutil.which("ffprobe") is None:
        return None
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def build_app(video_path: Path, labels: np.ndarray, fps: float,
              title: str, subtitle: str) -> Dash:
    payload = json.dumps({"fps": fps, "labels": labels.tolist()})

    app = Dash(__name__, title=title)
    server = app.server

    @server.route("/media/video.mp4")
    def video():
        return send_file(video_path, mimetype="video/mp4", conditional=True)

    @server.route("/media/data.json")
    def data():
        return server.response_class(payload, mimetype="application/json")

    app.layout = html.Div(className="page", children=[
        html.Div(className="header", children=[
            html.Span(title, className="title"),
            html.Span(subtitle, className="subtitle"),
        ]),
        html.Div(className="main", children=[
            html.Video(id="video", src="/media/video.mp4", preload="auto", muted=True),
            html.Div(className="side", children=[
                html.Div(id="readout"),
                html.Div(className="controls", children=[
                    html.Button("◀◀ bout", id="prev-bout"),
                    html.Button("◀", id="prev-frame"),
                    html.Button("play", id="play"),
                    html.Button("▶", id="next-frame"),
                    html.Button("bout ▶▶", id="next-bout"),
                ]),
                html.Div(className="controls", children=[
                    html.Label("speed"),
                    html.Select(id="speed", children=[
                        html.Option(s, value=s) for s in ["0.1", "0.25", "0.5", "1", "2", "4"]
                    ]),
                    html.Label("zoom ±"),
                    html.Select(id="zoom-half", children=[
                        html.Option(f"{s} fr", value=str(s)) for s in [30, 90, 150, 300, 900, 3000]
                    ]),
                ]),
                html.Div(className="controls", children=[
                    html.Label("frame"),
                    html.Div(id="goto-slot"),
                ]),
                html.Pre(className="help", children=(
                    "←/→        ±1 frame\n"
                    "shift+←/→  ±10 frames\n"
                    "[ / ]      prev / next syllable onset\n"
                    "space      play / pause\n"
                    "click or drag either timeline to seek"
                )),
            ]),
        ]),
        html.Div("zoom", className="track-label"),
        html.Canvas(id="zoom", className="track zoom"),
        html.Div("full segment", className="track-label"),
        html.Canvas(id="overview", className="track overview"),
    ])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True,
                        help="H.264/yuv420p/faststart mp4 whose frame n matches label n")
    parser.add_argument("--labels", type=Path, required=True, help=".npz or .npy of per-frame ints")
    parser.add_argument("--key", default=None, help="array name inside the .npz")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--title", default=None, help="page header (default: video file name)")
    parser.add_argument("--subtitle", default="", help="dim text next to the title, e.g. model name")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--skip-frame-check", action="store_true",
                        help="don't require label count == video frame count")
    args = parser.parse_args()

    video_path = args.video.resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    labels = load_labels(args.labels, args.key)

    if not args.skip_frame_check:
        n_video = video_frame_count(video_path)
        if n_video is None:
            print("ffprobe not found; skipping frame-count check", flush=True)
        elif n_video != len(labels):
            raise SystemExit(f"video has {n_video} frames but labels have {len(labels)}; "
                             "frame n would not match label n")

    title = args.title or (args.key or video_path.stem)
    app = build_app(video_path, labels, args.fps, title, args.subtitle)
    print(f"http://localhost:{args.port}/")
    print(f"http://{socket.getfqdn()}:{args.port}/", flush=True)
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
