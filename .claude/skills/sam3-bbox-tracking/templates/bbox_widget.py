"""Dash bounding-box widget for SAM3 PVS prompts.

Each invocation accepts one or more video paths via ``--sessions``. The widget
discovers a session id from each video's stem (or uses the explicit one passed
via ``--session-ids``), opens a Plotly canvas per session, lets the user draw
rectangles, label each with a polarity (``positive``/``negative``) and a free-text
role (e.g. ``"familiar"``/``"novel"``/``"pig"``), and saves a single ``boxes.json``
per session via atomic write.

Multi-session UX: pick a session from a dropdown, switch between them without
losing in-progress edits (state lives in a ``dcc.Store``). Save writes only the
currently-selected session's slice.

Output schema (``boxes.json``) — one file per session:

    {
      "session": "<session_id>",
      "video": "<path/relative/to/cwd_or_absolute>",
      "frame_idx": 0,
      "boxes": [
        {"label": "positive", "role": "<role_a>", "xyxy": [x0,y0,x1,y1]},
        {"label": "negative", "role": "<role_a>", "xyxy": [x0,y0,x1,y1]},
        {"label": "positive", "role": "<role_b>", "xyxy": [x0,y0,x1,y1]}
      ]
    }

``xyxy`` is in native video pixels, integer. The driver (run_sam3_pvs.py)
groups boxes by ``role`` to assign one ``obj_id`` per unique role.

Usage:
    uv run python bbox_widget.py --sessions VIDEO1 VIDEO2 ... [--boxes-dir DIR]

Then open http://127.0.0.1:8050.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update


LABEL_COLOR = {"positive": "green", "negative": "red"}


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return n


def video_dims(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return w, h


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """Return one frame as RGB uint8 ``(H, W, 3)``."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {frame_idx} from {video_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def normalize_box_corners(x0, y0, x1, y1):
    """Return (xmin, ymin, xmax, ymax) regardless of which corner was drag-started."""
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def atomic_write_json(path: Path, data) -> None:
    """Write JSON via a temp file in the same dir + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="_tmp_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# CLI + startup
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--sessions",
        nargs="+",
        required=True,
        type=Path,
        help="One or more video files. Each becomes one session in the widget.",
    )
    ap.add_argument(
        "--session-ids",
        nargs="+",
        default=None,
        help="Optional explicit session ids, one per --sessions (default: video stem).",
    )
    ap.add_argument(
        "--boxes-dir",
        type=Path,
        default=None,
        help="Where to write each session's boxes.json. Default: alongside the video, "
             "named '<session_id>.boxes.json'.",
    )
    ap.add_argument("--port", type=int, default=8050)
    return ap.parse_args()


def _build_sessions(args: argparse.Namespace) -> dict[str, dict]:
    if args.session_ids and len(args.session_ids) != len(args.sessions):
        sys.exit("ERROR: --session-ids must have the same length as --sessions")

    out: dict[str, dict] = {}
    for i, video_path in enumerate(args.sessions):
        video_path = video_path.resolve()
        if not video_path.exists():
            sys.exit(f"ERROR: video not found: {video_path}")

        session_id = args.session_ids[i] if args.session_ids else video_path.stem
        if session_id in out:
            sys.exit(f"ERROR: duplicate session_id {session_id!r}; pass --session-ids to disambiguate.")

        n_frames = video_frame_count(video_path)
        w, h = video_dims(video_path)

        if args.boxes_dir is not None:
            boxes_json = args.boxes_dir / f"{session_id}.boxes.json"
        else:
            boxes_json = video_path.with_name(f"{session_id}.boxes.json")

        if boxes_json.exists():
            try:
                state = json.loads(boxes_json.read_text())
            except Exception as e:
                sys.exit(f"ERROR: failed to parse {boxes_json}: {e}")
            state.setdefault("session", session_id)
            state.setdefault("video", str(video_path))
            state.setdefault("frame_idx", 0)
            state.setdefault("boxes", [])
        else:
            state = {
                "session": session_id,
                "video": str(video_path),
                "frame_idx": 0,
                "boxes": [],
            }

        out[session_id] = {
            "video": str(video_path),
            "frame_count": n_frames,
            "width": w,
            "height": h,
            "boxes_json": str(boxes_json),
            "state": state,
        }
        print(
            f"  session {session_id}: video={video_path.name} "
            f"({w}x{h}, {n_frames} frames), "
            f"existing boxes.json={'yes' if boxes_json.exists() else 'no'}",
            flush=True,
        )
    return out


ARGS = _parse_args()
SESSIONS = _build_sessions(ARGS)
SESSION_IDS = list(SESSIONS.keys())
DEFAULT_SESSION_ID = SESSION_IDS[0]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def build_figure(session_id: str, frame_idx: int, boxes: list[dict], label_mode: str) -> go.Figure:
    """Render the current frame with existing boxes drawn as shapes.

    Coordinate contract: ``go.Image(z=frame_rgb)`` + ``xaxis.range=[0, W]`` +
    ``yaxis.range=[H, 0]`` means Plotly reports drawn-shape coords in data-space,
    which equals native video pixels regardless of browser scaling.
    """
    sess = SESSIONS[session_id]
    frame_rgb = read_frame(Path(sess["video"]), frame_idx)
    w, h = sess["width"], sess["height"]

    shapes = []
    for box in boxes:
        color = LABEL_COLOR.get(box.get("label", "positive"), "green")
        shapes.append(
            dict(
                type="rect",
                x0=box["xyxy"][0],
                y0=box["xyxy"][1],
                x1=box["xyxy"][2],
                y1=box["xyxy"][3],
                line=dict(color=color, width=2),
                fillcolor="rgba(0,0,0,0)",
                editable=False,
            )
        )

    fig = go.Figure()
    fig.add_trace(go.Image(z=frame_rgb))
    fig.update_layout(
        xaxis=dict(range=[0, w], showgrid=False, zeroline=False, visible=False, constrain="domain"),
        yaxis=dict(range=[h, 0], showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="drawrect",
        newshape=dict(line=dict(color=LABEL_COLOR.get(label_mode, "green"), width=2)),
        shapes=shapes,
    )
    return fig


def _initial_store() -> dict:
    return {sid: SESSIONS[sid]["state"] for sid in SESSION_IDS}


def _dropdown_options(store: dict) -> list[dict]:
    """Annotate each session's dropdown label with its box count so the user can
    see at a glance which sessions are already labeled vs still TODO."""
    opts = []
    for sid in SESSION_IDS:
        n_boxes = len(store.get(sid, {}).get("boxes", []))
        suffix = f"  ({n_boxes} box{'es' if n_boxes != 1 else ''})" if n_boxes else "  -"
        opts.append({"label": f"{sid}{suffix}", "value": sid})
    return opts


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app = Dash(__name__)

_default_state = SESSIONS[DEFAULT_SESSION_ID]["state"]
_default_frame_count = SESSIONS[DEFAULT_SESSION_ID]["frame_count"]

app.layout = html.Div(
    [
        dcc.Store(id="boxes-store", data=_initial_store()),
        html.H2("SAM3 PVS — bbox + role widget"),
        html.Div(
            [
                html.Label("Session:"),
                dcc.Dropdown(
                    id="session-selector",
                    options=_dropdown_options(_initial_store()),
                    value=DEFAULT_SESSION_ID,
                    clearable=False,
                    style={"marginLeft": "8px", "minWidth": "420px", "display": "inline-block"},
                ),
            ],
            style={"marginBottom": "8px"},
        ),
        html.Div(
            [
                html.Label("Frame:"),
                dcc.Slider(
                    id="frame-slider",
                    min=0,
                    max=max(_default_frame_count - 1, 0),
                    step=1,
                    value=_default_state.get("frame_idx", 0),
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
            style={"marginBottom": "8px"},
        ),
        html.Div(
            [
                html.Label("Polarity:"),
                dcc.RadioItems(
                    id="label-mode",
                    options=[
                        {"label": "positive", "value": "positive"},
                        {"label": "negative", "value": "negative"},
                    ],
                    value="positive",
                    inline=True,
                    style={"marginLeft": "8px", "display": "inline-block"},
                ),
                html.Label("Role:", style={"marginLeft": "24px"}),
                dcc.Input(
                    id="role-input",
                    type="text",
                    placeholder="e.g. familiar, novel, object_a",
                    value="",
                    style={"marginLeft": "8px", "width": "240px"},
                ),
            ],
            style={"marginBottom": "8px"},
        ),
        dcc.Graph(
            id="canvas",
            config={"modeBarButtonsToAdd": ["drawrect", "eraseshape"], "displaylogo": False},
            style={"width": "100%", "height": "70vh"},
        ),
        html.Div(id="box-list", style={"marginTop": "8px"}),
        html.Div(
            [
                html.Button("Save", id="save-btn", n_clicks=0),
                html.Div(id="save-status", style={"marginTop": "4px", "color": "green"}),
            ],
            style={"marginTop": "12px"},
        ),
    ],
    style={"maxWidth": "1400px", "margin": "0 auto", "padding": "16px", "fontFamily": "sans-serif"},
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _delete_button_id(idx: int) -> dict:
    return {"type": "delete-box-btn", "index": idx}


def _render_box_list(session_id: str, store: dict) -> list:
    boxes = store.get(session_id, {}).get("boxes", [])
    if not boxes:
        return [html.P("No boxes for this session.", style={"color": "#888"})]
    rows = [
        html.Tr([
            html.Th("#"), html.Th("Polarity"), html.Th("Role"),
            html.Th("x0"), html.Th("y0"), html.Th("x1"), html.Th("y1"),
            html.Th(""),
        ])
    ]
    for i, box in enumerate(boxes):
        color = LABEL_COLOR.get(box.get("label", "positive"), "green")
        rows.append(
            html.Tr([
                html.Td(str(i)),
                html.Td(box.get("label", ""), style={"color": color, "fontWeight": "bold"}),
                html.Td(box.get("role", ""), style={"fontFamily": "monospace"}),
                html.Td(box["xyxy"][0]),
                html.Td(box["xyxy"][1]),
                html.Td(box["xyxy"][2]),
                html.Td(box["xyxy"][3]),
                html.Td(
                    html.Button(
                        "Delete",
                        id=_delete_button_id(i),
                        n_clicks=0,
                        style={"color": "red", "cursor": "pointer"},
                    )
                ),
            ])
        )
    return [html.Table(rows, style={"borderCollapse": "collapse", "width": "100%"})]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("frame-slider", "max"),
    Output("frame-slider", "value"),
    Output("canvas", "figure"),
    Output("box-list", "children"),
    Input("session-selector", "value"),
    State("boxes-store", "data"),
    State("label-mode", "value"),
)
def on_session_change(session_id, store, label_mode):
    if session_id is None:
        return no_update, no_update, no_update, no_update
    sess_state = store.get(session_id) or SESSIONS[session_id]["state"]
    frame_idx = sess_state.get("frame_idx", 0)
    slider_max = max(SESSIONS[session_id]["frame_count"] - 1, 0)
    if frame_idx > slider_max:
        frame_idx = slider_max
    fig = build_figure(session_id, frame_idx, sess_state.get("boxes", []), label_mode)
    return slider_max, frame_idx, fig, _render_box_list(session_id, store)


@app.callback(
    Output("canvas", "figure", allow_duplicate=True),
    Output("boxes-store", "data"),
    Input("frame-slider", "value"),
    State("session-selector", "value"),
    State("boxes-store", "data"),
    State("label-mode", "value"),
    prevent_initial_call=True,
)
def on_frame_change(frame_idx, session_id, store, label_mode):
    if session_id is None:
        return no_update, no_update
    store[session_id]["frame_idx"] = frame_idx
    fig = build_figure(session_id, frame_idx, store[session_id].get("boxes", []), label_mode)
    return fig, store


@app.callback(
    Output("boxes-store", "data", allow_duplicate=True),
    Output("box-list", "children", allow_duplicate=True),
    Input("canvas", "relayoutData"),
    State("session-selector", "value"),
    State("boxes-store", "data"),
    State("label-mode", "value"),
    State("role-input", "value"),
    prevent_initial_call=True,
)
def on_shape_drawn(relayout_data, session_id, store, label_mode, role):
    if not relayout_data or session_id is None:
        return no_update, no_update

    corners = None
    shapes = relayout_data.get("shapes")
    if isinstance(shapes, list):
        n_stored = len(store[session_id]["boxes"])
        if len(shapes) > n_stored:
            s = shapes[-1]
            corners = (s["x0"], s["y0"], s["x1"], s["y1"])
    elif "shapes[-1].x0" in relayout_data:
        corners = (
            relayout_data["shapes[-1].x0"],
            relayout_data["shapes[-1].y0"],
            relayout_data["shapes[-1].x1"],
            relayout_data["shapes[-1].y1"],
        )

    if corners is None:
        return no_update, no_update

    xmin, ymin, xmax, ymax = normalize_box_corners(*corners)
    box = {
        "label": label_mode,
        "role": (role or "").strip(),
        "xyxy": [round(xmin), round(ymin), round(xmax), round(ymax)],
    }
    store[session_id]["boxes"].append(box)
    return store, _render_box_list(session_id, store)


@app.callback(
    Output("boxes-store", "data", allow_duplicate=True),
    Output("box-list", "children", allow_duplicate=True),
    Output("canvas", "figure", allow_duplicate=True),
    Input({"type": "delete-box-btn", "index": ALL}, "n_clicks"),
    State("session-selector", "value"),
    State("boxes-store", "data"),
    State("label-mode", "value"),
    prevent_initial_call=True,
)
def on_delete_box(n_clicks_list, session_id, store, label_mode):
    ctx = callback_context
    if (
        not ctx.triggered
        or not any(n for n in n_clicks_list if n)
        or session_id is None
    ):
        return no_update, no_update, no_update

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    btn_id = json.loads(triggered_id)
    idx = btn_id["index"]

    boxes = store[session_id]["boxes"]
    if 0 <= idx < len(boxes):
        boxes.pop(idx)
    store[session_id]["boxes"] = boxes

    frame_idx = store[session_id].get("frame_idx", 0)
    fig = build_figure(session_id, frame_idx, boxes, label_mode)
    return store, _render_box_list(session_id, store), fig


@app.callback(
    Output("save-status", "children"),
    Input("save-btn", "n_clicks"),
    State("session-selector", "value"),
    State("boxes-store", "data"),
    prevent_initial_call=True,
)
def on_save(n_clicks, session_id, store):
    if session_id is None:
        return "no session selected"
    sess_state = store.get(session_id)
    if sess_state is None:
        return f"no store slice for session {session_id}"
    boxes_json = Path(SESSIONS[session_id]["boxes_json"])
    atomic_write_json(boxes_json, sess_state)
    boxes = sess_state.get("boxes", [])
    return (
        f"Saved {session_id} to {boxes_json} — {len(boxes)} box(es) "
        f"(roles: {sorted({b.get('role','') for b in boxes if b.get('role')})})"
    )


@app.callback(
    Output("session-selector", "options"),
    Input("boxes-store", "data"),
    prevent_initial_call=True,
)
def on_store_change(store):
    """Refresh the (N boxes) suffix in the session dropdown whenever the store
    changes. Lets the user see live which sessions are now labeled vs TODO
    without needing to restart the widget. Idempotent — if box counts didn't
    change Plotly Dash diff-renders away the no-op."""
    return _dropdown_options(store)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"sessions: {SESSION_IDS}", flush=True)
    app.run(debug=False, host="127.0.0.1", port=ARGS.port)
