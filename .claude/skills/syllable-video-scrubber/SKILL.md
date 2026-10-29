---
name: syllable-video-scrubber
description: Use when comparing a per-frame discrete label timeline (MoSeq or keypoint-MoSeq syllables, HMM states, behavior annotations, any integer label per frame) against its video, to check whether label transitions line up with visible behavior, or when asked for a frame-by-frame viewer, scrubber, or QC tool for syllable boundaries.
---

# Syllable video scrubber

## Overview

A small Dash app: the video on the left, a readout and controls on the right, and two canvas timelines underneath (a zoom track centered on the current frame and a full-length overview). Every seek is frame-exact, and the app shows the frame the browser actually presented, not the one it asked for. The design in `templates/` is known to work well for this job; reproduce it rather than redesigning.

Core rule: **video frame n must be label index n, and the screen must prove it.** Everything below serves that.

## Files

| Template | Role |
|---|---|
| `templates/app.py` | Layout, Flask routes for the video (Range-capable) and labels JSON, CLI args, frame-count check |
| `templates/assets/scrubber.js` | All interaction: seeking, frame readback, drawing, keys, mouse |
| `templates/assets/scrubber.css` | Dark layout |

Copy all three into the project, keeping `assets/` next to `app.py` (Dash auto-serves it). Change `app.py` args or layout text as needed; leave the JS/CSS behavior alone.

```bash
python app.py --video clip.mp4 --labels labels.npz --key SESSION_KEY \
    --fps 30 --title "SESSION_KEY" --subtitle "model name" --port 8051
```

## Workflow

1. **Environment.** Confirm with the user which env to use. The app needs only `dash` (which brings Flask) and `numpy`; `ffprobe`/`ffmpeg` on PATH.
2. **Export labels once** to an npz so the app never imports JAX or the model code (run this in the model's env):
   ```python
   labels = {str(k): np.asarray(v).astype(np.int16) for k, v in states.items()}
   np.savez_compressed(out_path, **labels)   # one int array per session/segment key
   ```
   For moseq/kpms results, `states` is the per-session label dict from the model or results file. Watch for padding or a lag offset the model applies (e.g. AR lags trimming the first frames); the exported array must be aligned to the video's first frame.
3. **Make or check the video** (see Alignment and Browser codec below).
4. **Verify alignment** before launching.
5. **Launch** on a free port (check with `ss -ltn`), bound to `0.0.0.0`. The app prints localhost and FQDN URLs; give both.
6. **Pre-handoff checks** (below).

## Frame-accurate seeking (implemented in scrubber.js)

- Frame n spans `[n/fps, (n+1)/fps)`. Always seek to the midpoint `(n + 0.5)/fps`. Seeking to `n/fps` lands on a boundary and the decoder shows n-1 or n unpredictably.
- Read back the presented frame with `requestVideoFrameCallback`: `round(meta.mediaTime * fps)`. The timelines and readout follow that value. While it differs from the target, the readout shows `seeking… (on screen: N)` in amber.
- No native `<video controls>`; their scrub bar seeks to boundaries.

## Alignment

- Label array length must equal the video's frame count: `ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 clip.mp4`. `app.py` refuses to start on a mismatch.
- `start_time` must be 0 and frames evenly spaced at 1/fps: `ffprobe -v error -select_streams v:0 -show_entries stream=start_time,r_frame_rate,avg_frame_rate clip.mp4`, and spot-check `-show_entries packet=pts_time` on the first few packets.
- If the video is a cut from a longer recording, cut it frame-exactly and re-encode:
  ```bash
  ffmpeg -y -i full.avi -vf "trim=start_frame=S:end_frame=E,setpts=PTS-STARTPTS" \
      -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -movflags +faststart -an clip.mp4
  ```
  `end_frame` is exclusive, so the clip has `E - S` frames. Never use `-ss ... -c copy`: stream copy snaps to keyframes and silently shifts the clip by several frames.
- If labels came from a different video stream than the one shown (e.g. depth vs. IR), confirm the two are 1:1 in frame count first.

## Browser codec

Chrome plays H.264 (`libx264`), `yuv420p`, even width/height, with `-movflags +faststart`. It refuses MPEG-4 Part 2 (`mp4v`, OpenCV's default) and 10-bit `yuv420p10le`. Confirm: `ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt,width,height clip.mp4` should report `h264`, `yuv420p`.

## Serving

- Video via `send_file(path, mimetype="video/mp4", conditional=True)`. `conditional=True` enables HTTP Range (206 responses); without it the browser cannot seek.
- Labels via a JSON route (`/media/data.json` with `fps` and `labels`).
- All interaction is client-side JS in `assets/`, which polls until Dash's async layout has rendered the elements before wiring up. Timelines are plain `<canvas>`, not `dcc.Graph`/Plotly, so redraw on every frame stays fast. No Dash callbacks.

## Features (keep all)

| Area | Behavior |
|---|---|
| Zoom track | Centered on cursor, ±30/90/150/300/900/3000 frames. Colored bouts, syllable id text when wide enough, black onset markers, per-frame ticks when zoomed in (taller every 10th), white box on current frame. Click jumps to frame, drag pans. |
| Overview | Whole video; click or drag to scrub; translucent box shows the zoom window. |
| Readout | Frame / last frame, m:ss.sss, syllable chip, "frame k of L in bout", bout index / total, prev → next syllable chips, seeking warning. |
| Keys | ←/→ ±1, shift ±10, `[` / `]` prev/next onset, space play/pause (ignored while typing in inputs). |
| Controls | Bout and frame step buttons, play, speed 0.1 to 4×, zoom select, frame-number jump box (Enter). |
| Colors | Hue = `id * 137.508 mod 360` (golden angle) with lightness and saturation cycling over 3 levels; text black or white by lightness. Onset lines keep boundaries visible when adjacent colors are similar. |

## Pre-handoff checks

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:PORT/                 # 200
curl -s http://localhost:PORT/media/data.json | head -c 200                      # fps + labels
curl -s -o /dev/null -D - -H "Range: bytes=0-1023" http://localhost:PORT/media/video.mp4 | head -1   # 206 PARTIAL CONTENT
node --check assets/scrubber.js
```

If a browser automation tool is available, open the page, step a few frames, jump with `]`, and confirm the readout never stays on "seeking…". Otherwise tell the user it was checked with curl only, not in a browser.

## Common mistakes

| Mistake | Effect |
|---|---|
| Seeking to `n/fps` | Off-by-one frames that come and go |
| Cutting with `-ss` + `-c copy` | Clip offset by several frames from the labels |
| `mp4v` or 10-bit output | Black video or no playback in Chrome |
| `send_file` without `conditional=True` | Video plays but seeking jumps to 0 or fails |
| Importing the model library in the app | Slow start, env conflicts; export to npz instead |
| Plotly/`dcc.Graph` timeline driven by callbacks | Laggy scrubbing, round-trips per frame |
| Trusting `currentTime` instead of the frame callback | Readout claims a frame that is not on screen |
