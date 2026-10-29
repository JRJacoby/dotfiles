// Frame-accurate syllable scrubber.
//
// Frame n of the video occupies [n/fps, (n+1)/fps). Every seek targets the
// midpoint (n + 0.5)/fps so it can never land on a frame boundary. The frame
// actually on screen is read back from requestVideoFrameCallback (mediaTime is
// the presented frame's timestamp, n/fps), and that is what the timelines show.

(function () {
  "use strict";

  const S = {
    labels: null, fps: 30, n: 0,
    boutStart: [], boutEnd: [], boutLabel: [],
    shown: 0,          // frame currently presented by the video
    target: null,      // frame requested by the last seek, until it is presented
    zoomHalf: 30,
    overviewBg: null,
  };
  let els = {};

  // ---------- colors ----------
  const colorCache = {};
  function sylColor(id) {
    if (!(id in colorCache)) {
      const hue = (id * 137.508) % 360;
      const light = [40, 55, 68][id % 3];
      const sat = [70, 60, 75][(id >> 1) % 3];
      colorCache[id] = { fill: `hsl(${hue.toFixed(1)},${sat}%,${light}%)`, text: light > 60 ? "#111" : "#fff" };
    }
    return colorCache[id];
  }

  // ---------- data ----------
  function buildBouts(labels) {
    for (let i = 0; i < labels.length; i++) {
      if (i === 0 || labels[i] !== labels[i - 1]) {
        if (i > 0) S.boutEnd.push(i);           // exclusive
        S.boutStart.push(i);
        S.boutLabel.push(labels[i]);
      }
    }
    S.boutEnd.push(labels.length);
  }

  function boutAt(frame) {                         // index of bout containing frame
    let lo = 0, hi = S.boutStart.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (S.boutStart[mid] <= frame) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  // ---------- seeking ----------
  const clamp = (f) => Math.max(0, Math.min(S.n - 1, f));

  function seek(frame) {
    frame = clamp(Math.round(frame));
    S.target = frame;
    els.video.currentTime = (frame + 0.5) / S.fps;
    render();
  }

  function currentFrame() { return S.target !== null ? S.target : S.shown; }

  function onFrame(_now, meta) {
    S.shown = clamp(Math.round(meta.mediaTime * S.fps));
    if (S.target !== null && (S.shown === S.target || !els.video.paused)) S.target = null;
    render();
    els.video.requestVideoFrameCallback(onFrame);
  }

  // ---------- drawing ----------
  function sizeCanvas(c) {
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
      return true;
    }
    return false;
  }

  function drawOverviewBg() {
    const c = els.overview, dpr = window.devicePixelRatio || 1;
    const off = document.createElement("canvas");
    off.width = c.width; off.height = c.height;
    const g = off.getContext("2d");
    const W = off.width, H = off.height;
    for (let b = 0; b < S.boutStart.length; b++) {
      const x0 = S.boutStart[b] / S.n * W, x1 = S.boutEnd[b] / S.n * W;
      g.fillStyle = sylColor(S.boutLabel[b]).fill;
      g.fillRect(x0, 0, Math.max(x1 - x0, 1 * dpr / 2), H);
    }
    S.overviewBg = off;
  }

  function drawOverview(frame) {
    const c = els.overview, g = c.getContext("2d"), dpr = window.devicePixelRatio || 1;
    const W = c.width, H = c.height;
    g.clearRect(0, 0, W, H);
    if (S.overviewBg) g.drawImage(S.overviewBg, 0, 0);
    // zoom window
    const lo = frame - S.zoomHalf, hi = frame + S.zoomHalf + 1;
    g.fillStyle = "rgba(255,255,255,0.25)";
    g.fillRect(lo / S.n * W, 0, Math.max((hi - lo) / S.n * W, 2 * dpr), H);
    // cursor
    g.fillStyle = "#fff";
    g.fillRect(frame / S.n * W - dpr, 0, 2 * dpr, H);
  }

  function drawZoom(frame) {
    const c = els.zoom, g = c.getContext("2d"), dpr = window.devicePixelRatio || 1;
    const W = c.width, H = c.height;
    const lo = frame - S.zoomHalf, span = 2 * S.zoomHalf + 1;
    const cell = W / span;
    const xOf = (f) => (f - lo) * cell;
    g.fillStyle = "#1b1d22";
    g.fillRect(0, 0, W, H);

    const b0 = boutAt(clamp(lo)), b1 = boutAt(clamp(lo + span - 1));
    g.textAlign = "center"; g.textBaseline = "middle";
    g.font = `${12 * dpr}px ui-monospace, monospace`;
    for (let b = b0; b <= b1; b++) {
      const x0 = Math.max(xOf(S.boutStart[b]), 0), x1 = Math.min(xOf(S.boutEnd[b]), W);
      const col = sylColor(S.boutLabel[b]);
      g.fillStyle = col.fill;
      g.fillRect(x0, 0, x1 - x0, H);
      if (x1 - x0 > 22 * dpr) {
        g.fillStyle = col.text;
        g.fillText(String(S.boutLabel[b]), (x0 + x1) / 2, H * 0.62);
      }
      // onset marker
      const xs = xOf(S.boutStart[b]);
      if (xs >= 0 && S.boutStart[b] > 0) {
        g.fillStyle = "#000";
        g.fillRect(xs - dpr, 0, 2 * dpr, H);
      }
    }
    // per-frame ticks when frames are wide enough to see
    if (cell >= 4 * dpr) {
      g.fillStyle = "rgba(0,0,0,0.35)";
      const first = Math.max(0, lo);
      for (let f = first; f < Math.min(S.n, lo + span); f++) {
        g.fillRect(xOf(f), 0, Math.max(dpr / 2, 0.5), H * (f % 10 === 0 ? 0.3 : 0.15));
      }
    }
    // out-of-range shading
    g.fillStyle = "#0e0f12";
    if (lo < 0) g.fillRect(0, 0, xOf(0), H);
    if (lo + span > S.n) g.fillRect(xOf(S.n), 0, W - xOf(S.n), H);
    // current frame
    const cx = xOf(frame), cw = Math.max(cell, 2 * dpr);
    g.strokeStyle = "#fff"; g.lineWidth = 2 * dpr;
    g.strokeRect(cx + (cell - cw) / 2, dpr, cw, H - 2 * dpr);
  }

  function fmtTime(frame) {
    const t = frame / S.fps, m = Math.floor(t / 60), s = t - 60 * m;
    return `${m}:${s.toFixed(3).padStart(6, "0")}`;
  }

  function render() {
    if (!S.labels) return;
    const frame = currentFrame();
    if (sizeCanvas(els.overview)) drawOverviewBg();
    sizeCanvas(els.zoom);
    drawOverview(frame);
    drawZoom(frame);

    const b = boutAt(frame);
    const len = S.boutEnd[b] - S.boutStart[b];
    const prev = b > 0 ? S.boutLabel[b - 1] : "—";
    const next = b < S.boutStart.length - 1 ? S.boutLabel[b + 1] : "—";
    const sw = (id) => id === "—" ? "—" :
      `<span class="chip" style="background:${sylColor(id).fill};color:${sylColor(id).text}">${id}</span>`;
    const pending = S.target !== null
      ? `<div class="pending">seeking… (on screen: ${S.shown})</div>` : "";
    els.readout.innerHTML =
      `<div class="big">frame ${frame.toLocaleString()} <span class="dim">/ ${(S.n - 1).toLocaleString()}</span></div>` +
      `<div class="dim">${fmtTime(frame)}</div>` +
      `<div class="row">syllable ${sw(S.boutLabel[b])}</div>` +
      `<div class="row dim">frame ${frame - S.boutStart[b] + 1} of ${len} in bout · bout ${b + 1} / ${S.boutStart.length}</div>` +
      `<div class="row dim">prev ${sw(prev)} → next ${sw(next)}</div>` + pending;
    els.gotoInput.placeholder = String(frame);
  }

  // ---------- interaction ----------
  function togglePlay() {
    if (els.video.paused) { S.target = null; els.video.play(); } else { els.video.pause(); }
  }
  function stepBout(dir) {
    const f = currentFrame(), b = boutAt(f);
    if (dir < 0) seek(f > S.boutStart[b] ? S.boutStart[b] : S.boutStart[Math.max(b - 1, 0)]);
    else if (b < S.boutStart.length - 1) seek(S.boutStart[b + 1]);
  }
  function step(d) { if (!els.video.paused) els.video.pause(); seek(currentFrame() + d); }

  function wireOverview() {
    let dragging = false, pending = null;
    const frameAt = (e) => {
      const r = els.overview.getBoundingClientRect();
      return (e.clientX - r.left) / r.width * S.n;
    };
    const flush = () => { if (pending !== null) { seek(pending); pending = null; } };
    els.overview.addEventListener("mousedown", (e) => {
      dragging = true; els.video.pause(); seek(frameAt(e));
    });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      pending = frameAt(e); requestAnimationFrame(flush);
    });
    window.addEventListener("mouseup", () => { dragging = false; });
  }

  function wireZoom() {
    let drag = null, pending = null;
    const cellPx = () => els.zoom.getBoundingClientRect().width / (2 * S.zoomHalf + 1);
    const flush = () => { if (pending !== null) { seek(pending); pending = null; } };
    els.zoom.addEventListener("mousedown", (e) => {
      els.video.pause();
      drag = { x: e.clientX, frame: currentFrame(), moved: false };
    });
    window.addEventListener("mousemove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x;
      if (Math.abs(dx) > 3) drag.moved = true;
      if (drag.moved) { pending = drag.frame - Math.round(dx / cellPx()); requestAnimationFrame(flush); }
    });
    window.addEventListener("mouseup", (e) => {
      if (drag && !drag.moved) {             // plain click: jump to the clicked frame
        const r = els.zoom.getBoundingClientRect();
        const lo = drag.frame - S.zoomHalf;
        seek(lo + Math.floor((e.clientX - r.left) / cellPx()));
      }
      drag = null;
    });
  }

  function wireControls() {
    const on = (id, fn) => document.getElementById(id).addEventListener("click", fn);
    on("prev-frame", () => step(-1));
    on("next-frame", () => step(1));
    on("prev-bout", () => { els.video.pause(); stepBout(-1); });
    on("next-bout", () => { els.video.pause(); stepBout(1); });
    on("play", togglePlay);
    els.video.addEventListener("play", () => { document.getElementById("play").textContent = "pause"; });
    els.video.addEventListener("pause", () => { document.getElementById("play").textContent = "play"; });
    els.video.addEventListener("ended", () => { document.getElementById("play").textContent = "play"; });

    const speed = document.getElementById("speed");
    speed.value = "1";
    speed.addEventListener("change", () => { els.video.playbackRate = parseFloat(speed.value); });

    const zoom = document.getElementById("zoom-half");
    zoom.value = String(S.zoomHalf);
    zoom.addEventListener("change", () => { S.zoomHalf = parseInt(zoom.value, 10); render(); });

    const input = document.createElement("input");
    input.type = "number"; input.min = 0; input.max = S.n - 1;
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && input.value !== "") { els.video.pause(); seek(parseInt(input.value, 10)); input.value = ""; input.blur(); }
    });
    document.getElementById("goto-slot").appendChild(input);
    els.gotoInput = input;

    window.addEventListener("keydown", (e) => {
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "select") return;
      const big = e.shiftKey ? 10 : 1;
      if (e.key === "ArrowRight") { step(big); e.preventDefault(); }
      else if (e.key === "ArrowLeft") { step(-big); e.preventDefault(); }
      else if (e.key === "]") { els.video.pause(); stepBout(1); }
      else if (e.key === "[") { els.video.pause(); stepBout(-1); }
      else if (e.key === " ") { togglePlay(); e.preventDefault(); }
    });
    window.addEventListener("resize", render);
  }

  // ---------- startup ----------
  async function start() {
    els = {
      video: document.getElementById("video"),
      overview: document.getElementById("overview"),
      zoom: document.getElementById("zoom"),
      readout: document.getElementById("readout"),
    };
    const d = await (await fetch("/media/data.json")).json();
    S.labels = d.labels; S.fps = d.fps; S.n = d.labels.length;
    buildBouts(S.labels);
    wireControls(); wireOverview(); wireZoom();
    els.video.requestVideoFrameCallback(onFrame);
    seek(0);
  }

  // Dash renders the layout asynchronously; wait for it.
  const wait = setInterval(() => {
    if (document.getElementById("video") && document.getElementById("zoom") && document.getElementById("goto-slot")) {
      clearInterval(wait);
      start();
    }
  }, 50);
})();
