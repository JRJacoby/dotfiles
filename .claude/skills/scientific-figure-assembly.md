# Scientific Figure Assembly

Build publication-quality multi-panel scientific figures using a two-phase
workflow: data preparation (Python, slow, run once) then rendering and
layout iteration (HTML/CSS/Plotly.js, instant refresh).

## When to use

When building publication-quality scientific figures from analysis outputs.

## Workflow

### Phase 1: Data preparation (Python)

1. Analysis scripts save pickled DataFrames alongside their PNG outputs
2. A `pickle_to_json.py` converter transforms pickles into a single JSON
   file that the browser can load
3. Pre-aggregate where possible (compute mean/SEM in Python, not JS)
4. Normalize keys (e.g., genotype display names) and color strings
   during conversion
5. This phase runs once per data change (~seconds for pickled data)

### Phase 2: Rendering and layout (HTML + CSS + Plotly.js)

1. A static HTML page loads the JSON and renders each panel with Plotly.js
2. CSS flexbox handles all layout (gaps, margins, padding, sizing)
3. Plotly renders plots at container size — no pre-rendered images
4. Edit CSS or JS constants, refresh browser, see changes instantly
5. No build tools, no server framework — just `python -m http.server`

### File structure

```
scripts/<date>_<name>/
  pickle_to_json.py          # Python: pickle → JSON
  editor/
    index.html               # Layout (CSS flexbox) + init script
    render.js                 # Plotly renderer functions + constants
    figure_N_config.json      # Panel layout config (checked in)
    figure_N_data.json        # Generated data (gitignored)
```

## Layout: CSS flexbox

**Do not use matplotlib or any plotting library for multi-panel layout.**
Use CSS flexbox. It provides:

- `gap` — space between panels (absolute, does not affect panel size)
- `padding` — shrinks content inside a panel without affecting neighbors
- `margin` — pushes neighbors away without affecting content
- `flex: N` — relative sizing weights
- `max-width` / `max-height` — caps on specific elements
- `min-width` — ensures panels don't get too small

### Key patterns

```css
body { width: 8.5in; height: 9.5in; }  /* fixed figure size */

.figure {
  display: flex;
  flex-direction: column;
  padding: 0.1in;    /* outer margins */
  gap: 0.1in;        /* vertical gap between rows */
}

.row {
  display: flex;
  flex: 1;           /* equal height rows by default */
  gap: 0.1in;        /* horizontal gap between columns */
  min-height: 0;
  overflow: hidden;
}

.panel {
  flex: 1;           /* equal width by default */
  position: relative; /* for panel labels */
  min-width: 0;
  min-height: 0;
}
```

To override defaults on specific elements:

```html
<div class="row" style="flex: 1.3">        <!-- taller row -->
<div class="panel" style="flex: 2">        <!-- wider column -->
<div class="row" style="gap: 0">           <!-- remove auto-gap -->
<div class="panel" style="min-width: 250px"> <!-- minimum size -->
```

## Plotly: use for the plot area only

Use Plotly only for data rendering (lines, scatter, heatmaps, axes,
ticks). Handle everything else in HTML/CSS:

### Externalize axis labels

Put x and y axis labels in HTML, not Plotly. This ensures labels align
across panels regardless of Plotly's internal margin behavior.

```html
<div class="y-label">Syllable Rate (Hz)</div>
<div class="panel" id="my-plot"></div>
```

```css
.y-label {
  max-width: 20px;
  writing-mode: vertical-rl;
  transform: rotate(180deg);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: var(--font-axis-label);
  flex-shrink: 0;
}
```

Set `yaxis.title: ""` and `xaxis.title: ""` in Plotly layout to suppress
the built-in labels.

### Externalize colorbars for heatmaps

Plotly's built-in colorbar competes with the plot area for space inside
the container. Instead, render heatmaps with `showscale: false` and
build colorbars as CSS gradient divs in their own flex column:

```html
<div class="heatmap-group" style="display: flex; gap: 2px">
  <div class="panel" id="panel-G" style="flex: 1"></div>
  <div class="colorbar" id="colorbar-G" style="width: 38px"></div>
</div>
```

Build the gradient in JS:
```javascript
grad.style.background = `linear-gradient(to top, ${colorLow}, ${colorMid} 50%, ${colorHigh})`;
```

Note: Plotly's built-in colorbar is fine for panels where the colorbar
doesn't need to be separated (e.g., a standalone heatmap). Only
externalize when the colorbar is squishing the plot area.

### Panel labels

Position with CSS absolute inside each panel div:

```css
.panel-label {
  position: absolute;
  top: 2px;
  left: 2px;
  font-weight: bold;
  font-size: 14pt;
  background: rgba(255, 255, 255, 0.7);
  padding: 0 3px;
  z-index: 10;
  pointer-events: none;
}
```

```html
<div class="panel" id="panel-A"><span class="panel-label">A</span></div>
```

## Font size standardization

Define all sizes in pt (typographic points). Convert to px for Plotly
since it only accepts px. CSS uses pt natively.

```javascript
// render.js
const PT = 96 / 72;  // px per pt
const FONT_PT = {
  axisLabel: 9,
  tickLabel: 7,
  colorbarTitle: 7,
  colorbarTick: 6,
  subtitle: 8,
};
const AXIS_FONT = FONT_PT.axisLabel * PT;
const TICK_FONT = FONT_PT.tickLabel * PT;
```

```css
/* index.html — must match FONT_PT in render.js */
:root {
  --font-axis-label: 9pt;
  --font-tick-label: 7pt;
  --font-colorbar-title: 7pt;
  --font-colorbar-tick: 6pt;
}
```

**`9pt` in CSS and `9` in Plotly are NOT the same.** CSS pt is a real
typographic point (1/72 inch). Plotly font size is in pixels. `1pt ≈ 1.33px`.
Always convert via the `PT` constant.

## Plotly renderer patterns

### Base layout function

Centralize defaults. Destructure opts to merge xaxis/yaxis/margin
correctly:

```javascript
function baseLayout(data, opts = {}, config = {}) {
  const { xaxis: xOpts, yaxis: yOpts, margin: mOpts, ...restOpts } = opts;
  return {
    font: { family: "Arial, Liberation Sans, sans-serif", size: TICK_FONT },
    margin: { l: 30, r: 10, t: 10, b: 40, ...mOpts },
    xaxis: { title: "", tickfont: { size: TICK_FONT }, ...xOpts },
    yaxis: { title: "", tickfont: { size: TICK_FONT }, ...yOpts },
    autosize: true,
    ...restOpts,
  };
}
```

**Gotcha:** If you spread `...opts` at the top level AND have `xaxis`
in opts, the spread overwrites the entire merged xaxis object. Always
destructure xaxis/yaxis/margin out first, spread them inside their
respective blocks, then spread `...restOpts` at the top level.

### Shared y-axis panels

For panels that share a y-axis:
- Compute the shared y-range in `pickle_to_json.py` (include as `_yRange`)
- First panel shows tick labels, others suppress them
- Reduce left margin on non-first panels

### Heatmaps with equal aspect ratio

For spatial data, use `scaleanchor` and consider masking non-significant
values:

```javascript
yaxis: { scaleanchor: "x", scaleratio: 1 }
// Mask non-significant values to null → renders as white
const masked = grid.map(row => row.map(v => Math.abs(v) < threshold ? null : v));
```

### Custom colorscales

Define stops as RGB arrays for both line coloring and Plotly colorbars:

```javascript
const STOPS = [[0.0, [48,18,59]], [0.5, [163,220,53]], [1.0, [122,4,3]]];

function sampleColorscale(t, stops) { /* interpolate at position t (0-1) */ }

// For Plotly colorbar:
const plotlyScale = stops.map(([pos, [r,g,b]]) => [pos, `rgb(${r},${g},${b})`]);
```

### HTML overlay legends

When Plotly's built-in legend can't fit in one line or needs to span
multiple panels, build the legend as an HTML `<div>` with `position:
absolute` overlaid on the panel row. Use CSS `repeating-linear-gradient`
for hatched pattern swatches. Set `overflow: visible` on the parent
row so the legend can extend outside.

```html
<div style="position: absolute; top: -11px; left: 40px; display: flex;
            gap: 8px; font-size: 6pt; z-index: 20; ...">
  <span>
    <span style="width:10px; height:8px; background:#999"></span>
    Label
  </span>
  <!-- hatched swatch -->
  <span>
    <span style="background:repeating-linear-gradient(45deg,
      #2ca02c,#2ca02c 2px,white 2px,white 4px)"></span>
    Observed
  </span>
</div>
```

### Aligning axis labels across panels with different tick lengths

Plotly's `automargin` ignores `automargin: false` when an axis title
is present — it still adjusts position based on tick label height.
Workaround: suppress the Plotly axis title (`title: ""`) and use a
paper-coordinate annotation instead:

```javascript
layout.annotations = [{
  text: "Syllable",
  xref: "paper", yref: "paper",
  x: 0.5, y: -0.22,  // fixed y — same for both panels
  xanchor: "center", yanchor: "top",
  showarrow: false,
  font: { size: AXIS_FONT },
}];
```

This pins the label to an absolute position regardless of tick label
length. Tune the `y` value once.

### Shared y-axis scale across side-by-side panels

Compute the shared max across both panels' data in `pickle_to_json.py`
and store it in the JSON. Pass it via config and set
`yaxis.range = [0, sharedMax]` in the renderer.

## Gotchas

### Plotly ignores flex container height

Plotly renders at ~450px default height regardless of its container's
flex-assigned height. Set `overflow: hidden` on `.row` to clip it to
the correct size. Use `min-width` on panels to ensure they get enough
horizontal space.

### Colorbar on invisible trace

When creating a dummy trace just for a colorbar, `colorbar` must be
nested inside `marker`, not as a sibling property. Use real values in
the `marker.color` array (not null) and `size: 0.001` to hide markers.

### CSS gap + Plotly margin stack

Both create visual spacing. With `gap: 0`, adjacent panels still have
visual space from Plotly's internal margins adding up (panel A's
`margin.r` + panel B's `margin.l`). This can be useful — or confusing.

### preserveAspectRatio distorts text

Never use `preserveAspectRatio="none"` on SVG elements. It stretches
text characters horizontally, making them look squished. Let Plotly
autosize instead.

### pt vs px in SVG exports

SVG files saved with `width` in `pt` are misinterpreted by some
applications (e.g., Illustrator reads them at ~80% expected size).
If exporting SVGs for external tools, post-process to convert `pt`
units to `in`.
