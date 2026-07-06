---
name: patchworklib
description: Use when composing multi-panel matplotlib figures with patchworklib (pw.Brick, pw.hstack / pw.vstack, | and / operators). Covers the default-behavior gotchas, the function-form API, manual row-balancing, and tick-label cleanup.
---

# patchworklib layouts

Recipes for composing multi-panel matplotlib figures with patchworklib
(`pip install patchworklib`). Treats matplotlib axes as "bricks" that
compose with `|` (horizontal) and `/` (vertical) operators.

## The two-form rule

Every operator has a function-form equivalent — `pw.hstack` for `|`,
`pw.vstack` for `/`. The function form takes explicit keyword arguments
(`adjust_width`, `adjust_height`, `margin`, `direction`). Use the
function form whenever default operator behavior bites you, which is
often.

## Global margin

```python
pw.param["margin"] = 0.25  # inches; applies to all subsequent | and /
```

Set this once before composing. Default is 0.5, usually too much for
scientific layouts.

## Gotcha: `/` operator inflates narrower rows

The default `/` operator (and `pw.vstack(..., adjust_width=True)`)
detects mismatched row widths and uniformly scales the narrower row via
`expand(brick, 1/hratio, 1/hratio)`. *Uniform* means both width and
height by the same factor — the wider row stays put, the narrower row
inflates in *both* dimensions. Visible symptom: rows that started at
the same figsize end up at different heights.

Fix: use the function form with `adjust_*=False`:

```python
combined = pw.vstack(row_b, row_a, adjust_height=False, adjust_width=False)
combined = pw.vstack(row_c, combined, adjust_height=False, adjust_width=False)
```

`vstack(a, b)` puts `b` on top of `a`. Build bottom-up by accumulating
into `combined`.

## Gotcha: `pw.spacer()` ignores `pw.param["margin"]`

`pw.spacer.__or__` / `__ror__` set `pw.param["margin"] = None`
internally during composition. Even with margin set to 0 globally,
spacer-bordered compositions use the library's default (≈0.5").

Fix: for true-margin-respecting empty space, use an axis-off Brick:

```python
def empty_brick():
    b = pw.Brick(figsize=(2, 2))  # match the size of the bricks it replaces
    b.axis("off")
    return b

row = a | b | empty_brick()  # respects pw.param["margin"]
```

## Why natural row widths can differ

`figsize=(W, H)` on a Brick means a W×H inch *plot area*. y-axis
labels, tick labels, and titles render OUTSIDE that region — typically
adding 0.4–0.6" per panel to the brick's outer width. A row with 3
labeled panels carries ~3 × 0.5" = 1.5" of decoration; a row with 2
carries ~1.0". Even when plot areas total the same (e.g., 3 × 2 = 6 in
both rows), the *outer* widths differ.

If a design needs all rows to end at the same right edge, options are:

1. Match panel counts across rows.
2. Suppress y-tick labels on some panels: `ax.set_yticklabels([])`.
3. Auto-balance — widen one panel in the narrower row by the measured
   difference (recipe below).

## Recipe: auto-balance rows to equal outer width

```python
target_w = wider_row.get_outer_corner()[1] - wider_row.get_outer_corner()[0]
row_w = narrower_row.get_outer_corner()[1] - narrower_row.get_outer_corner()[0]
diff = target_w - row_w

if diff > 0:
    new_panel_w = original_first_panel_figsize_w + diff
    first_panel.change_plotsize((new_panel_w, original_h))
    narrower_row = first_panel | rest_of_row  # re-compose to pick up new size
```

`Brick.change_plotsize(new_size)` is patchworklib's supported resize
API. It updates both the matplotlib axes position AND the brick's
`_originalsize` attribute. Without updating `_originalsize`, the next
`hstack`/`vstack` call would reset the brick to its previous size
(`hstack` line ~1219 of `patchworklib.py` does this unconditionally at
the start of each composition).

## Width-only scaling exists, but use `change_plotsize`

The underlying `pw.expand(brick, fx, fy)` accepts independent factors,
including `expand(brick, fx, 1)` for width-only scaling. But after the
next re-composition, hstack resets the brick to its `_originalsize` —
silently undoing the scale. Use `Brick.change_plotsize()` instead;
it's the durable version that updates `_originalsize` too.

## Bbox introspection

```python
ox0, ox1, oy0, oy1 = brick.get_outer_corner()   # includes labels/ticks
ix0, ix1, iy0, iy1 = brick.get_inner_corner()   # plot area only
pos = brick.get_position()                       # matplotlib Bbox
```

`get_outer_corner` works on single Bricks and on composed Bricks
objects. Use these for width/height measurements when planning a
layout or computing balance offsets.

## Panel letter labels aligned across rows

The canonical matplotlib idiom for figure-letter labels is
`ax.text(x, y, "A", transform=ax.transAxes, ...)` with x and y in
axes coordinates (0..1). With `transform=ax.transAxes`, x = -0.15
places the label just outside the plot area on the left.

If panels have *different widths*, an axes-coord x like -0.15 lands at
*different figure x positions* in each row (because figure x =
axes_x × plot_width). To align letters in a column across panels, place
labels AFTER auto-balance / final composition and compute the
axes-coord x per panel so each label lands at the same target figure x.
In patchworklib, every Brick's plot area starts at `inner_x0 = 0` in
the basefigure's coord system, so:

```python
def panel_label(ax, letter, target_x):
    inner = ax.get_inner_corner()
    plot_w = inner[1] - inner[0]
    axes_x = (target_x - inner[0]) / plot_w
    ax.text(axes_x, 1.05, letter, transform=ax.transAxes,
            fontsize=28, fontweight="bold", va="top", ha="right")


# After all composition + auto-balance is done:
ax_h_inner = ax_h.get_inner_corner()
target_x = ax_h_inner[0] + (-0.15) * (ax_h_inner[1] - ax_h_inner[0])
panel_label(ax1, "A", target_x)
panel_label(ax4, "B", target_x)
panel_label(ax_h, "D", target_x)
```

`target_x` is in patchworklib's internal inch coords. Pick it by
deciding where the leftmost panel's letter should sit, then compute
per-panel `axes_x` such that the letter lands at that target.

## Tick-label cleanup (matplotlib)

Not patchworklib-specific, but you'll want them with most
patchworklib-composed figures.

```python
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter

ax.yaxis.set_major_locator(MaxNLocator(integer=True))           # whole-number ticks
ax.yaxis.set_major_locator(MultipleLocator(0.2))                # fixed-spacing ticks
ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))        # fixed-precision format
```

Consistent tick label widths across panels reduces width-decoration
variation between rows, making natural widths more predictable.

## Composition checklist

When building a multi-row figure:

1. `pw.param["margin"] = 0.25` (or whatever you want; set once).
2. Build each row with `|`. The default `|` is fine within rows.
3. For empty space within a row, use `empty_brick()` not `pw.spacer()`.
4. Standardize tick labels across panels (helps both visual consistency
   and natural-width consistency).
5. Stack rows with `pw.vstack(adjust_width=False, adjust_height=False)`
   — never the bare `/` operator on the final composition.
6. If you need rows to share outer width, measure with
   `get_outer_corner()` and balance via `change_plotsize()` on the
   first panel of each row that needs widening.
