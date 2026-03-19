---
name: plot-design
description: Research notes on publication-ready scientific plots — technical requirements, useful libraries, and aesthetic best practices. Reference material for future skill development; not yet prescriptive conventions.
---

# Plot Design Research Notes

These are research notes collected in March 2026, not yet finalized conventions. Use as reference when making plotting decisions.

## Publication-Ready Technical Requirements

**Resolution:**
- 300 DPI minimum (universal floor for most journals)
- 600 DPI for line art and graphs
- 1000-1200 DPI for pure B&W line art
- Prefer vector formats (PDF/SVG/EPS) for data plots — resolution-independent and editable

**File formats:**
- Vector (PDF, EPS, SVG) strongly preferred for data plots
- TIFF is the "safe" raster format; PNG for screen display; JPEG is lossy
- Nature accepts JPEG/TIFF/EPS; Cell Press prefers TIFF/PDF; PLOS requires 300-600 DPI

**Dimensions (remarkably consistent across journals):**
- Single-column: ~8.5-8.9 cm (3.3-3.5 in)
- Two-column/full page: ~17-18 cm (6.7-7.1 in)
- Nature: 8.9 cm (1-col) / 18 cm (2-col), max height 24 cm

**Fonts:**
- Sans-serif (Arial, Helvetica) universally accepted; Nature mandates sans-serif
- Minimum 6-8 pt at final print size
- In matplotlib: set `rcParams["pdf.fonttype"] = 42` to embed TrueType fonts (some publishers reject Type 3)

## Aesthetic Best Practices

**Color and accessibility:**
- Nature now requires colorblind-accessible figures; red-green contrasts must be avoided
- Recommended palettes: viridis, cividis, Okabe-Ito (categorical)
- Never rely on color alone — add patterns, shapes, or text as redundant coding
- Colormap libraries: CMasher (perceptually uniform), colorcet (256-color accurate), cmocean (oceanographic)

**Design principles:**
- Maximize data-ink ratio (Tufte): no chartjunk, unnecessary gridlines, 3D effects
- Direct labeling over legends when feasible
- Every axis labeled with units
- Bold panel labels (a, b, c) outside the plot area
- Consistent styling across all figures in a paper
- "As small and simple as is compatible with clarity" (Nature's guideline)

## Useful Libraries

### Style and Sizing
- **SciencePlots**: one-liner matplotlib stylesheets (`plt.style.use('science')`, plus `ieee`, `nature`). Requires LaTeX. Highest single-install impact.
- **tueplots**: generates rcParams dicts for specific venues (ICML, NeurIPS, JMLR) with correct widths/fonts. More principled — derives sizes from actual venue specs.

### Layout and Assembly
- **patchworklib**: Python port of R's patchwork — combine subplots with `|` and `/` operators
- **ProPlot**: matplotlib wrapper with physical units, journal sizing, automatic tight layout. Powerful but learning curve.
- **pylustrator**: drag-and-drop GUI editing of matplotlib figures, saves changes as reproducible code
- **svgutils**: programmatic SVG assembly for combining panels from different sources

### Specific Annoyances
- **statannotations**: significance bars and p-value annotations on seaborn plots. Essential for bio/neuro.
- **starbars**: lighter-weight alternative to statannotations
- **adjustText**: auto-repositions overlapping text labels on scatter plots

### Grammar-of-Graphics
- **plotnine**: full ggplot2 implementation in Python

### Colormaps
- **CMasher**: perceptually uniform, colorblind-friendly scientific colormaps
- **colorcet**: 256-color perceptually accurate maps

## LLM Tools Assessment (March 2026)

No AI tool replaces the full publication-figure workflow end-to-end. The most effective approach is hybrid:

1. Use an LLM to **write the plotting code**
2. Use SciencePlots/tueplots to **set the style**
3. Use statannotations/adjustText to **handle annoyances**
4. Final polish in code or with pylustrator/Inkscape

Notable tools:
- **Plotivy** (plotivy.app): generates matplotlib code from natural language with journal-specific templates. Good starting point.
- **Microsoft LIDA**: open-source four-module system for automated visualization. Best for exploration, not final figures.
- Claude Code plotting skills on mcpmarket.com exist but are essentially prompt packages.
