---
name: scientific-figures
description: Design theory and principles for scientific figures — narrative structure, visual hierarchy, color, composition, and common pitfalls. Use when planning or reviewing figure layouts.
---

# Scientific Figure Design: Theory and Principles

Use this skill when planning figure content and layout, reviewing draft figures, or deciding how to present data visually.

## 1. A Figure Is an Argument

A figure is not a data dump. It is a visual argument with a claim and evidence.

**Before touching any tool, answer these three questions:**
1. What is the one-sentence message of this figure?
2. What evidence (data) directly supports that message?
3. What would distract from it?

If a panel doesn't serve the message, cut it. If a visual element doesn't encode data or guide interpretation, remove it. Everything in the figure should earn its ink.

> "Above all else, show the data." — Edward Tufte

### The data-ink ratio

Tufte's core metric: what fraction of the ink on the page represents data? Maximize it. Erase non-data ink (decorative borders, background fills, redundant gridlines, 3D effects). Erase redundant data-ink (if the same information is encoded twice, remove one encoding). But don't be dogmatic — a light reference band or annotation that aids interpretation is justified even though it's "non-data."

---

## 2. Narrative Structure

A multi-panel figure should read like a paragraph, not a gallery wall. Panels are sentences; the figure is the argument.

**Reading order**: Audiences read left-to-right, top-to-bottom (in Western conventions). Arrange panels so the logical flow follows this natural scan path. Don't force readers to jump around.

**Three narrative patterns for multi-panel figures:**

| Pattern | Structure | When to use |
|---------|-----------|-------------|
| **Setup → Evidence → Conclusion** | A shows context/method, B-D show data, E synthesizes | Most common; works for any result |
| **Overview → Zoom** | A shows the big picture, B-F drill into components | When readers need global context before details |
| **Parallel comparison** | Rows or columns show the same analysis across conditions | When the point is contrast between groups/methods |

**The ABT test**: Can you describe your figure as "We observed A, BUT B, THEREFORE C"? If yes, the figure has narrative tension and resolution. If it's just "A, and B, and C, and D..." it's a list, not a story.

**Panel count**: More panels ≠ more convincing. Each panel competes for attention. A 4-panel figure where every panel is clear beats an 8-panel figure where panels are cramped and half are filler. When in doubt, move panels to supplementary figures.

---

## 3. Visual Hierarchy and Composition

### Guide the eye

Not all panels are equally important. The most critical result should be visually dominant — larger, more saturated, or positioned at top-left (first thing seen).

**Hierarchy tools:**
- **Size**: Make the key panel larger than supporting panels
- **Color saturation**: Key data in vivid color; context in gray or muted tones
- **Position**: Top-left gets seen first; bottom-right is the conclusion position
- **Whitespace**: Generous spacing around important elements draws attention to them

### Panel layout principles

- **Consistent alignment**: Use a grid. Panels should share edges, not float loosely. Misaligned panels look amateur.
- **Shared axes**: When panels show the same variable, share and align the axis. Redundant axis labels waste space and slow comparison.
- **Uniform spacing**: Keep horizontal and vertical gaps equal between all panels. Panels should never touch.
- **Panel labels**: Bold uppercase letters (A, B, C) in the upper-left corner of each panel. Consistent size. Distinct from axis labels.
- **Aspect ratios**: Match the aspect ratio to the data. Time series → wide. Scatter → square. Heatmaps → match the data matrix shape.

### Whitespace is not wasted space

Whitespace separates logical groups, reduces cognitive load, and makes figures feel composed rather than crammed. Resist the urge to fill every pixel.

---

## 4. Color

### The three palette types

| Type | Data kind | Examples |
|------|-----------|---------|
| **Sequential** | Ordered continuous values (intensity, age, concentration) | viridis, magma, inferno, batlow |
| **Diverging** | Values above/below a meaningful center (fold change, residuals) | RdBu, PiYG, coolwarm |
| **Qualitative** | Unordered categories (genotypes, conditions) | Okabe-Ito, Tol, ColorBrewer Set2 |

**Match palette to data type.** Sequential data with a qualitative palette obscures ordering. Categorical data with a sequential palette implies false ordering.

### Colorblind accessibility is mandatory

~8% of men have some form of color vision deficiency. Red-green is the most common confusion axis.

**The Okabe-Ito palette** (recommended by Nature Methods):
- Orange `#E69F00`, Sky Blue `#56B4E9`, Bluish Green `#009E73`, Yellow `#F0E442`, Blue `#0072B2`, Vermillion `#D55E00`, Reddish Purple `#CC79A7`, Black `#000000`
- Safe for protanopia, deuteranopia, and tritanopia
- Works in grayscale

**Safe pairings**: Blue + Orange (universally distinguishable), Blue + Red, Purple + Yellow.
**Avoid**: Red + Green, Green + Brown, Light Green + Yellow.

### Color principles

- **Never rely on color alone.** Always add a redundant encoding: shape, line style, pattern, or direct label. The figure should still work in grayscale.
- **Limit categories.** The eye pre-attentively distinguishes ~5 colors. Beyond that, add shapes or facet into sub-panels.
- **Use color purposefully.** If everything is colorful, nothing stands out. Keep context/reference data in gray; reserve vivid color for the key message.
- **Avoid rainbow/jet colormaps.** They are not perceptually uniform (equal data intervals don't map to equal perceived color differences), they create false boundaries, and they fail for colorblind viewers.
- **Respect conventions.** If your field uses blue=cold/red=hot or specific genotype colors, follow them unless there's a strong reason not to.

---

## 5. Choosing the Right Plot Type

### The cardinal rule: show the data

Bar charts of continuous data are the single most criticized visualization in modern science. They hide the distribution, create perceptual bias (viewers assume data is inside the bar), and are meaningless for small samples.

**Instead of bar + error bar, use:**

| Sample size | Recommended | Why |
|------------|-------------|-----|
| n < 20 | Strip/dot plot with all individual points, mean/median line | Every observation visible; nothing hidden |
| n = 20-60 | Dot plot + box plot overlay | Shows distribution shape + individual points |
| n > 60 | Violin plot or box plot (points optional) | Distribution shape is the message |

### Common plot types and when they work

| Plot type | Best for | Watch out for |
|-----------|----------|---------------|
| **Scatter** | Two continuous variables, correlations | Overplotting — use transparency or density |
| **Line** | Time series, trajectories, dose-response | Implies continuity — don't use for categorical x-axis |
| **Box plot** | Comparing distributions across groups | Hides modality — add points or use violin |
| **Violin** | Distribution shape comparison | Can look odd with small n; always add sample size |
| **Heatmap** | Matrix data, expression, usage over time | Needs good colormap; row/column ordering matters enormously |
| **Bar** | Counts, proportions, percentages | NEVER for continuous measurements with error bars |
| **Histogram** | Single distribution shape | Bin width changes the story — try multiple or use KDE |
| **Paired lines** | Before/after, repeated measures | Shows individual trajectories; highlight the group trend |

### Error bars and uncertainty

- **Always define what error bars represent** in the caption: SD, SEM, 95% CI, or bootstrap CI. "Error bars" alone is meaningless.
- SEM shrinks with n and tells you about the mean estimate; SD tells you about the spread of the data. Choose based on your message.
- For small n, show all individual points rather than summary statistics. "Summary statistics are only meaningful when there are enough data to summarize."
- Confidence bands (e.g., LOESS with bootstrap CI) are often more informative than point-wise error bars for trajectory data.

---

## 6. Typography and Labeling

- **Axis labels**: Always include units. "Velocity" is incomplete; "Velocity (mm/s)" is correct.
- **Font size**: Must be legible at final print size (typically 6-8 pt minimum after the figure is scaled to column width). Test by printing.
- **Direct labeling > legends**: When feasible, label data series directly on or near the data rather than forcing readers to cross-reference a legend. This reduces cognitive load.
- **Titles as messages**: Panel titles should state the conclusion ("SynGAP1 mice are hyperactive") not just describe the data ("Velocity by genotype"). Reserve descriptive titles for methods figures.
- **Captions**: Write captions that make the figure self-contained. A reader should understand the figure without reading the main text. Include: what each panel shows, what statistical tests were used, what error bars represent, sample sizes.
- **Avoid abbreviations** in labels unless they're standard in the field. Spell out on first use in the caption.

---

## 7. Common Mistakes (and What Reviewers Flag)

### Content mistakes
1. **Dynamite plots** (bar + error bar for continuous data) — the #1 complaint in modern biology
2. **Missing sample sizes** — always report n per group, either on the axis, in the caption, or in the panel
3. **Undefined error bars** — "Error bars represent..." must appear in every caption
4. **Rainbow colormaps** — perceptually non-uniform and inaccessible
5. **3D effects on 2D data** — distort perception, never justified
6. **Pie charts** — angles are hard to compare; use bar charts for composition data

### Layout mistakes
7. **Too many panels** — dilutes the key finding; move supporting evidence to supplement
8. **Inconsistent formatting** — different fonts, colors, or axis styles across panels
9. **Cramped panels** — text too small, no whitespace, panels touching
10. **Illogical panel order** — panels don't follow the narrative arc described in the text

### Technical mistakes
11. **Raster artifacts** — PNG/JPEG that pixelate at print size; use vector formats for data plots
12. **Text too small after scaling** — design at final print dimensions, not at screen zoom
13. **Color as only differentiator** — fails for colorblind readers and grayscale printing

---

## 8. The Figure Design Checklist

Use this when reviewing a figure before finalizing:

**Message**
- [ ] Can you state the figure's message in one sentence?
- [ ] Does every panel contribute to that message?
- [ ] Would removing any panel weaken the argument?

**Data representation**
- [ ] Is each plot type appropriate for its data?
- [ ] Are individual data points visible (for n < ~60)?
- [ ] Are error bars defined in the caption?
- [ ] Are sample sizes reported?

**Visual design**
- [ ] Does the panel layout follow a logical reading order?
- [ ] Is the most important result visually prominent?
- [ ] Are axes shared where appropriate?
- [ ] Is the color palette colorblind-accessible?
- [ ] Does the figure work in grayscale?
- [ ] Is there a redundant encoding beyond color?

**Typography**
- [ ] All axes labeled with units?
- [ ] Font legible at final print size?
- [ ] Panel labels bold, consistent, upper-left?
- [ ] Caption makes the figure self-contained?

**Polish**
- [ ] No chartjunk (unnecessary gridlines, 3D, decorative elements)?
- [ ] Consistent formatting across all panels?
- [ ] Sufficient whitespace between panels?
- [ ] Exported in vector format (or ≥300 DPI raster)?

---

## Key References

- Rougier, Droettboom & Bourne (2014). "Ten Simple Rules for Better Figures." PLOS Computational Biology.
- Tufte, E. (1983/2001). *The Visual Display of Quantitative Information.*
- Weissgerber et al. (2015). "Beyond Bar and Line Graphs." PLOS Biology.
- Weissgerber et al. (2022). "Reveal, Don't Conceal: Transforming Data Visualization to Improve Transparency." Circulation.
- Wong, B. (2010-2013). "Points of View" column series. Nature Methods.
- Rougier & Roldán (2021). "Let Us FIGURE It Out." PLOS Computational Biology.
- Okabe & Ito (2008). Color Universal Design palette.
