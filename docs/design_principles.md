# Design principles for the LeakAtlas dashboard

Sources were read before any view was designed (fetched 2026-10-15). Where a page could not be read, that is
stated rather than paraphrased from memory.

## What each reference contributes

### Claus Wilke, *Fundamentals of Data Visualization* (github.com/clauswilke/dataviz, read via clauswilke.com/dataviz)
The repo page lists only chapter sources, so the chapters themselves were read.
- **Redundant coding / CVD:** never rely on hue alone; pair color with shape, outline or a label. Do not put
  confusable hues side by side. → Members carry a dark **outline** *and* sit in their own panel; "at risk" is
  the accent **plus** a label.
- **Direct labeling over legends:** label lines at their ends and regions in place. → ROC panels, the trade-off
  scatter and ridgelines are labeled in place; no free-floating legends.
- **Distributions as distributions:** don't reduce groups to mean ± error bar ("error bars show uncertainty of an
  estimate, not variability in a population"); ridgelines suit many groups and compare *shapes*. → Subgroup view
  draws full score densities; flag-rate CIs are drawn as an *estimate* in a separate aligned column.
- **Uncertainty:** say what an interval is (here: 95% percentile/BCa bootstrap CI over records); don't judge
  differences by overlapping bars, compare the CI of the difference. → Gap CIs are reported, not eyeballed.
- **Axes:** ratios and quantities spanning orders of magnitude belong on log scales; label the axis with the
  variable name and state the base. → Log-log ROC ("FPR (log10)"), gradient norm and loss ratio in log10.
- **Titles state the point**, axis titles state units, and labels that are self-explanatory need no title.

### Rohan Alexander, *Telling Stories with Data* (tellingstorieswithdata.com, preface)
- Communicate so others can **understand and trust** the result; be prescriptive about **reproducibility**.
  → Every headline is computed from DuckDB by the same code `make reproduce` runs; the dashboard footer shows
  the query source; nothing is hard-coded.

### Nadieh Bremer, Visual Cinnamon portfolio (visualcinnamon.com/portfolio)
The page describes projects at the level of titles/filters, so only these patterns are taken from it:
- **Choose the form per question, go beyond defaults** (beeswarm, bespoke forms). → Stability is a beeswarm
  strip; the hero is a custom split atlas, not a default scatter.
- **Short framing title + one-line blurb per piece; curated summary above full index.** → Each view leads with
  a one-sentence data-computed takeaway; the hero Record Atlas comes first.
- **Show the method next to the output.** → Each view has a one-line "how to read" note (threshold source,
  CI method, seeds pooled or not).

### FlowingData (flowingdata.com)
**Could not be read:** the site returned HTTP 403 to automated fetches (homepage and an article URL). Its
annotation-led approach is therefore applied only in the form the brief itself specifies: lead with an
annotated one-sentence takeaway, annotate notable points (top-5 call-outs, threshold label written on the
line), and favor direct labels.

### Distill, "How to Use t-SNE Effectively" (distill.pub/2016/misread-tsne)
- **One plot is not enough; parameters and seeds change the picture; check run-to-run variation.**
  → Global controls for **seed / all seeds, defense, ε and FPR threshold**; changing them re-renders every view
  and the takeaway sentence, so the reader watches whether "this record is at risk" survives. "All seeds" is
  the default and a single seed is labeled as such. The Stability Strip exists to show which flags are
  one-run noise.

## The rules the code enforces (`src/leakatlas/theme.py`)

| Rule | Implementation |
|---|---|
| One accent, reserved for "at risk" | `#c2410c` (burnt orange). Used only for flagged records / the FPR threshold. Validated with the dataviz skill's `validate_palette.js`: worst CVD ΔE 11.5 vs the cividis anchors (≥ 8 target), normal-vision ΔE 17.4. |
| Sequential = one CVD-safe ramp | cividis, truncated to its lower 85% (the top yellow is 1.2:1 on the light surface). Used for scores / flag-rate across seeds. |
| No categorical hue per defense | 6 defenses exceed the 3-hue all-pairs limit, so defenses are separated by **small multiples and direct labels**, never by color. |
| Neutrals carry everything else | ink `#16161a`, secondary `#55555e`, muted `#9a9aa3`, rules `#e4e4e7`, surface `#fcfcfb`. |
| Text never wears a series color | Labels are ink/secondary; marks carry the color. |
| No dual axes, no pies, no KPI tiles, no rainbow, no unlabeled legends | Enforced by construction: every plot has one y-scale; the brief's "flag-rate overlay" is an aligned separate column, not a second axis. |
| Honest axes | Rates start at 0; log axes are labelled "(log10)"; ROC shows the chance diagonal. |
| Relief for low-contrast fills | Members get a 1.5px ink ring; every view has a hover tooltip and the record table. |
| Typography | System sans stack (no web-font fetch, works offline); hierarchy 22/15/12px, weights 650/500/400; takeaways in ink, notes in secondary. |
