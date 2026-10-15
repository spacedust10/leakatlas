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

## Layout: a forensic console, not a report

The first version stacked seven white sections under a sticky control bar and read as a generic report.
The redesign (UI/UX Pro Max design-system search: data-dense dashboard, Fira type pairing; its blue/amber
palette and KPI cards were rejected as generic and against the brief) gives the app a console identity:

- **Case-file sidebar** (fixed): brand and thesis, controls grouped by the question they answer
  (scope, defense, evidence, subgroup) under visible labels, the selected record pinned as a case file,
  and a numbered view index that jumps to each card through Panel's shadow roots.
- **Verdict ledger**: one monospace line stating what the current configuration leaks and costs, as
  prose rather than KPI tiles.
- **Cards in investigation order**: who is exposed (Atlas) → is it reproducible, and who is it
  (Stability beside the Inspector) → attack strength (ROC) → cost of defending (Trade-off) → could a
  proxy warn (Proxy vs truth) → who bears it (Subgroups).
- **Quiet chrome**: plot toolbars appear on hover only; 150 ms colour transitions, none under
  `prefers-reduced-motion`; visible focus states on the index.

## The rules the code enforces (`src/leakatlas/theme.py`)

| Rule | Implementation |
|---|---|
| One accent, reserved for "at risk" | `#ff7a45` on the dark surface. Used only for flagged records, the FPR threshold and call-outs. Re-validated for dark mode with the dataviz skill's `validate_palette.js`: worst CVD ΔE 11.7 vs the cividis anchors (≥ 8 target); 7.3:1 text contrast. |
| Sequential = one CVD-safe ramp | cividis from step 91 of 256, the first step at ≥ 3:1 on the near-black surface (its dark blues vanish there). Used for scores / flag-rate across seeds. |
| No categorical hue per defense | 6 defenses exceed the 3-hue all-pairs limit, so defenses are separated by **small multiples and direct labels**, never by color. |
| Neutrals carry everything else | surface `#0f1115`, card `#15181e`, ink `#e9e6df` (15.2:1), secondary `#a7adb8` (8.4:1), muted `#6f7682` (4.1:1), rules `#262a33`. |
| Text never wears a series color | Labels are ink/secondary; marks carry the color. |
| No dual axes, no pies, no KPI tiles, no rainbow, no unlabeled legends | Enforced by construction: every plot has one y-scale; the brief's "flag-rate overlay" is an aligned separate column, not a second axis. |
| Honest axes | Rates start at 0; log axes are labelled "(log10)"; ROC shows the chance diagonal. |
| Relief for low-contrast fills | Members get a 1.5px ink ring; every view has a hover tooltip and the record table. |
| Typography | Fira Code for numbers, labels and ticks (tabular figures); Fira Sans for prose; system fallbacks keep it offline-safe. Scale 11/12/13/15/19/26 px. |
