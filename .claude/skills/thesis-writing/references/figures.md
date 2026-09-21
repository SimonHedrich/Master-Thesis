# Figures — choosing the medium, and laying out image panels

Two separate decisions live here: which medium carries a numeric comparison, and how a
multi-panel image figure is built once a figure is the right call. Neither is covered by
`claims.md` (which governs traceability once the medium is chosen) or `mechanics.md` §3 (which
covers referencing and captioning mechanics once the float exists).

## 1. Medium choice: prose, table, or plot

**Rule:** the reader gets the cheapest medium that still delivers the insight.

- **Two numbers → prose.** "X reaches 0.41 mAP against Y's 0.37" is a sentence, not a table row.
  A table of two numbers wastes a float on what one clause already said.
- **More numbers, where the *comparison itself* isn't the point → table.** A cost/throughput
  breakdown, a per-class count, a ranking with several columns — the reader looks up a value or
  scans a column, they don't need to perceive a shape. If the reader's task is "find the number
  for X," it's a table.
- **A plot only earns its place when prose or a table would lose the insight.** That happens
  when what matters is a *relationship among values* that only reads at a glance: a trend across
  an ordered axis, a cluster or outlier, a trade-off frontier (cost vs. accuracy), a
  distribution's shape, or a dense grid where color does the scanning a table column can't.
  *Test: if you deleted the plot and left the table it was built from, would the reader still
  see what the plot was for? If yes, the table alone was enough.*

This is the same reasoning `4-Results.tex` already applies without stating it — e.g.
`fig:generator-cost-vs-map` (cost vs. accuracy trade-off, a frontier no table row conveys) and
`fig:generator-per-class-heatmap` (12 cells × many classes, a table nobody would scan) sit next
to `tab:generator-ranking` (a ranked lookup table, correctly not a plot). Apply this test before
adding a new figure, not after.

## 2. Multi-panel image figures

When the content is images rather than numbers — qualitative samples, side-by-side comparisons,
photographs — use `subcaption` (already loaded in `main.tex`) so each panel gets its own
caption and label while the group gets one outer caption. This pattern is lifted directly from
`thesis/docs/old_bachelor_thesis/chapters/3-Methods_and_Implementation.tex`, the author's own
prior thesis, kept in-repo as a worked example — read it there for more variants before
inventing a new layout.

**Three-across** (`old_bachelor_thesis/chapters/3-Methods_and_Implementation.tex:20-44`):

```latex
\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.31\textwidth}
        \centering
        \includegraphics[width=1\textwidth]{figures/images/<file-a>.png}
        \caption{<short panel caption>}
        \label{fig:<slug>-a}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.31\textwidth}
        \centering
        \includegraphics[width=1\textwidth]{figures/images/<file-b>.png}
        \caption{<short panel caption>}
        \label{fig:<slug>-b}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.31\textwidth}
        \centering
        \includegraphics[width=1\textwidth]{figures/images/<file-c>.png}
        \caption{<short panel caption>}
        \label{fig:<slug>-c}
    \end{subfigure}
    \caption{<one sentence saying what the group of panels together shows>}
    \label{fig:<slug>}
\end{figure}
```

**Two-across** uses `0.49\textwidth` panels instead of `0.31\textwidth`, same `\hfill` pattern.

**Renaming the float for a themed group** — when several image figures in a row form one
category (a generated-sample gallery, a qualitative-failure set), add
`\renewcommand{\figurename}{<Category>}` right after `\centering` in the *first* figure of the
group, so the List of Figures reads as a distinct section rather than an undifferentiated
sequence of "Figure N". `old_bachelor_thesis` does this for "Plots" and "Images"
(`3-Methods_and_Implementation.tex:70`, `4-Results.tex:21`). Revert with another
`\renewcommand` before the next unrelated figure, or it leaks into every figure that follows.

**Every panel still gets its own `\label`** even though only the outer caption is normally
`\Cref`-referenced from prose — an individual panel is occasionally worth pointing at on its own
(e.g. "the FLUX.2-klein panel in \Cref{fig:generator-gallery-kinkajou}").

**Placement:** `[H]` (hard "here", needs the `float` package, already loaded) is what
`old_bachelor_thesis` uses for image galleries specifically, as opposed to `[htbp]` for
single-image floats elsewhere in that same document — a deliberate distinction, since a
multi-panel comparison loses its point if LaTeX separates it from the sentence introducing it.
