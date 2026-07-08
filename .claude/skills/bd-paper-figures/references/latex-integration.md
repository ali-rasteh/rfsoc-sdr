# Embedding the figures in a LaTeX paper

The goal: vector figures in the manuscript, zero new build warnings, references that
resolve, and a visual check of the actual typeset pages — without breaking the user's
existing build or "fixing" things they didn't ask about.

## 1 — Check the toolchain before promising SVG

LaTeX cannot embed SVG natively. `\usepackage{svg}` + `\includesvg` works only with
Inkscape installed **and** unrestricted shell-escape. Verify before choosing a route:

- `main.log` header shows the engine and escape mode: `restricted \write18 enabled`
  means no shell-escape → `\includesvg` will fail.
- Check Inkscape: `Get-Command inkscape` / `which inkscape`.
- Overleaf has both and `\includesvg` works there; a local MiKTeX/TeX Live build
  usually doesn't.

**Robust default:** embed the dot-rendered vector **PDF** (`dot -Tpdf`) via
`\includegraphics` — identical vector content to the SVG — and copy the `.svg` next to
it in the image directory so the source of truth travels with the paper. If the user
asked for "SVG" specifically, explain this constraint in the summary rather than
silently substituting.

## 2 — Files and preamble

- Copy into the paper's existing image directory with descriptive names
  (`Images/fpga_system_architecture.pdf` + `.svg`), matching the naming style already
  in use.
- Graphviz emits PDF 1.7; pdflatex defaults to 1.5 and warns on every inclusion. Add
  once, right after `\documentclass`:

```latex
\pdfminorversion=7 % allow inclusion of PDF 1.7 figures (Graphviz output) without warnings
```

## 3 — Figure environments

Wide LR diagrams are legitimately full-width figures — don't fight it:

```latex
% Vector sources: Images/fpga_clocking_sync.svg (PDF used for pdflatex compatibility)
\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{Images/fpga_clocking_sync.pdf}
\caption{Clocking and synchronization architecture ... (2–3 sentences that actually
explain the figure: what the reader should see, key frequencies, what crosses where).}
\label{fig:fpga_clocking}
\end{figure*}
```

- `figure*` for two-column classes (IEEEtran); it floats to a page top — that's normal.
- Keep embedded titles out of the DOT; the caption is the title.
- Match the manuscript's existing reference style (`Fig.~\ref{...}` vs `Fig.\ref{...}`)
  and label conventions before adding new ones.

## 4 — Text edits: reference, don't retell

When the user says "briefly explain in the text", add roughly one sentence per figure
at the paragraph that already discusses that subsystem, e.g. "The resulting clock
distribution, synchronization, and reset architecture is summarized in
Fig.~\ref{fig:fpga_clocking}." Put the actual explanation in the caption. Place each
`figure*` block in the source near its first reference; LaTeX renumbers any later
figures automatically since everything goes through `\ref`.

## 5 — Verify the build

1. Compile the way the user does (check `main.log` / `latexmkrc`); default:
   `pdflatex -interaction=nonstopmode main.tex`, twice, so references settle.
2. **Pre-existing failures are not yours to absorb or to fix silently.** If
   `-halt-on-error` aborts on something like an invalid UTF-8 byte in `main.bbl`,
   check whether the error predates your change (previous `main.pdf` exists, error is
   in a file you never touched) — then build in plain nonstopmode like the user does,
   and *report* the pre-existing issue in your summary.
3. Grep the fresh log for:
   - `used on input line` for each new figure file (confirms inclusion),
   - `undefined` (references must resolve after pass 2),
   - `PDF inclusion` warnings (should be gone after `\pdfminorversion=7`).

## 6 — Look at the typeset pages

Never declare figure placement done from the log alone. Rasterize the pages around the
insertion point and view them:

```powershell
$gs = Get-Command mgs, gswin64c, gs -ErrorAction SilentlyContinue | Select-Object -First 1
& $gs.Source -dBATCH -dNOPAUSE -sDEVICE=png16m -r100 -dFirstPage=8 -dLastPage=10 `
  -sOutputFile="$scratch\page%d.png" main.pdf
```

(`mgs` is MiKTeX's bundled Ghostscript.) Check: figure spans the intended width, caption
sits with it, in-figure text is legible relative to body text, and the in-text
"Fig. N" numbers match the rendered captions.

## Report back

Summarize: where the figures landed (page numbers), the label names, any intentional
deviations (PDF instead of SVG and why), preamble changes, and any pre-existing build
issues you noticed but left alone.
