---
name: bd-paper-figures
description: >-
  Hand-author publication-ready Graphviz figures from a parsed Vivado block-design
  model (the arch_diagrams/model.json produced by the vivado-bd-diagrams skill) and
  embed them as vector figures in a LaTeX paper. Use this whenever the user asks for
  clean, final, publication-quality, journal, or camera-ready diagrams of an FPGA /
  RFSoC / Zynq block design — a simplified system-architecture figure, a
  clocking/synchronization figure, or any filtered view — or asks to clean up the
  auto-generated diagram set for a paper, or to include such diagrams (SVG/PDF) as
  figures in a LaTeX manuscript, even if they never say "Graphviz" or "DOT".
  Complements vivado-bd-diagrams: that skill parses the TCL and emits the full
  scripted diagram set; this one is for hand-written, reader-first figures and the
  paper integration.
---

# Publication figures from a Vivado BD model + LaTeX embedding

Turn `arch_diagrams/model.json` into a small number of hand-authored, publication-ready
diagrams, then place them in the user's LaTeX paper. The scripted generator in
`vivado-bd-diagrams` optimizes for completeness; a journal figure optimizes for a reader
who has never seen the design. That means fewer nodes, functional names, verified
connectivity, and layout you control deliberately — not a prettified netlist dump.

**Prerequisite:** `model.json` must exist. If it doesn't, run the parser from the sibling
skill first (`../vivado-bd-diagrams/scripts/parse_bd.py path/to/create_bd.tcl`). Check
`dot -V`; if Graphviz is missing, read `../vivado-bd-diagrams/references/graphviz_setup.md`.

## Ground rules

- **Never draw a connection that is not in the model.** Every edge must be traceable to
  an `intf_nets` or `nets` entry. Merging (one node standing for a hierarchy) is fine;
  inventing or "probably connected" is not.
- **Only print numbers the design confirms.** Frequencies and rates come from cell
  `props` (e.g. `CLKOUT1_REQUESTED_OUT_FREQ`, `C0.DDR4_InputClockPeriod`) or from
  `CONFIG.*` values in the source `create_bd.tcl` when the parser left `props` empty
  (common for the PS and RFDC). If a derived number contradicts an instance name
  (e.g. computed ui_clk = 300 MHz vs. an instance named `rst_ddr4_200M`), leave the
  number out of the figure and surface the discrepancy to the user.
- **Omissions are design decisions — record and report them.** Dropping interrupts,
  GPIO plumbing, tie-offs, or debug logic from a figure is usually right; say so in the
  final summary so the user can veto.
- **The user asked for hand-authored figures.** Write the DOT yourself. Do not run
  `bd_to_dot.py`; do read its output conventions so the new figures match the family.

## Workflow

### 1 — Extract ground truth

Dump the model before drawing anything:

```bash
python scripts/dump_model.py path/to/arch_diagrams/model.json            # everything
python scripts/dump_model.py path/to/model.json --proc create_root_design
python scripts/dump_model.py path/to/model.json --props clk_wiz ddr4    # cell configs
```

`model.json` structure: top-level `root`, `root_ports`, `root_intf_ports`, `addresses`,
and `parsed`. Each `parsed.create_*` proc holds `cells` (name, vlnv, **props**),
`hier_calls`, `pins`, `intf_nets`, and `nets` (both with `endpoints` lists). Scalar
`nets` are where the clock/reset story lives — a single net's endpoint list *is* the
fan-out of that clock or reset. If a hero cell's `props` are empty, grep the original
`create_bd.tcl` for its `CONFIG.` block.

### 2 — Decide content per figure

Ask what the one job of each figure is, then cut everything that doesn't serve it.
Typical pair for a paper: a **system architecture** (main blocks + data/control/memory
connections; no resets, no interrupts, minimal clocking) and a **clocking/synchronization**
figure (clock sources, buffers/MMCM, sync/CDC paths, clock domains, reset fan-out; no
data detail). Aim for 10–15 nodes per figure. Collapse each BD hierarchy into one node
labeled by function ("RX capture — combiner · capture gate · CDC + buffer FIFO"), and
collapse identical sinks into domain nodes ("DMA · SmartConnect · PS HP (ui_clk domain)").
Functional names beat instance names in a paper; keep instance names out unless the user
wants traceability.

### 3 — Write the DOT

Follow the family conventions (colors, shapes, edge classes, legend) in
`../vivado-bd-diagrams/references/visual_conventions.md`, and the layout techniques in
`references/dot-recipes.md` (read it — it encodes the tricks that make hand layout
converge: pinned rank columns, invisible ordering edges, `constraint=false`,
`xlabel` with ortho splines, port-table quoting, label-collision fixes).

Name files to extend the existing set in the same `arch_diagrams/` folder
(e.g. `8_paper_system_architecture.dot`, `9_paper_clocking_sync.dot`). Skip embedded
titles — journal captions carry that job.

### 4 — Render, look, fix, repeat

```bash
dot -Tpng -Gdpi=110 fig.dot -o fig.png   # raster for your own QA
```

**Actually view the PNG every cycle.** Layout defects are invisible in DOT source.
Per-cycle checklist: no edge routed through a node or cluster it doesn't belong to; no
label sitting on a line; arrows point the way the data/clock flows; external inputs
enter from the canvas edge, not from mid-diagram; legend lists only classes present.
Expect 3–5 cycles; fix one class of problem per cycle.

### 5 — Size for print, then emit vectors

Read the natural size from the SVG header (`<svg width="1286pt" ...`). Effective print
font size = DOT font pt × target width ÷ natural width (IEEE two-column `\textwidth`
≈ 7.16 in, single column ≈ 3.5 in). Keep node text ≥ ~5 pt and edge labels ≥ ~4 pt at
final size; below that, raise fonts (nodes 14, small nodes 13, edges 11) and tighten
`nodesep`/`ranksep` (≈0.35/0.5) before cutting content. Re-run the step-4 loop after any
size change — fonts shift the layout. Then:

```bash
dot -Tsvg fig.dot -o fig.svg
dot -Tpdf fig.dot -o fig.pdf
```

### 6 — Embed in the LaTeX paper

Read `references/latex-integration.md` before touching the manuscript. Short version:
pdflatex cannot embed SVG — check the actual toolchain (`main.log`, Inkscape presence)
before promising `\includesvg`; the robust default is the dot-rendered vector PDF with
the `.svg` copied alongside, `\pdfminorversion=7` in the preamble, `figure*[!t]` at
`\textwidth` for wide diagrams, one reference sentence per figure in the text with the
explanation living in the caption, then compile and **visually verify the rendered pages**.

## Reference outputs in this repo

`vivado/sounder_bbf_sivers_ddr4_2x2/arch_diagrams/8_paper_system_architecture.dot` and
`9_paper_clocking_sync.dot` are finished examples of this workflow (with their SVG/PDF
renders), embedded in the paper as `Images/fpga_system_architecture.*` and
`Images/fpga_clocking_sync.*`. Reuse their structure for new designs rather than
starting from a blank file.
