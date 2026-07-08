# DOT layout recipes for hand-authored publication figures

Techniques that make hand layout converge in a few render cycles instead of many.
Colors, shapes, and edge classes live in
`../../vivado-bd-diagrams/references/visual_conventions.md`; base Graphviz gotchas in
`../../vivado-bd-diagrams/references/graphviz_setup.md`. This file is about *controlling*
the layout, which the scripted generator never needed to do.

## Skeleton

```dot
digraph "figure_name" {
  graph [rankdir=LR, splines=ortho, nodesep=0.35, ranksep=0.5, fontname="Helvetica",
         bgcolor="white", forcelabels=true, pack=false];
  node  [fontname="Helvetica", fontsize=14, shape=box, style="filled,rounded",
         penwidth=1.4, margin="0.14,0.09"];
  edge  [fontname="Helvetica", fontsize=11, penwidth=1.4, arrowsize=0.7];
}
```

`splines=ortho` gives the engineering-drawing look but **silently drops inline edge
labels** — always use `xlabel` and keep `forcelabels=true`. `pack=false` lets the
disconnected legend node nest into a free corner.

## Pin the columns with rank groups

Don't let dot infer the macro-layout. Declare one `{ rank=same; ... }` group per visual
column, left to right:

```dot
{ rank=same; "rf_in"; "rf_out"; "clk_lmx"; "clk_lmk"; }   // external I/O column
{ rank=same; "rfdc"; "clktree"; }
{ rank=same; "rx_cap"; "tx_pb"; }
{ rank=same; "dma_rx"; "ctrl"; "dma_tx"; }
```

With ranks pinned, declare every edge in its **true direction** (arrow = signal flow)
and add `constraint=false` to any edge that points against the left-to-right flow
(control fan-out from a right-side interconnect, a TX return path, a status line back
into an earlier column). The rank groups carry the layout; `constraint=false` stops
those edges from fighting it. This is cleaner than `dir=back` tricks — semantics stay
honest.

## Order nodes vertically inside a column

Within a rank, dot orders nodes by crossing minimization, which often puts an input in
the wrong corner. Force the order with invisible flat edges (in LR layouts, flat edges
point top → bottom):

```dot
"clk_lmk" -> "rf_out" [style=invis];
"rf_out"  -> "rf_in"  [style=invis];
"rf_in"   -> "clk_lmx" [style=invis];
```

Use this to keep a signal's source vertically adjacent to its sink — e.g. an external
clock that feeds the top-right node belongs at the top of the input column, or its edge
will cut across the whole figure (or worse, through a cluster's background, which reads
as a false connection).

Two lighter-touch tools: `weight=8` on an edge pulls its endpoints level with each other
(good for keeping `RF in → RFDC` horizontal); avoid `headport=n`-style compass entries
with ortho — the final approach segment tends to overlap the node's own label.

## Labels that survive ortho routing

- Redundant labels are the first thing to delete: if a port row or the legend already
  says it, the edge doesn't need an `xlabel`.
- When two parallel edges' labels collide, prefer (in order): delete one, shorten,
  move the information into the node label (`"SYSREF CDC → MTS"` beats an edge label
  floating in a dense bundle), or pad with newlines to shift the text off the line
  (`xlabel="text\n\n"` moves it up; leading `\n\n` moves it down).
- In plain quoted labels use literal UTF-8 (`·`, `→`, `×`); reserve HTML entities
  (`&#183;`, `&#8594;`, `&amp;`) for HTML-like `<...>` labels, where they are required.

## Port tables for hero blocks

Give a block explicit interface rows only when edges must attach at specific rows
(a PS with HP/HPM ports, a wide DDR controller). Pattern:

```dot
"ps" [shape=plaintext, label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="5" bgcolor="white">
  <tr><td bgcolor="#D79B00"><font color="white" point-size="14"><b>Zynq UltraScale+ PS</b></font></td></tr>
  <tr><td port="hp0" align="left" bgcolor="#FFE6CC"><font point-size="13">S_AXI_HP0</font></td></tr>
</table>>];

"dma_rx" -> "ps":"hp0":w [...];   // "node":"port" — quote separately, or you create a
                                  // phantom node literally named node:port
```

Order the rows to minimize crossings: match the vertical order of the blocks that
connect to them. Blank cells must be `<td> </td>` — an empty `<font></font>` is a parse
error.

## Collapsing without lying

- One node per BD hierarchy, labeled by function plus the key IP inside:
  `"RX capture\nAXIS combiner · capture gate\nCDC + buffer FIFO"`.
- Collapse identical clock/reset sinks into a domain node:
  `"DMA · SmartConnect · PS HP + DDR4 S_AXI\n(ui_clk domain)"`. One dashed clock edge
  into the domain node replaces a thicket.
- A clock-domain crossing in the datapath can be a single `dir=both` edge between two
  domain nodes, labeled with the CDC mechanism (`"AXIS CDC\nclk-conv (RX)\nasync FIFO (TX)"`).
- Merging interconnects is allowed only when it creates no false path. Three
  SmartConnects can become one node only if every drawn master→slave route exists in
  the netlist; otherwise keep them separate or route edges per real path.
- External ports: `shape=cds` for inputs, `shape=invhouse` for outputs, white fill.

## Legend

One self-contained `shape=plaintext` HTML-table node (swatch + plane name, colored line
sample + edge class), listing **only** what the figure uses. Never build a legend from
nodes joined by invisible edges. With `pack=false` it settles into a corner.

## Print-size math (do this before declaring done)

1. Natural size: read `<svg width="1286pt" height="525pt"` from the SVG (÷72 → inches).
2. Effective font at print: `font_pt × target_width / natural_width`.
   IEEE two-column `\textwidth` ≈ 7.16 in; single column ≈ 3.5 in.
3. Keep node text ≥ ~5 pt and edge labels ≥ ~4 pt effective. Fix by raising fonts
   (nodes 14, small/port text 13, edges 11) and tightening `nodesep≈0.35`,
   `ranksep≈0.5` — separations don't scale with font, so this wins ratio. If still
   short, cut content; never ship 3 pt text.
4. Any font/sep change reflows the layout — repeat the render-and-look loop after.

## QA loop

`dot -Tpng -Gdpi=110 fig.dot -o fig.png`, then **view the image**. Checklist:

- No edge passes through a node, a label, or an unrelated cluster's background band
  (a line crossing a cluster reads as membership — reroute via column ordering or
  raise the sink with `weight`).
- Every label readable, none struck through by a line.
- Arrowheads match signal direction everywhere (spot-check reversed-flow edges).
- External inputs enter from the canvas edge; nothing important is in a far corner.
- Legend complete and minimal; colors match the family conventions.

Fix one class of problem per cycle; expect 3–5 cycles. Final emit: `-Tsvg` and `-Tpdf`.
