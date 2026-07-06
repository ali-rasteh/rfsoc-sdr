---
name: vivado-bd-diagrams
description: >-
  Parse a Vivado IP Integrator Block Design TCL export (write_bd_tcl / create_bd.tcl)
  and produce filtered, plane-separated architecture diagrams — data plane, control
  plane, clocking/synchronization, and per-subsystem — as Graphviz DOT rendered to
  SVG/PDF. Use this whenever the user points at a create_bd.tcl or *_bd.tcl, a
  write_bd_tcl export, or a Vivado/Vitis IP Integrator design, OR asks to visualize,
  document, diagram, or explain the architecture of an FPGA / RFSoC / Zynq / Zynq
  MPSoC / UltraScale block design, its AXI / AXI-Stream / AXI-Lite dataflow, its clock
  and reset structure, or its DMA/DDR/RFDC subsystems — even if they never say
  "Graphviz", "DOT", or "diagram" explicitly. Prefer this skill over hand-reading the
  TCL: the bundled parser and generator are far more reliable than eyeballing 2000+
  lines of generated Tcl.
---

# Vivado Block Design → architecture diagrams

Turn a `write_bd_tcl` export into a small model, classify it into functional planes,
report the inventory, then generate publication-quality plane-filtered diagrams. The
bundled scripts do the deterministic heavy lifting (parsing, connectivity, first-pass
layout); you apply judgment on classification, ambiguities, and final polish.

**Golden rule:** infer function from the IP VLNV and instance names; never draw a
connection that isn't in the TCL. Where the TCL is ambiguous, state your assumption
rather than guessing silently. Preserve exact instance names so every block is
traceable back to the source.

## Workflow

### Step 1 — Parse

Run the parser on the target TCL. It builds `model.json` and prints an inventory
(hierarchy tree, IP count by VLNV, ports, address assignments):

```bash
python <skill>/scripts/parse_bd.py path/to/create_bd.tcl --out model.json
```

The parser is design-agnostic: it walks the `create_hier_cell_*` / `create_root_design`
procs, extracting cells + VLNVs + architecture params, interface nets, scalar nets,
external ports, and the address map. It only extracts — classification comes next.

### Step 2 — Classify and report (before drawing)

Read `references/classification.md` and map the model onto planes and subsystems:

- **Data plane** — AXI-Stream + AXI-MM dataflow (DMA, FIFOs, DDR/HBM, stream processing).
- **Control plane** — AXI-Lite register access, the processor system, interrupt plumbing.
- **Clocking / sync** — clock sources, MMCM/PLL, `proc_sys_reset`, CDC, clock/reset fan-out.
- **Subsystems** — top-level hierarchies and root IP with external I/O (RF front-end,
  DDR memory, networking, PCIe, debug). Run `bd_to_dot.py --list-subsystems` to enumerate.

Present a concise inventory to the user **before** rendering: the instance list with
inferred type, the hierarchy tree, proposed plane/subsystem assignments, anything
ambiguous (see the "Ambiguities to state" section of the classification doc — e.g.
smartconnect-vs-interconnect, which PS HP master is control vs memory), and which blocks
you're treating as clutter (tie-offs, slices, glue logic, debug cores — suppressed by
default but recorded).

### Step 3 — Offer diagram options

Ask which to generate (one, several, or all):

1. Complete architecture (full filtered)
2. Data plane
3. Control plane
4. Clocking / synchronization
5. One diagram per major subsystem you identified

### Step 4 — Generate, render, refine (hybrid)

For each chosen view, generate a first-pass DOT, render it, **look at the PNG**, then
refine for publication quality.

```bash
python <skill>/scripts/bd_to_dot.py model.json --view complete   --out complete.dot
python <skill>/scripts/bd_to_dot.py model.json --view data       --out data.dot
python <skill>/scripts/bd_to_dot.py model.json --view control    --out control.dot
python <skill>/scripts/bd_to_dot.py model.json --view clocking   --out clocking.dot
python <skill>/scripts/bd_to_dot.py model.json --subsystem adc_path --out adc_path.dot
# clutter is suppressed by default; add --show-clutter to include tie-offs/slices/glue

dot -Tsvg complete.dot -o complete.svg
dot -Tpdf complete.dot -o complete.pdf
dot -Tpng -Gdpi=110 complete.dot -o complete.png   # for your own visual QA
```

If `dot` isn't installed, read `references/graphviz_setup.md` (it covers the Windows
winget quirk and the portable-ZIP fallback) and the rendering gotchas.

The generator emits correct connectivity, plane coloring, hierarchy clusters, typed
edges, and a compact legend — a faithful first pass, not the final word. To reach
publication quality, follow the "Refinement pass" in `references/visual_conventions.md`:
give the 3–5 hero blocks (PS, RFDC, DDR4, DMAs) HTML port tables, collapse the clock/reset
sink thicket in busy clocking diagrams, add register offsets to control-plane targets,
and fix any colliding edge labels. Keep colors/shapes/legend consistent across the set
so the diagrams read as one family — all the exact values are in that reference.

When hand-authoring hero blocks with ports, heed the `"node":"port"` quoting gotcha and
the `splines=ortho`→`xlabel` rule in `references/graphviz_setup.md`; getting these wrong
produces phantom nodes and dropped labels that waste a rendering cycle.

## Bundled resources

- `scripts/parse_bd.py` — `write_bd_tcl` → `model.json` + inventory. Design-agnostic.
- `scripts/bd_to_dot.py` — `model.json` → plane-filtered DOT. Views: `complete`, `data`,
  `control`, `clocking`, `--subsystem <hier>`, `--list-subsystems`, `--show-clutter`.
- `references/classification.md` — VLNV→plane table, edge-class rules, clutter policy,
  subsystem detection, and the ambiguities to surface. Read in Step 2.
- `references/visual_conventions.md` — exact colors/shapes/edge styles/legend/layout, and
  the refinement checklist. Read in Step 4 when polishing.
- `references/graphviz_setup.md` — installing/locating `dot`, render commands, and the
  hard-won DOT gotchas. Read when rendering or hand-editing DOT.
