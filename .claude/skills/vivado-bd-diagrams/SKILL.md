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

Run the parser on the target TCL, passed as an argument. It builds `model.json` and prints
an inventory (hierarchy tree, IP count by VLNV, ports, address assignments):

```bash
python <skill>/scripts/parse_bd.py path/to/create_bd.tcl
```

The `create_bd.tcl` path is a required positional argument (no longer hardcoded). All
generated files go into an `arch_diagrams/` directory created in the **same folder as the
TCL** — so `model.json` is written to `path/to/arch_diagrams/model.json` (override with
`--out`). The parser walks the `create_hier_cell_*` / `create_root_design` procs,
extracting cells + VLNVs + architecture params, interface nets, scalar nets, external
ports, and the address map. It only extracts — classification comes next.

### Step 2 — Classify and report (before drawing)

Read `references/classification.md` and map the model onto planes and subsystems:

- **Data plane** — AXI-Stream + AXI-MM dataflow (DMA, FIFOs, DDR/HBM, stream processing).
- **Control plane** — AXI-Lite register access, the processor system, interrupt plumbing.
- **Clocking / sync** — clock sources, MMCM/PLL, `proc_sys_reset`, CDC, clock/reset fan-out.
- **Subsystems** — top-level hierarchies and root IP with external I/O (RF front-end,
  DDR memory, networking, PCIe, debug). The hierarchy tree printed by `parse_bd.py`
  enumerates the top-level hierarchies to treat as subsystems.

Present a concise inventory to the user **before** rendering: the instance list with
inferred type, the hierarchy tree, proposed plane/subsystem assignments, anything
ambiguous (see the "Ambiguities to state" section of the classification doc — e.g.
smartconnect-vs-interconnect, which PS HP master is control vs memory), and which blocks
you're treating as clutter (tie-offs, slices, glue logic, debug cores — suppressed by
default but recorded).

### Step 3 — Diagram set

The generator emits a fixed set of seven plane- and subsystem-filtered diagrams. Confirm
with the user which of these they want to keep/render (one, several, or all):

1. `1_complete_architecture` — full filtered architecture, all planes, hierarchy clusters
2. `2_data_plane` — AXI-Stream / DMA / DDR dataflow (RX capture + TX playback)
3. `3_control_plane` — PS + AXI-Lite interconnect + interrupts (with register offsets)
4. `4_clocking_sync` — clock tree, MMCMs, SYSREF/MTS sync, reset fan-out
5. `5_subsystem_rf_frontend` — RF Data Converter + analog vin/vout + SYSREF
6. `6_subsystem_ddr4_memory` — DDR4 controller + root SmartConnect
7. `7_subsystem_gpio_control` — PS EMIO GPIO fan-out

The diagram bodies are hand-authored inside `scripts/bd_to_dot.py` (design-specific,
clutter suppressed by default). To retarget the set to a different design, edit the
diagram functions in that file.

### Step 4 — Generate, render, refine (hybrid)

Run the generator once, passing the same TCL path — it writes all seven `.dot` files into
`arch_diagrams/` beside the TCL (the same folder as `model.json`). Then render each,
**look at the PNG**, and refine for publication quality. The rendered SVG/PDF land in
`arch_diagrams/` too, so every generated file stays together.

```bash
python <skill>/scripts/bd_to_dot.py path/to/create_bd.tcl   # writes path/to/arch_diagrams/*.dot

# render every diagram (Graphviz):
for f in path/to/arch_diagrams/*.dot; do
  dot -Tsvg "$f" -o "${f%.dot}.svg"
  dot -Tpdf "$f" -o "${f%.dot}.pdf"
done

# for your own visual QA on a single view:
dot -Tpng -Gdpi=110 path/to/arch_diagrams/1_complete_architecture.dot -o path/to/arch_diagrams/complete.png
```

If `dot` isn't installed, read `references/graphviz_setup.md` (it covers the Windows
winget quirk and the portable-ZIP fallback) and the rendering gotchas.

The generator already hand-authors publication-quality DOT — correct connectivity, plane
coloring, hierarchy clusters, typed edges, HTML port tables for the hero blocks (PS, RFDC,
DDR4, DMAs), register offsets on control-plane targets, and a compact legend. After
rendering, verify against the "Refinement pass" in `references/visual_conventions.md`:
collapse the clock/reset sink thicket in busy clocking diagrams and fix any colliding edge
labels. Keep colors/shapes/legend consistent across the set so the diagrams read as one
family — all the exact values are in that reference.

When hand-authoring hero blocks with ports, heed the `"node":"port"` quoting gotcha and
the `splines=ortho`→`xlabel` rule in `references/graphviz_setup.md`; getting these wrong
produces phantom nodes and dropped labels that waste a rendering cycle.

## Bundled resources

- `scripts/parse_bd.py` — `write_bd_tcl` → `model.json` + printed inventory. Takes the
  target TCL path as a positional argument; writes into `arch_diagrams/` beside the TCL
  (`--out` overrides the model path).
- `scripts/bd_to_dot.py` — emits seven hand-authored, plane-/subsystem-filtered `.dot`
  files into `arch_diagrams/` beside the TCL (takes the TCL path to locate that folder).
  Clutter suppressed by default. Diagram bodies are design-specific — edit the `diagram_*`
  functions to retarget.
- `scripts/README.md` — the generated diagram set, visual conventions, and the key
  facts/assumptions extracted from the source TCL.
- `references/classification.md` — VLNV→plane table, edge-class rules, clutter policy,
  subsystem detection, and the ambiguities to surface. Read in Step 2.
- `references/visual_conventions.md` — exact colors/shapes/edge styles/legend/layout, and
  the refinement checklist. Read in Step 4 when polishing.
- `references/graphviz_setup.md` — installing/locating `dot`, render commands, and the
  hard-won DOT gotchas. Read when rendering or hand-editing DOT.
