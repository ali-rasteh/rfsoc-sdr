# Visual conventions

Keep these consistent across every diagram in a set so the whole family reads as one
system. `bd_to_dot.py` already emits all of this; the values are here so hand-refined
DOT (adding port tables, etc.) stays in sync.

## Colors

| Plane | Fill | Border |
|---|---|---|
| Data | `#DAE8FC` | `#1F6FC4` |
| Control | `#D5E8D4` | `#2E8B57` |
| Clocking / sync | `#FFF2CC` | `#C99A00` |
| RF front-end | `#E1D5E7` | `#8E44AD` |
| Memory / DDR | `#F8CECC` | `#B85450` |
| Processor system | `#FFE6CC` | `#D79B00` |
| Other / clutter | `#EEEEEE` | `#888888` |

| Edge class | Color | Style | Width |
|---|---|---|---|
| AXI-Stream | `#1F6FC4` | solid | 2.0 |
| AXI-MM data | `#B85450` | solid | 1.7 |
| AXI-Lite ctrl | `#2E8B57` | solid | 1.4 |
| clock | `#C99A00` | dashed | 1.3 |
| reset | `#B85450` | dotted | 1.4 |
| interrupt | `#8E44AD` | dashed | 1.3 |

## Shapes

- Rounded box — default IP block.
- `box3d` — memory / FIFO (bulk storage).
- `octagon` — MMCM / PLL (`clk_wiz`).
- `cds` — external input port · `invhouse` — external output port.
- Plaintext **HTML table with named ports** — hero blocks (PS, RFDC, DDR4, DMA) when
  hand-refining, so AXI/AXIS interfaces attach to specific rows. See graphviz_setup.md
  for the `"node":"port"` gotcha.

## Layout (graph attributes)

```
rankdir=LR, splines=ortho, nodesep=0.32, ranksep=0.55,
forcelabels=true, pack=false, fontname="Helvetica"
node fontsize 11 · edge fontsize 9 · cluster label fontsize 12–13
```

- **`pack=false`** matters: it lets the disconnected legend node nest into free corner
  space instead of inflating the canvas with a big empty band.
- Hierarchy → `subgraph cluster_*`, nested to mirror the BD hierarchy.

## Legend

A single self-contained `shape=plaintext` HTML-table node — swatch paired with plane
name, colored line-sample (`———` solid, `– – –` dashed, `······` dotted) paired with
edge class. Show only the planes/classes actually present in that diagram. Never build
the legend from nodes joined by invisible edges — it sprawls and distorts the layout.

## Refinement pass (the "hybrid" step)

The generator gives a correct, plane-colored first pass. To reach publication quality:
- Replace the 3–5 hero blocks (PS, RFDC, DDR4, DMAs) with HTML port tables so major
  AXI/AXIS interfaces are labeled and attach at the right row.
- In busy clocking diagrams, collapse the many identical clock/reset sinks into one
  "fabric (aclk + aresetn fan-out)" node to cut the dashed-line thicket.
- Add register offsets from the address map onto control-plane AXI-Lite targets.
- Nudge or drop edge labels that collide; direction can be corrected from pin roles.
