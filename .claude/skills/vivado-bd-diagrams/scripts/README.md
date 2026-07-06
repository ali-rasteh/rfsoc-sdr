# Architecture Diagrams — `system_bd` (sounder_bbf_sivers_ddr4_2x2)

Filtered, plane-separated block diagrams generated from
[`../create_bd.tcl`](../create_bd.tcl) (Vivado 2022.1 `write_bd_tcl` export,
target **xczu28dr** / RFSoC 2x2). All instance names are exact and traceable
back to the TCL. Architectural clutter (constant tie-offs, single-bit slices,
`util_vector_logic` glue, per-signal reset nets) is suppressed by default.

## Diagrams
| # | File | Scope |
|---|------|-------|
| 1 | `1_complete_architecture` | Full filtered architecture, all planes, hierarchy clusters |
| 2 | `2_data_plane` | AXI-Stream / DMA / DDR dataflow (RX capture + TX playback) |
| 3 | `3_control_plane` | PS + AXI-Lite interconnect + interrupts (with register offsets) |
| 4 | `4_clocking_sync` | clocktreeMTS, reset_block, MMCMs, SYSREF/MTS sync, reset fan-out |
| 5 | `5_subsystem_rf_frontend` | RF Data Converter + analog vin/vout + SYSREF |
| 6 | `6_subsystem_ddr4_memory` | DDR4 controller + root SmartConnect (3 masters) |
| 7 | `7_subsystem_gpio_control` | PS EMIO GPIO fan-out (lmk_rst, gpio_test, datapath enables) |

Each diagram is provided as `.dot` (source), `.svg`, and `.pdf`.
`dot_verified/` holds a pristine snapshot of the visually-reviewed `.dot` files;
`gen_dot.py` reproduces them byte-for-byte (verified by diff), so the generator is
the single source of truth and the two cannot drift.

## Visual conventions (consistent across all diagrams)
- **Fill color = plane/subsystem:** data (blue), control (green), clocking (yellow),
  RF front-end (purple), DDR memory (red), processor system (orange).
- **Edge color/style = signal class:** AXI-Stream (bold blue), AXI-MM data (red),
  AXI-Lite control (green), clock (dashed yellow), reset (dotted red), interrupt (dashed purple).
- **Shapes:** `box3d` = memory/FIFO, `octagon` = MMCM/PLL, `cds`/`invhouse` = external in/out ports,
  HTML tables (with named ports) = hero blocks (PS, RFDC, DDR4, DMA).
- A compact, self-contained legend (single HTML node) is embedded in every diagram.
- Layout: `rankdir=LR`, `splines=ortho`, `ranksep=0.55`, `nodesep=0.32`, `pack=false`
  (lets the legend nest in free corner space instead of inflating the canvas),
  `forcelabels=true` with edge `xlabel`s (ortho routing drops inline edge labels).

## Regenerating
All outputs go into an `arch_diagrams/` folder created beside the target TCL.
```bash
python parse_bd.py  path/to/create_bd.tcl   # parse -> arch_diagrams/model.json + inventory
python bd_to_dot.py path/to/create_bd.tcl   # emit the 7 .dot files under arch_diagrams/
# render (Graphviz):
for f in path/to/arch_diagrams/*.dot; do dot -Tsvg "$f" -o "${f%.dot}.svg"; dot -Tpdf "$f" -o "${f%.dot}.pdf"; done
```

## Key facts extracted from the TCL
- **65 IP instances** across 9 hierarchies; 21 interface ports; 3 scalar ports; 16 address assignments.
- **RX:** RFDC `m00`/`m20` → `axis_combiner_0` → `axis_flow_ctrl_0` → CDC/FIFO(32768) → `axi_dma_0` (S2MM) → DDR4 / PS-HP0.
- **TX:** PS-HP1 / DDR4 → `axi_dma_0` (MM2S) → async FIFO → `dac_strm_mux` (live/replay) → FIFO(2048) → broadcaster → RFDC `s00`/`s10`.
- **Control:** PS `M_AXI_HPM1` → `ps8_axi_periph` (5 MI) → DAC-DMA (0xB0002000), ADC-DMA (0xB000A000),
  RFDC (0xB0040000), flow-ctrl (0xB0020000), clk_wiz (0xB0010000).
- **Clocking:** LMK/DDR refs → `clocktreeMTS` clk_wiz (rf_clk 245.76 MHz, ddr_clk 200 MHz) +
  SYSREF sync for multi-tile sync; `reset_block` derives 100/200/245 MHz resets.

### Stated assumptions / ambiguities
1. `M_AXI_HPM0_FPD → root smartconnect_0 → DDR4` is drawn on the **data/memory** plane
   (PS-initiated memory access), while `M_AXI_HPM1_FPD` is the control master.
2. LMK clock ports appear **name-crossed** in the TCL: net `lmk_clk1_1` drives `IBUFDS_SYSREF`
   from port **`lmk_clk2`**, and `lmk_clk2_1` drives `IBUFDS_PL_CLK` from port **`lmk_clk1`**.
   Exact TCL endpoints are preserved; the crossover is flagged, not "corrected".
3. Two distinct `clk_wiz_0` MMCMs exist (`clocktreeMTS/clk_wiz_0`, `reset_block/clk_wiz_0`);
   full hierarchical paths are kept.
4. `binary_latch_counter_0` inferred as lock/power-up sequencing from naming + `dcm_locked` connectivity.
