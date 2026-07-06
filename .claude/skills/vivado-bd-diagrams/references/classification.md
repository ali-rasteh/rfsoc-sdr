# Classification reference — planes, subsystems, clutter

How to turn a parsed Block Design into functional planes. `bd_to_dot.py` encodes
these rules; this doc is the human-readable version and the place to reason about
ambiguous cases. Infer function from the **IP VLNV** and the **instance name** —
never from a connection the TCL doesn't contain.

## Block plane by IP VLNV (fill color)

| Plane (color) | VLNV short names | Why |
|---|---|---|
| **ps** (orange) | `zynq_ultra_ps_e`, `zynq_ps7`, `processing_system7`, `microblaze` | The processor that masters control traffic |
| **memory** (red) | `ddr4`, `ddr3`, `mig_7series`, `hbm`, `axi_bram_ctrl`, `blk_mem_gen` | Bulk memory endpoints |
| **rf** (purple) | `usp_rf_data_converter` / anything matching `rf_data_converter`, `rfdc` | ADC/DAC front-end |
| **clocking** (yellow) | `clk_wiz`, `proc_sys_reset`, `util_ds_buf`, user `sync` (CDC) | Clock gen, resets, CDC |
| **data** (blue) | `axi_dma`, `axi_datamover`, `axi_vdma`, `smartconnect`, all `axis_*` (fifo, broadcaster, combiner, switch, subset/dwidth converter, register_slice, clock_converter), user stream muxes | High-throughput dataflow |
| **control** (green) | `axi_interconnect`, `axi_intc`, `xlconcat`, `axi_gpio`, `axi_uartlite`, `axi_timer`, `axi_iic`, `axi_quad_spi` | Register access + IRQ plumbing |
| **other** (gray) | anything unmatched | Inspect and reclassify if it matters |

## Edge class by connection (color/style)

A **plane is the subgraph induced by a class of connection** — this is more honest
than bucketing blocks, because one block (e.g. a DMA) lives on several planes.

| Class | How it's detected (endpoint pin names) | Style |
|---|---|---|
| **stream** | contains `axis`, `m##_axis`/`s##_axis`, or RFDC analog `vin#`/`vout#` | bold blue |
| **mm** | `hp#`, `hpm`, `saxigp`/`maxigp`, `s2mm`/`mm2s`, `ddr`, `_mem`, or generic `axi` | red |
| **lite** | contains `lite`, or a bare `s_axi` register port (not `S_AXI_HP*`) | green |
| **clock** | `aclk`, `*_clk`, `clk_in`/`clk_ref`, `sysref`, diff-clock interfaces | dashed yellow |
| **reset** | `aresetn`, `resetn`, `rstn`, `sys_rst`, `reset` | dotted red |
| **irq** | `irq`, `introut`, `interrupt`, `intr` | dashed purple |

View filters: **data** = stream+mm · **control** = lite+irq · **clocking** = clock+reset.

## Clutter (suppressed by default, `--show-clutter` to include)

`xlconstant`, `xlslice`, `util_vector_logic`, `util_reduced_logic` (tie-offs / bit
slices / boolean glue), and debug cores `ila`, `vio`, `system_ila`, `axis_ila`,
`jtag_axi`. Keep a note of any slice that *gates a major path* (e.g. a GPIO bit that
enables a datapath) — mention it as an annotation rather than hiding it silently.

## Subsystem detection

Two honest sources of subsystems:
1. **Top-level hierarchies** — each `create_hier_cell_*` block is a designer-declared
   subsystem. `bd_to_dot.py --list-subsystems` prints them with cell counts + planes.
2. **Root IP with external I/O** — RF front-end (RFDC + `vin*`/`vout*`/`sysref`),
   memory (`ddr4` + external DDR4 port), networking (`cmac`/`xxv_ethernet` + GT),
   PCIe, debug. Name each and state the evidence (VLNV + which external ports it owns).

## Ambiguities to state, not guess

- **`smartconnect` vs `axi_interconnect`.** SmartConnect usually aggregates HP/DDR
  *data*; an AXI Interconnect with many MIs usually fans out AXI-*Lite* control. Default
  colors reflect that, but check the actual masters/slaves and say so.
- **PS `M_AXI_HPM0` vs `HPM1`.** Often one is a control master (→ interconnect) and one
  is a memory path (→ DDR). Classify by what it connects to, and flag it.
- **Bare `s_axi`.** Treated as control (register access). If it's actually a wide data
  slave, note the exception.
- **Repeated hierarchies.** `write_bd_tcl` emits `_1`, `_2` suffixed procs for repeated
  structures; each maps to one instance path. Preserve exact instance names so diagrams
  stay traceable to the source.
