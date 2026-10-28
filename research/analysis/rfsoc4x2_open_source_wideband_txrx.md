# Analysis: Which open-source RFSoC 4x2 projects send and capture continuous wideband signals, and what is worth reusing?
**Reports:** [Claude](../deep_research/rfsoc4x2_open_source_wideband_txrx/claude.md), [ChatGPT](../deep_research/rfsoc4x2_open_source_wideband_txrx/chatgpt.md), [Gemini](../deep_research/rfsoc4x2_open_source_wideband_txrx/gemini.md) · **Prompt:** [prompts/rfsoc4x2_open_source_wideband_txrx.md](../prompts/rfsoc4x2_open_source_wideband_txrx.md) · **Analyzed:** 2026-10-28

> ✅ means I checked the claim against a primary source on 2026-10-28: the project's own README, issue or slides, or your own design files. For performance figures, ✅ means the authors state the number. Nobody has reproduced it.

## First: the reports compared against the wrong baseline
All three reports describe your design as 2 TX / 4 RX at 983.04 MSPS each, with TX played back from PL DDR4. That description came from the prompt. Your repo shows something different:

| Item | What the reports assumed | What your design does | Evidence |
|---|---|---|---|
| Channels | 2 TX / 4 RX | 2 TX / 2 RX at 983.04 MSPS ✅ | [params.json:107](../../python/params/params.json#L107), [:115](../../python/params/params.json#L115), [:81](../../python/params/params.json#L81) |
| TX source | Cyclic playback from PL DDR4 | One DMA load from **PS** DDR, then replay from PL logic (`dac_mux_sel`). TX frames are 1024 samples ✅ | [rfsoc.py:239](../../python/rfsoc.py#L239), [rfsoc.py:556-569](../../python/rfsoc.py#L556-L569) |
| RX path | 4 × 983 MSPS into PL DDR4 (15.7 GB/s) | 4 ADC streams merge into one 256-bit stream at 245.76 MHz. That is 7.86 GB/s, or 2 complex channels, written to PL DDR4 ✅ | [create_bd.tcl:1141](../../vivado/sounder_fr3_if_ddr4_mimo_4x2/create_bd.tcl#L1141), [rfsoc.py:231](../../python/rfsoc.py#L231) |
| PL DDR4 | 64-bit, 2400 MT/s, 19.2 GB/s peak | Same ✅ | [create_bd.tcl:1319-1322](../../vivado/sounder_fr3_if_ddr4_mimo_4x2/create_bd.tcl#L1319-L1322) |
| MTS | On for DACs and ADCs | **On for ADCs, off for DACs** in the RFDC IP ✅ | [create_bd.tcl:1352](../../vivado/sounder_fr3_if_ddr4_mimo_4x2/create_bd.tcl#L1352), [:1410](../../vivado/sounder_fr3_if_ddr4_mimo_4x2/create_bd.tcl#L1410) |

What this changes:
- **The DDR4 overload warning doesn't apply.** Claude and Gemini warn that 23.6 GB/s exceeds the 19.2 GB/s peak, but TX never reads PL DDR4 and RX writes at most 7.86 GB/s.
- **Streaming now fits.** All three say streaming can't replace your DDR4 buffer because 125.8 Gb/s exceeds 100 GbE. That is only true for 4 RX. Your 2 RX need 62.9 Gb/s, which fits one 100 GbE link and is close to rates already demonstrated.
- **Long TX is a gap, not an advantage.** ChatGPT calls your "DDR4-backed cyclic player" unusual, but your TX is a short on-chip replay, like most of the projects surveyed.
- **DAC MTS may explain your phase problem.** [README.md:104](../../README.md#L104) notes a DAC-to-DAC phase offset that changes on every power cycle in mixer mode. DAC MTS being off, together with no synchronized NCO reset, is a likely cause.

## Bottom line
- **Borrow parts rather than switch.** No open project matches your design on the RFSoC 4x2. (High: all three agree.)
- **For sync, borrow DAC MTS plus NCO reset.** rfsoc4x2_uhd reports < 1 ps delay and about 1° phase difference, "the same after every boot" ✅. That is exactly your README's problem. (Medium: self-reported by a single author.)
- **For longer or continuous capture of your 2 RX, a 100 GbE path now fits.** rfsoc4x2_rdma_ofdm streams 64 Gb/s each way at once with MTS ✅. Strathclyde's offload measured 73.40 Gb/s on average ✅. (Medium: neither has been reproduced, and RDMA needs an ERNIC licence and no PYNQ.)
- **Keep PL DDR4 capture if you move to 4 RX at full rate**, because 125.8 Gb/s exceeds 100 GbE. Your RX DMA would also need double its current width. (High: arithmetic plus your block design.)
- **For longer TX waveforms, reimplement rather than copy.** The patterns exist (SPEAR's streaming DMA and a load-once replay buffer), but neither has a licence. (Medium.)
- **Don't build on QICK, CASPER, the Strathclyde modem demos, or a SPEAR port.** (High.)

## Report scorecard
| | Claude | ChatGPT | Gemini |
|---|---|---|---|
| Native 4x2 projects | ✅ | ✅ widest; only report with rfsoc4x2_uhd | ◐ missed rfsoc4x2_rdma_ofdm and rfsoc4x2_uhd |
| Portable projects and porting effort | ✅ | ✅ | ✅ |
| Named starting points checked | ◐ mmwsdr left unverified | ✅ | ✅ |
| Per-project criteria | ✅ | ✅ | ✅ |
| Comparison table | ✅ | ✅ | ✅ |
| Ranked shortlist (beats / falls short) | ✅ | ✅ | ✅ |
| Cites every claim, marks unverified | ✅ | ◐ citation markers broken in export (`fileciteturn…`), no source list; repo links survive | ◐ marks some items unverified, but presents unreleased features as released |
| Sources | 54 numbered; mostly repos, docs, papers | Untraceable except inline repo links | 108; repos and arXiv plus Reddit, retailer pages, Scribd |
| Recency | Oct 2026 | Oct 2026 | Oct 2026 |
| Strengths | Careful hedging; correctly guessed TX might play from on-chip RAM; RDMA-to-DDR4 idea | Careful on licences; right on the Strathclyde repo's public status; found rfsoc4x2_uhd | Unique correct QICK sync finding; board-level MTS detail; found OwnDPDLab, MKIDGen3, CASPER UPRM, SLAC, PUEO |
| Red flags | Ranks your own repo #2 as an "alternative"; "80 Gb/s sustained" vs 73.40 measured | Says your repo reuses the RFSoC-MTS player (wrong) | Missed both top streaming projects; calls your repo a ZCU111 mmwsdr port (wrong); 44 equations exported as images |
| **Score /100** | **80** | **84** | **62** |

- **Claude 80:** The claims I checked held up: rdma_ofdm, RFSoC-MTS, the SPEAR licence, and the DDR4 spec. It lost points for overstating Strathclyde's TX and throughput, and for not recognizing your repo.
- **ChatGPT 84:** Every claim I checked held up, including two that the other reports got wrong. It lost points for misdescribing your repo and for citations nobody can trace.
- **Gemini 62:** It was right about QICK and the Strathclyde slide numbers. But it missed the two top projects, presented unreleased features as released, and misidentified your repo.

## Where the reports agree
| Topic | Finding | Agents | Confidence |
|---|---|---|---|
| Overall | No open project matches; reuse parts | All | High |
| RFSoC-MTS | MIT; 4x2 and ZCU208; PYNQ 3.0.1, Vivado 2022.1; 3 ADCs to 64 k-sample on-chip memory, the 4th to PL-DRAM "deep capture" ✅ | All | High |
| RFSoC-MTS | DAC player depth isn't documented ✅ | All (Claude cites "64 k" from a forum) | Medium |
| Strathclyde qsfp_offload | RFDC → UDP → 100G CMAC; BSD-3; Vivado 2022.1 ✅ | All | High |
| Link math | 4 × 983.04 MSPS × 32 bit = 125.83 Gb/s, more than 100 GbE; 2 × = 62.91 Gb/s fits ✅ | All | High (but you have 2 RX) |
| rfsoc4x2_rdma_ofdm | 2.0 GSPS I/Q, 64 Gb/s each way at once, 0 gaps in 10 s, MTS skew 6–7 ps, BSD-3, ERNIC licence, Vivado 2023.2, no PYNQ ✅ | Claude, ChatGPT | Medium (self-reported, 0 stars) |
| rfsoc4x2_mts | Chirp training corrects residual skew after MTS (≈ −0.002 samples) | Claude, ChatGPT | Medium |
| SPEAR family | ZCU216 only; streaming DMA at 42.6 Gbps; 1.2 GHz aggregate; MTS; no licence ✅ | Claude, ChatGPT | High |
| QICK | Native 4x2, MIT; 65,536-sample pulse memories; not a raw wideband recorder | All | High |
| CASPER | Supports the 4x2; build needs MATLAB/Simulink | All | High |
| AWGs | TX only; on-chip waveforms of 2^18–2^19 samples | Claude, ChatGPT | Medium |
| Strathclyde sam/ofdm/radio, PYNQ base overlay | Narrowband or CW demos | All | High |
| Hybrid | Keep a local buffer and add a 100 GbE drain or live path | All | High |
| mmwsdr | No confirmed buildable 4x2 design | ChatGPT, Gemini (Claude: unverified) | Medium |

## Where they disagree
| Topic | Claude | ChatGPT | Gemini | Assessment |
|---|---|---|---|---|
| Strathclyde TX and MTS | TX added at GRCon'23: 61.44 MSPS ×80 on DAC B; no MTS | Public repo is RX-only; the 2026 bidirectional design is unverified | "GRCon24 4-ADC/TX release" with full MTS, 1–2 TX / 4 RX | **ChatGPT is right.** ✅ The README describes RX only (9 commits). ✅ The 2026 slides ("Universal High-Bandwidth SDR Transceiver") call 4 ADCs + network-fed TX + MTS an "updated design" and don't mention a release. |
| Strathclyde throughput | ~80 Gb/s sustained | ~80 Gb/s | 73.40 Gb/s average, 80 peak | **Gemini is right.** ✅ The slides report 73.40 Gb/s measured average at 2457.6 MHz; 80 Gb/s is the headline capability. |
| Your repo (ali-rasteh/RFSoC_SDR) | A separate #2 pick; DDR4 implied, unverified | Built on the RFSoC-MTS player | NYU mmwsdr lineage on ZCU111, needs porting | **None.** It is your design. ✅ It is native to the 4x2: TX uses AXI DMA plus PL replay, and RX goes to PL DDR4. Claude's hedged version is closest. |
| DDR4 bottleneck | 23.6 > 19.2 GB/s; "TX from URAM would explain it" | Not raised | Same overload; move TX to URAM | **Doesn't apply.** ✅ TX replays from PL; RX uses ≤ 7.86 GB/s. Claude's guess was right. |
| QICK sync on 4x2 | Not raised | Cross-tile MTS "not confirmed" | Standard 4x2 firmware has no DAC MTS (issue #277) | **Gemini is right.** ✅ The maintainer says MTS isn't implemented. The issue was closed on Mar 5, 2025 as cleanup, not as a fix. |
| #1 pick | rfsoc4x2_rdma_ofdm | rdma_ofdm, then rfsoc4x2_uhd | Strathclyde qsfp_offload | **rdma_ofdm** for raw streaming ✅. Gemini's pick rests on TX and MTS that aren't released. |
| SPEAR_plus on 4x2 | ZCU216 only | `rfsoc4x2_v43.ipynb` exists; no 4x2 build | — | **ChatGPT is right.** ✅ The notebook exists, but the build flow targets ZCU216 only. |
| RFSoC-PYNQ version | v4.0, Vivado 2025.2 | v4.0 (Sept 2026) | PYNQ 3.0.1, Vivado 2022.1 | Likely v4.0 (two reports to one; not checked). Low impact. |
| OwnDPDLab licence | — | — | "unverified" | ✅ Apache-2.0; bitstream marked "upload pending". |

## Found in only one report
| Finding | Agent | Why it matters | Credibility |
|---|---|---|---|
| rfsoc4x2_uhd: a UHD/RFNoC port. 2 RX + 2 TX at 500 MS/s per channel over DPDK with no drops. MTS + NCO reset gives < 1 ps and ≈ 1°, the same after every boot | ChatGPT | Targets your per-boot phase problem; standard UHD/GNU Radio API | ✅ README: GPL/LGPL, Vivado 2023.2, CMAC licence (free), no bitstreams. Single author |
| rfsoc4x2_ernic: host RDMA WRITE/READ to PL DDR4 at ~98 Gb/s | Claude | Fast fill and drain of your DDR4 | Not checked; same author as rdma_ofdm |
| QICK's standard 4x2 firmware has no DAC MTS | Gemini | Rules QICK out for phase-coherent TX | ✅ issue #277 |
| On the 4x2, tile 225/229 must pass clocks for MTS; the LMK04828 has a lock-LED/SPI quirk | Gemini | A likely reason DAC MTS is hard to enable | ✅ Strathclyde slides: "Tile 225 required to pass the CLK… DAC subsystem follows the same rules". LMK quirk is forum-sourced |
| CASPER UPRM records 4 ADCs at 1966.08 MSPS to NVMe for 2 h; BVEX provides 1 PPS + external 10 MHz LMK config | Gemini | Ready-made external-reference and PPS pieces | Medium (report/thesis; not checked) |
| OwnDPDLab (TUHH): 4x2 DPD testbed, ~983 MHz bandwidth, 1 TX / 1 RX | Gemini | Same bandwidth as yours; DPD, VOP and invsinc control code | ✅ repo exists, Apache-2.0; rates not in README |
| MKIDGen3 / gen3-vivado-top: pairs DACs and ADCs for 4 GHz complex bandwidth on the 4x2 | Gemini | Shows I/Q pairing across converters | ◐ ✅ 4x2 board files and Vivado 2022.1, but no licence found; bandwidth figure from paper |
| selfoluap/RFSoC4x2-AWG: MIT, 131,072 samples/DAC at 9.83 GSPS, DAC MTS, external 10 MHz | ChatGPT | A small, current (Vivado 2024.1) DAC-MTS reference | Not checked |
| lilyliu-nu/rfsoc-replay-buffer: load once from DDR into 1 MB BRAM, then replay indefinitely | ChatGPT | Your TX pattern with more memory | No licence; not checked |
| RadioStream: 4x2 prototype at ~90 Gb/s bidirectional; FPGA code not released | ChatGPT | A near-miss to watch | Medium |
| Inline coherent averaging (Lund/Ghent sounder) cuts the write rate by N | Gemini | Would let 4 RX fit in DDR or 100 GbE for periodic sounding | Medium (paper; HDL location unverified) |
| SLAC Simple-rfsoc-4x2-Example; PUEO pure-HDL MTS base | Gemini | Non-PYNQ, git-friendly MTS references | Not checked |

## Decision guide
| If your next need is… | Then… | Because… |
|---|---|---|
| Phase that repeats across power cycles | Enable DAC MTS (DAC0/DAC2) and NCO reset, following RFSoC-MTS and rfsoc4x2_uhd. Add rfsoc4x2_mts chirp calibration for residual skew | DAC MTS is off in your IP ✅; uhd reports ≈ 1° that repeats ✅ |
| Longer or continuous RX capture at today's 2 RX | Add a 100 GbE path. Simplest is Strathclyde's CMAC/UDP (BSD-3, PYNQ). Strongest is rdma_ofdm's ERNIC rings | 62.9 Gb/s needed; 73.40 and 64 Gb/s measured ✅ |
| 4 RX at the full 983 MSPS | Keep PL DDR4 capture and widen the RX DMA (15.73 GB/s needed vs 7.86 now). Drain over 100 GbE afterwards | 125.8 Gb/s exceeds 100 GbE ✅ |
| TX waveforms longer than on-chip memory | Replay from PL DDR4, which has headroom at today's RX rate. Copy the SPEAR or replay-buffer pattern, not their code | No licence on either ✅ (SPEAR_plus) |
| A standard host API | Evaluate rfsoc4x2_uhd | It tops out at 500 MS/s per channel, with no PYNQ and no bitstreams |
| External 10 MHz / PPS | Use the QICK docs (10 MHz on CLK_IN) and the CASPER/BVEX LMK files | Claude and Gemini; not checked |

## Verify before acting
1. **rdma_ofdm's 64 Gb/s with MTS.** It is the main reuse target but a 0-star, single-author project. Reproduce its 10 s tone test on your board, and get an ERNIC licence quote from AMD.
2. **rfsoc4x2_uhd's "≈ 1°, same after every boot".** This decides the sync fix. Replicate MTS + NCO reset on your design and log the phase over 5 or more power cycles.
3. **Strathclyde's 4-ADC + TX + MTS release.** If it ships, the PYNQ route gets much easier. Watch [the repo](https://github.com/strath-sdr/rfsoc_qsfp_offload) (9 commits, RX-only README) or ask the author (github.com/marsiau).
4. **DDR4 headroom for 4 RX or DDR-backed TX.** 15.7 of 19.2 GB/s peak is tight. Measure sustained mixed read/write throughput on your PL DDR4.
5. **Licences for SPEAR / SPEAR_plus / SPEAR_BF, rfsoc-replay-buffer, and MazinLab gen3-vivado-top.** None has a LICENSE file. Ask the authors before copying any code.
