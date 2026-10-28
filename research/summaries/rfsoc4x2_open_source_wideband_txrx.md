# Summary: What to borrow from open-source RFSoC 4x2 projects
**Built from:** [analysis/rfsoc4x2_open_source_wideband_txrx.md](../analysis/rfsoc4x2_open_source_wideband_txrx.md) · the 3 reports in [deep_research/rfsoc4x2_open_source_wideband_txrx/](../deep_research/rfsoc4x2_open_source_wideband_txrx/) (Claude, ChatGPT, Gemini) · [the prompt](../prompts/rfsoc4x2_open_source_wideband_txrx.md) · **Written:** 2026-10-28
> Scores are out of 10. They are my judgment, based on the analysis and its ✅-verified checks.

## 1. The situation
| | |
|---|---|
| Your design | 2 TX / 2 RX at 983.04 MSPS. TX is a short replay from PL after one DMA load; RX goes to PL DDR4. MTS is on for ADCs, off for DACs ✅ |
| Known pain point | The DAC-to-DAC phase offset changes on every power cycle (your README) |
| Question | Is there a better project to switch to, or parts worth borrowing? |
| Answer | Nothing to switch to. Borrow parts. |

**Report reliability:** ChatGPT 84, Claude 80, Gemini 62. Where they conflict, lean on ChatGPT. Use Gemini for its QICK finding and board-level MTS detail. All three assumed 2 TX / 4 RX with DDR4-backed TX, so discount their "your design wins" claims.

## 2. The facts that drive every decision
1. ✅ Your 2 RX need 62.9 Gb/s, which fits one 100 GbE link. 4 RX (125.8 Gb/s) would not.
2. ✅ rfsoc4x2_rdma_ofdm streams 64 Gb/s to the DACs and from the ADCs at the same time, with MTS (6–7 ps skew). It needs an ERNIC licence and Vivado 2023.2, and doesn't use PYNQ.
3. ✅ Strathclyde's 100 GbE offload measured 73.40 Gb/s on average. The public repo is RX-only, BSD-3 and PYNQ-based.
4. ✅ DAC MTS is off in your RFDC IP. rfsoc4x2_uhd reports that MTS + NCO reset gives ≈ 1° of phase difference that repeats on every boot.
5. ✅ Your PL DDR4 peaks at 19.2 GB/s and RX uses at most 7.86 GB/s. That leaves room for DDR-backed TX or more RX.

## 3. Top options
| Rank | Option | Score /10 | Speed/cost | Best when |
|---|---|---|---|---|
| 1 | Turn on DAC MTS + NCO reset | 8.5 | Small; free | Phase must repeat across boots (MIMO, AoA) |
| 2 | Add a 100 GbE RX stream (Strathclyde CMAC/UDP) | 7.5 | Medium; free (CMAC licence is free) | You want captures longer than DDR4 allows, and want to stay on PYNQ |
| 3 | Add RDMA streaming or drain (rdma_ofdm / rfsoc4x2_ernic) | 7 | Medium–large; ERNIC licence | You need TX and RX streaming at once with no PS in the path |
| 4 | Build your own DDR-backed long TX replay | 6.5 | Medium | Sounding waveforms outgrow on-chip memory |
| 5 | Move to rfsoc4x2_uhd | 4.5 | Large migration | A UHD/GNU Radio API matters more than 983 MSPS |

**1. DAC MTS + NCO reset**
- **Why:** It fixes a documented problem with a known recipe, and three projects show it working on this board.
- **Why not higher:** MTS on the 4x2 is fiddly (tile 225/229 clock passing, LMK quirks). uhd's 1° figure is self-reported.
- **How:**
  - Enable `DAC0`/`DAC2_Multi_Tile_Sync` in the RFDC IP and route SYSREF as RFSoC-MTS does.
  - Run MTS on DACs and ADCs, then reset the NCOs on SYSREF.
  - Add rfsoc4x2_mts's chirp calibration for residual skew.
  - Log the phase over several power cycles.

**2. Strathclyde 100 GbE RX stream**
- **Why:** 73.40 Gb/s measured covers your 62.9 Gb/s. It is BSD-3, runs on PYNQ, and targets the same board.
- **Why not higher:**
  - The public code has no TX and no MTS.
  - 73.40 Gb/s leaves little margin over 62.9.
  - You need a 100 GbE NIC and jumbo frames.
- **How:** Lift its CMAC + network layer + packetizer and feed it your combined 256-bit RX stream. Keep DDR4 capture as a fallback.

**3. RDMA (rdma_ofdm / rfsoc4x2_ernic)**
- **Why:** It is the only open design proven at 64 Gb/s in both directions with MTS.
- **Why not higher:**
  - It needs an ERNIC licence.
  - It runs bare-metal over JTAG, not PYNQ.
  - It is a 0-star project by a single author.
- **How:** Reproduce its 10 s test first, then remap its I/Q channel layout to your 2 complex channels.

**4. DDR-backed long TX**
- **Why:** DDR4 has headroom, and this removes the TX length limit.
- **Why not higher:** There is no licensed open code to copy, and gap-free looping needs a custom or scatter-gather DMA.
- **How:** Loop reads from PL DDR4 with a DataMover state machine, using SPEAR's pattern but written from scratch.

**5. rfsoc4x2_uhd**
- **Why:** It gives you a standard API and repeatable phase.
- **Why not higher:** It tops out at 500 MS/s per channel, about half your rate. It runs on PetaLinux rather than PYNQ, and ships no bitstreams.

> **Rejected:**
> - **QICK (2/10):** pulse-sized memories, and no DAC MTS in the standard 4x2 firmware.
> - **CASPER (3/10):** needs MATLAB/Simulink and is RX-oriented. Take only its external 10 MHz/PPS files.
> - **Porting SPEAR (2/10):** ZCU216 only, with no licence.
> - **Strathclyde demos and the PYNQ base overlay (1/10):** narrowband or CW only.
> - **mmwsdr (1/10):** no buildable 4x2 design.

## 4. What to do, and when
| When | Action |
|---|---|
| This week | Check whether DAC MTS was turned off on purpose. If not, try option 1 and log the phase over 5 power cycles |
| This week | If you rerun the research, fix the prompt's baseline (2 TX / 2 RX; TX replays from PL) |
| Next | Decide what limits you, then branch below |
| Before option 3 | Reproduce rdma_ofdm's 10 s test and get an ERNIC licence quote |
| Ongoing | Watch strath-sdr/rfsoc_qsfp_offload for the 4-ADC + TX + MTS release |

| If the limit is… | Do |
|---|---|
| Capture length | Option 2, or option 3 if you also need TX streaming |
| TX waveform length | Option 4 |
| Channel count (4 RX) | Widen the RX DMA to 15.73 GB/s, keep DDR4 capture, and drain over 100 GbE |

## 5. Verify before acting (top 5)
1. **rdma_ofdm's 64 Gb/s with MTS.** It is self-reported by a 0-star project. Reproduce the README's 10 s test.
2. **uhd's "≈ 1° after every boot".** This decides the sync fix. Test MTS + NCO reset over several power cycles.
3. **Strathclyde's TX/MTS release.** It would make option 2 bidirectional. Watch the repo or ask github.com/marsiau.
4. **DDR4 headroom.** 15.7 of 19.2 GB/s peak is tight for 4 RX or DDR-backed TX. Measure mixed read/write throughput.
5. **Licences for SPEAR, rfsoc-replay-buffer and gen3-vivado-top.** None has a LICENSE file. Ask the authors before copying code.
