# Prior art: do open-source RFSoC projects already switch the DAC sample rate?

- **Date:** 2026-10-28
- **Plan:** [dac_sample_rate_switching.md](dac_sample_rate_switching.md)
- **Projects checked:** every major project named in [research/analysis/rfsoc4x2_open_source_wideband_txrx.md](../../../research/analysis/rfsoc4x2_open_source_wideband_txrx.md) and the three deep-research reports behind it, 34 repos in total.
- **Method:**
  - Each repo was shallow-cloned at the commit listed at the end of this file.
  - The code was searched for runtime control of the sample rate, LMX, tile PLL, interpolation/decimation, fabric clock and Nyquist zone, for example `DynamicPLLConfig`, `InterpolationFactor`, `NyquistZone`, `set_ref_clks`, `FabClkOutDiv` and MMCM dynamic reconfiguration.
  - Every hit was read in context.

## Question

Does any project already do what the plan needs: switch the DAC sample rate at runtime, change the interpolation factor with it so the fabric clock stays fixed, and pick the Nyquist zone per carrier?

## Answer

**No.** No project switches the sample rate and the interpolation factor together on a running design with a fixed fabric clock. Several projects do parts of it:

- **QICK** comes closest: it can change the sample rate when the bitstream loads, and its docs name zone-edge avoidance as a reason.
- **casperfpga** has most of the runtime building blocks, but cannot change interpolation.
- **Automatic zone selection on tune** is common, but always at a fixed sample rate.

A note on "Nyquist zone support" in the tables below: setting `NyquistZone` only changes the DAC output pulse shape. Zone 1 uses NRZ; zone 2 uses mix-mode, which makes zone-2 copies louder. Every image is still produced, and the gap between carrier and image stays the same. Zone switching on its own therefore does not solve the plan's problem.

## Partial matches

| Project | What it does at runtime | Gap vs. the plan |
|---|---|---|
| [QICK](https://github.com/openquantumhardware/qick) (MIT) | `QickSoc(dac_sample_rates={tile: fs})` replaces the compiled sample rate ([`configure_sample_rates`](https://github.com/openquantumhardware/qick/blob/bea61c1/qick_lib/qick/qick.py#L482)). [`valid_sample_rates()`](https://github.com/openquantumhardware/qick/blob/bea61c1/qick_lib/qick/qick.py#L368) lists the rates the tile PLL can reach, including the 7000 MSPS cap in DAC datapath mode 1 and the 6882–7863 MSPS DAC PLL hole. The [docs](https://github.com/openquantumhardware/qick/blob/bea61c1/docs/source/topics/changing_fs.rst) give "avoiding Nyquist zone boundaries" as a use. | Only when the bitstream loads, and marked experimental. Uses the tile PLL, not an external LMX. Can only lower the rate. The fabric clock slows with the sample rate, and tiles that share a logic clock must scale together. Interpolation is not changed. |
| [casperfpga](https://github.com/casper-astro/casperfpga) (GPL-2.0) | At runtime: load an LMX register file ([`progpll`](https://github.com/casper-astro/casperfpga/blob/f083e0b/src/rfdc.py#L399)), [`set_pll_config`](https://github.com/casper-astro/casperfpga/blob/f083e0b/src/rfdc.py#L1083), [`set_nyquist_zone`](https://github.com/casper-astro/casperfpga/blob/f083e0b/src/rfdc.py#L750), [`set_fab_clk_div_out`](https://github.com/casper-astro/casperfpga/blob/f083e0b/src/rfdc.py#L606), `set_invsinc_fir`, `run_mts` | No interpolation or decimation setter, and nothing that runs these steps as one sequence. Needs a CASPER-built bitstream with its control server (`tcpborphserver`). Not checked: whether `progpll` can program only the DAC LMX on the 4x2. |
| [SpectrumX mep-rfsoc-sdr](https://github.com/spectrumx/mep-rfsoc-sdr) (BSD-3) | [`set_dac_nco(f_c, f_s)`](https://github.com/spectrumx/mep-rfsoc-sdr/blob/5876405/src/mep_rfsoc_sdr/sdr_overlay.py#L97) picks the zone from f_c and Fs ([`_nyquist_zone`](https://github.com/spectrumx/mep-rfsoc-sdr/blob/5876405/src/mep_rfsoc_sdr/sdr_overlay.py#L172)) and calls `DynamicPLLConfig` on every tune | Fs is a constant, [1024 MSPS](https://github.com/spectrumx/mep-rfsoc-sdr/blob/5876405/src/mep_rfsoc_sdr/start_capture_rx.py#L41) |
| [RFSoC-PYNQ base overlay](https://github.com/Xilinx/RFSoC-PYNQ) | [`FrequencySelector`](https://github.com/Xilinx/RFSoC-PYNQ/blob/2a6beba/boards/RFSoC4x2/packages/rfsystem/package/rfsystem/spectrum_sweep.py#L159) switches the zone and flips the NCO sign when the tuned frequency crosses Fs/2 | Fixed at 4915.2 MSPS from the tile PLL |
| [rfsoc4x2_uhd](https://github.com/uceeyuf/rfsoc4x2_uhd) (GPL-3.0) | Tuning sets the NCO and the Nyquist zone | [`coerce_mcr`](https://github.com/uceeyuf/rfsoc4x2_uhd/blob/ef0bc76/mpm/python/usrp_mpm/periph_manager/rfsoc4x2_clocking.py#L288) rejects any master clock rate other than 500 MS/s. The converters are fixed at 4.0 GS/s and per-tile PLL reconfiguration is deliberately skipped. |
| [Strathclyde rfsoc_qsfp_offload](https://github.com/strath-sdr/rfsoc_qsfp_offload) (BSD-3) | [`set_decimation()`](https://github.com/strath-sdr/rfsoc_qsfp_offload/blob/8e856d5/boards/RFSoC4x2/rfsoc_qsfp_offload/drivers/overlay.py#L71) changes ADC decimation and `FabClkOutDiv` at runtime: stop the stream, `ShutDown()`, change the settings, `StartUp()` | RX only. Fs stays at 4915.2 MSPS and the output rate changes, which is the reverse of the plan's trade. |
| [OwnDPDLab](https://github.com/MarvinJaegerIng/OwnDPDLab) (Apache-2.0) | TCP commands set the DAC/ADC Nyquist zone and NCO (`RFSoC/pynq_driver.ipynb`) | Fs is fixed |
| [SPEAR](https://github.com/functions-lab/SPEAR), [SPEAR_plus](https://github.com/functions-lab/SPEAR_plus), [SPEAR_BF](https://github.com/functions-lab/SPEAR_BF) (no licence) | Python sets Fs (`DynamicPLLConfig`) and `InterpolationFactor` at startup ([`rfdc.py`](https://github.com/functions-lab/SPEAR_plus/blob/5b5e63e/rfsoc_rfdc/rfdc.py#L285)) | One rate per bitstream, for example `rfsoc_rfdc_v39.bit` vs `rfsoc_rfdc_v39_312M.bit`. ZCU216 only. |
| [mmWaveSDR](https://github.com/alphansahin/mmWaveSDR) (no licence) | Picks the LMK/LMX files and the bitstream together by rate; the bitstream name encodes `frfdc` | One rate (1536 MSPS) shipped. RFSoC 2x2. |

## Fixed rate, no runtime rate control

- **[RFSoC-MTS](https://github.com/Xilinx/RFSoC-MTS):** fixed, via a single `xrfclk` call.
- **[rfsoc4x2_rdma_ofdm](https://github.com/uceeyuf/rfsoc4x2_rdma_ofdm), [rfsoc4x2_mts](https://github.com/uceeyuf/rfsoc4x2_mts), [rfsoc4x2_ernic](https://github.com/uceeyuf/rfsoc4x2_ernic):** bare-metal designs with a fixed `LMX2594_500.0` file.
- **AWG designs:** [selfoluap RFSoC4x2-AWG](https://github.com/selfoluap/RFSoC4x2-AWG), [awg-rfsoc](https://github.com/manbhattarai/awg-rfsoc), [AWG-MultiChannel-RFSoC](https://github.com/manbhattarai/AWG-MultiChannel-RFSoC) and [RFSoC-MTS-AWG](https://github.com/tave0002/RFSoC-MTS-AWG).
- **[MKIDGen3](https://github.com/MazinLab/MKIDGen3) and [gen3-vivado-top](https://github.com/MazinLab/gen3-vivado-top):** 4.096 GS/s. **[BVEX_rfsoc_4x2](https://github.com/mayukh4/BVEX_rfsoc_4x2):** 3932.16 MSPS.
- **Strathclyde demos:** [rfsoc_sam](https://github.com/strath-sdr/rfsoc_sam), [rfsoc_radio](https://github.com/strath-sdr/rfsoc_radio) and [rfsoc_ofdm](https://github.com/strath-sdr/rfsoc_ofdm). rfsoc_ofdm's `configure_dacs(sample_freq=…)` takes a rate argument but is only called once, at startup.
- **Other designs:**
  - [Simple-rfsoc-4x2-Example](https://github.com/slaclab/Simple-rfsoc-4x2-Example)
  - [firmware-rfsoc4x2](https://github.com/pueo-pynq/firmware-rfsoc4x2)
  - [real_time_rfsoc_phase_correction](https://github.com/marcus-o/real_time_rfsoc_phase_correction)
  - [advanced-rfsoc-front-end](https://github.com/jorgeiglesiasc/advanced-rfsoc-front-end)
  - [rfsoc-replay-buffer](https://github.com/lilyliu-nu/rfsoc-replay-buffer)
  - [RFSoC4x2-App](https://github.com/RealDigitalOrg/RFSoC4x2-App)
- **No usable FPGA design:**
  - [mmwsdr](https://github.com/nyu-wireless/mmwsdr) has no FPGA source in the repo.
  - [radiostream-gnuradio](https://github.com/ucsdwcsng/radiostream-gnuradio) is host code only.
  - [mlib_devel](https://github.com/casper-astro/mlib_devel) sets the rate in the Simulink design at compile time; runtime control is in casperfpga, above.

## Not checked

- **AMD RF Data Converter Evaluation Tool (ZCU208/ZCU216).**
  - The Gemini report says its reference design changes Fs at runtime and reconfigures the MMCM to match.
  - It is not open source and does not target the 4x2.
  - A web search did not confirm the claim.

## What this means for the plan

- **The approach stands as written.** Nothing can be adopted as a whole.
- **The first risk is still open.** No project changes `InterpolationFactor` on a running design. The closest evidence is Strathclyde changing ADC `DecimationFactor` at runtime on the 4x2. Testing 3× and 5× interpolation remains the first task.
- **QICK shows what the rejected alternative costs.** In that alternative, the fabric clock follows Fs. QICK can change the rate only at load time, only downward, and marks the feature experimental because faster or changed logic clocks can destabilise the firmware.
- **The plan's 7.0 GSPS limit matches QICK's check.** QICK also caps DAC datapath mode 1 at 7000 MSPS. The 6882–7863 MSPS hole applies only to the tile PLL, so it does not affect the plan's LMX-direct clocking.

## Pieces worth borrowing

| For work item | Borrow | Licence |
|---|---|---|
| 3. Frequency planner | The rate rules in QICK's `valid_sample_rates()`: Fs limits, the datapath-mode cap, and the AXI clock limit (614 MHz on Gen3) | MIT |
| 3, 4. Zone and NCO sign | SpectrumX `_nyquist_zone()` (zone parity from f_c and Fs) or PYNQ `FrequencySelector` (NCO sign flip in even zones) | BSD-3 / PYNQ licence |
| 4. `retune_dac` sequence | Strathclyde `set_decimation()`: stop data, `ShutDown()`, change factors, `StartUp()`, restart data | BSD-3 |
| 2. DAC-only LMX writes | casperfpga `progpll` as a pattern for loading TICS files at runtime. Pattern only, because of GPL-2.0. | GPL-2.0 |

## Repos and commits checked

| Repo | Commit | Date |
|---|---|---|
| Xilinx/RFSoC-MTS | f8e93e7 | 2023-06-22 |
| Xilinx/RFSoC-PYNQ | 2a6beba | 2026-09-29 |
| uceeyuf/rfsoc4x2_rdma_ofdm | f2975ea | 2026-10-09 |
| uceeyuf/rfsoc4x2_uhd | ef0bc76 | 2026-10-15 |
| uceeyuf/rfsoc4x2_mts | 4d2a5aa | 2026-10-01 |
| uceeyuf/rfsoc4x2_ernic | a5a10b3 | 2026-09-30 |
| strath-sdr/rfsoc_qsfp_offload | 8e856d5 | 2026-03-16 |
| strath-sdr/rfsoc_sam | 1d090fe | 2026-10-19 |
| strath-sdr/rfsoc_radio | 721381c | 2026-10-19 |
| strath-sdr/rfsoc_ofdm | 2090ff4 | 2026-10-16 |
| openquantumhardware/qick | bea61c1 | 2026-10-09 |
| casper-astro/casperfpga | f083e0b | 2026-02-05 |
| casper-astro/mlib_devel | 0b8150e | 2026-05-14 |
| spectrumx/mep-rfsoc-sdr | 5876405 | 2026-09-02 |
| selfoluap/RFSoC4x2-AWG | 76065c2 | 2026-10-22 |
| manbhattarai/awg-rfsoc | 12f2dc5 | 2024-10-28 |
| manbhattarai/AWG-MultiChannel-RFSoC | 74aa311 | 2026-05-12 |
| lilyliu-nu/rfsoc-replay-buffer | d56f6f2 | 2026-06-12 |
| tave0002/RFSoC-MTS-AWG | 2416629 | 2025-10-09 |
| MazinLab/MKIDGen3 | ed70dda | 2025-10-30 |
| MazinLab/gen3-vivado-top | 3f7ef94 | 2026-06-12 |
| MarvinJaegerIng/OwnDPDLab | af86e86 | 2026-09-21 |
| mayukh4/BVEX_rfsoc_4x2 | 1c7d397 | 2025-09-06 |
| slaclab/Simple-rfsoc-4x2-Example | 849d31a | 2026-10-19 |
| pueo-pynq/firmware-rfsoc4x2 | ed76be7 | 2026-10-02 |
| marcus-o/real_time_rfsoc_phase_correction | 749cc33 | 2025-07-13 |
| jorgeiglesiasc/advanced-rfsoc-front-end | 7a6b47b | 2024-09-03 |
| functions-lab/SPEAR | 4b2428c | 2024-11-18 |
| functions-lab/SPEAR_plus | 5b5e63e | 2026-03-09 |
| functions-lab/SPEAR_BF | 0549416 | 2026-03-09 |
| alphansahin/mmWaveSDR | 709f4b9 | 2024-09-19 |
| nyu-wireless/mmwsdr | 37530db | 2025-04-04 |
| RealDigitalOrg/RFSoC4x2-App | 1e06adc | 2025-11-20 |
| ucsdwcsng/radiostream-gnuradio | adca1c4 | 2026-06-10 |
