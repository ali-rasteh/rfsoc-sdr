# Open-Source Wideband TX/RX Projects for the RFSoC 4x2: Survey and Comparison (October 2026)

No open-source project I found matches your full combination on the RFSoC 4x2: PYNQ control, 2 TX + 4 RX at about 1 GSPS I/Q each, deep PL-DDR4 cyclic playback and capture, and MTS. The closest options each cover part of it. Xilinx/RFSoC-MTS is the official MTS reference with BRAM players and a PL-DRAM "deep capture" path.\[1\] ali-rasteh/RFSoC_SDR is a PYNQ + MTS + DDR4 MIMO channel sounder for the 4x2.\[2\] uceeyuf/rfsoc4x2_rdma_ofdm is the only design I found that streams continuously in both directions with MTS (2 GSPS I/Q over 100G RDMA). Strathclyde's rfsoc_qsfp_offload is the established PYNQ/GNU Radio streaming path.\[3\]\[4\] The best way to reuse them is to take parts: the RDMA-to-PL-DDR4 engine, the streaming-DMA ideas from SPEAR, and QICK's external-clock and timing handling. None of them is a replacement for your design.

## TL;DR
- **Closest relatives on the 4x2:**
  - ali-rasteh/RFSoC_SDR (PYNQ, RFSoC-MTS, a DDR4 MIMO sounder bitstream, MIT).
  - Xilinx/RFSoC-MTS (MIT; 64-kilosample URAM/BRAM players and captures plus a PL-DRAM deep capture at 4 GSPS).
  - uceeyuf/rfsoc4x2_rdma_ofdm (continuous 64 Gb/s each way, 2.0 GSPS I/Q, MTS skew of 6–7 ps; no PYNQ, needs an ERNIC licence).
- **Streaming cannot replace your DDR4 buffer at your rates.** 4 RX × 983.04 MSPS × 32-bit I/Q ≈ 126 Gb/s, which is more than one QSFP28 can carry. The best demonstrated open designs sustain about 64–80 Gb/s. Streaming designs therefore cover 1–2 channels in real time, and your PL-DDR4 capture remains the right way to get 4-RX snapshots at full rate.
- **Worth reusing:**
  - uceeyuf/rfsoc4x2_ernic (RDMA WRITE/READ into PL DDR4 at about 98 Gb/s) to load and unload your DDR4 buffers quickly.
  - QICK's 10 MHz external-reference support and tProcessor timing, if you need timed triggering.
  - The SPEAR family's (ZCU216) FSM-driven streaming DMA, which streams up to 2^31 samples, as an architecture pattern.

## Key Findings

1. **Only one open RFSoC 4x2 design streams continuously in both directions at GSPS rates with MTS: uceeyuf/rfsoc4x2_rdma_ofdm.**
   - It moves "64 Gbit/s to the DACs and 64 Gbit/s from the ADCs at the same time, i.e. 2.0 GSPS × I/Q × 16 bit each way". A 10 s test showed 0 RX gaps, 0 TX underflows and 0 RX overflows.\[5\]
   - Post-sync skew between the two DAC/ADC pairs was "+0.012 … +0.014 samples (6 … 7 ps)".\[5\]
   - Its I/Q path uses DAC_A for I and DAC_B for Q (and ADC_B / ADC_D). That is one complex baseband stream carried on two real converters, not two independent complex TX channels.
   - It is a single-author project with 0 stars. It runs bare-metal over JTAG (no PYNQ), needs Vivado 2023.2 plus a licensed ERNIC IP, and its OFDM modem RTL is not published.\[5\]
2. **Strathclyde's QSFP offload is the mature PYNQ/GNU Radio streaming option, but it is RX-centric.**
   - The RFSoC 4x2 design packetises ADC data into UDP over the CMAC.\[3\]
   - The Strathclyde SDR Lab reports "a sustained throughput of approximately 80 Gbit/s over the QSFP link" and up to 2 GHz of instantaneous bandwidth (Šiaučiulis et al., IEEE NEWCAS 2023, DOI 10.1109/NEWCAS57931.2023.10198070).
   - The GRCon 2023 extension added a TX path, but it carries only 61.44 MSPS from the host, interpolated ×80 to 4915.2 MSPS on DAC B.\[6\]
   - ADC capture runs from 307.2 to 2457.5 MSPS, and an AXI-Stream switch selects one ADC at a time.\[6\]
3. **Xilinx/RFSoC-MTS is the upstream reference that the newer designs build on.**
   - It broadcasts a waveform from DAC RAM to two DAC tiles at 4 GS and captures three ADCs into "internal memory storing 64 kilosamples". A fourth ADC goes to a "deep capture" module that can use PL-DRAM.\[1\]
   - It supports the ZCU208 and RFSoC4x2, with PYNQ v3.0.1 and Vivado 2022.1, under the MIT licence.\[1\]
   - Both ali-rasteh/RFSoC_SDR and uceeyuf/rfsoc4x2_rdma_ofdm reuse it.\[2\] The latter credits it for its "URAM player / capture blocks".
4. **QICK has native 4x2 support and the best external clocking and timing, but its buffers are pulse-sized.**
   - The 4x2 is supported with a prebuilt qick_4x2.bit, and the QICK 0.2.436 docs say the external-clock option is supported for the "RFSoC 4x2 (put 10 MHz on CLK_IN)".
   - Documented memories are a pulse memory of 65536 samples per channel and a decimated ADC buffer of 1024 samples per I/Q component (older firmware documentation).\[7\]
   - The standard 4x2 firmware activates 2 DACs and 2 ADCs.\[8\]\[9\]
   - A DDR4 buffer driver (AxisBufferDdrV1) exists in the library.\[10\] Whether it is present in the standard 4x2 image is *unverified*.
5. **The strongest multi-channel streaming SDR (SPEAR/SPEAR+/SPEAR-BF, Duke) is ZCU216-only.**
   - It uses a custom FSM streaming DMA that streams PS-DRAM to DAC and ADC to PL-DRAM at 42.6 Gbps.\[11\]
   - SPEAR (ACM MobiCom '24) reports real-time bandwidth up to 1.25 GHz for 256QAM within the 3GPP 3.5% EVM requirement, and SPEAR+ (IEEE MILCOM 2025) reports "up to 1.2 GHz aggregated band-width across 2T2R, 4T4R, 8T8R, or 16T16R configurations", demonstrated in 1T1R, 2T2R and 4T4R, with MTS.
   - SPEAR-BF states continuous waveforms "up to 2^31 samples".\[12\]
   - Porting it to the 4x2 is a substantial job (see Details).
6. **Check your own design's DDR4 budget.**
   - The RealDigital RFSoC 4x2 Reference Manual (Rev. A3) lists 4 GB of 64-bit, 2400 MHz DDR4 on the PL, so its theoretical peak is about 19.2 GB/s.
   - Four RX at 983.04 MSPS × 4 B ≈ 15.7 GB/s on their own. Adding two TX streams read from the same DDR4 (≈ 7.9 GB/s) gives about 23.6 GB/s, which exceeds that peak.
   - If your design really runs full-rate cyclic TX from DDR4 while capturing all four RX into DDR4, confirm how it arbitrates. Playing TX from URAM, or time-multiplexing, would explain it. This is arithmetic from board specs, not a measured result.

## Comparison Table

Legend for "4x2 support": **Native** = builds/runs on RFSoC 4x2 as published; **Portable** = other board, port feasible; **No** = not realistically reusable for this purpose.

| Project (repo) | 4x2 support | Continuous TX (method, max length) | RX capture (buffer / streaming, max) | Bandwidth & channels | Sync (MTS, TX-RX, ext clk/PPS) | Practicalities (license, boards, tools, bitstreams, activity) |
|---|---|---|---|---|---|---|
| Xilinx/RFSoC-MTS — https://github.com/Xilinx/RFSoC-MTS | Native | Cyclic playback from DAC RAM (URAM/BRAM), broadcast to two DAC tiles; 64 k-sample class (forum users report "64 kilosamples worth of real samples")\[13\] | 3 ADCs to on-chip memory, 64 kilosamples; 4th ADC to "deep capture" in PL-DRAM (length *unverified*)\[1\] | 4 GS real sampling; multi-tile broadcast loopback | MTS demonstrated with/without comparison; ext PPS/10 MHz *unverified*\[1\] | MIT; ZCU208 + RFSoC4x2; PYNQ v3.0.1, Vivado 2022.1; prebuilt bit/hwh installed via install.sh; 4 commits, 13 open issues (low activity)\[1\] |
| ali-rasteh/RFSoC_SDR (rfsoc-sdr) — https://github.com/ali-rasteh/RFSoC_SDR | Native | Build named "sounder_fr3_if_ddr4_mimo_4x2" implies DDR4-backed MIMO playback; exact mechanism and length *unverified* | DDR4-based capture implied by build name; length *unverified*; host processing over Ethernet via sounder.py | Uses DAC_A/DAC_B and ADC_B/ADC_D as IF to Pi-Radio FR3 (7–24 GHz) front-end; filters DC–1300 MHz on ports; rates *unverified*\[2\] | Requires RFSoC-MTS install; notes a constant DAC-DAC/ADC-ADC phase offset that changes on every power cycle, calibrated by software\[2\] | MIT; RFSoC4x2; PYNQ v3.0.1 tested; prebuilt .bit/.hwh in vivado/.../builds; 307 commits; Zenodo DOI 10.5281/zenodo.14846067\[2\] |
| uceeyuf/rfsoc4x2_rdma_ofdm — https://github.com/uceeyuf/rfsoc4x2_rdma_ofdm | Native | True streaming from host over 100G RoCE v2 (ERNIC) into a 2 MB on-chip TX ring; unlimited length (host-fed) | True streaming: 64 KB chunks via RDMA WRITE WITH IMMEDIATE into a 64 MB host ring; 512 KB FPGA RX ring | 2.0 GSPS I/Q × 16 bit each way (64 Gb/s each direction); 4 GSPS in bursts; one complex stream on DAC_A(I)/DAC_B(Q), ADC_B/ADC_D (+A/C raw) | MTS, measured +0.012…+0.014 samples (6–7 ps); LMK04828/LMX2594 from A53, SYSREF 5 MHz; ext ref *unverified* | BSD-3 (modem closed); Vivado 2023.2 + ERNIC licence; bare-metal/JTAG, no PYNQ; no prebuilt bitstream noted; active (Oct 2026), 23 commits, 0 stars |
| uceeyuf/rfsoc4x2_ernic — https://github.com/uceeyuf/rfsoc4x2_ernic | Native (building block) | N/A (transport only): host RDMA WRITE into PL DDR4 | Host RDMA READ from PL DDR4 | 256 MB WRITE 97.98 Gbit/s, READ 98.02 Gbit/s\[14\]\[15\] | N/A | Vivado 2023.2, ERNIC v4.0 (licence/eval needed); PL-only; updated Sep 29, 2026 |
| uceeyuf/rfsoc4x2_mts — https://github.com/uceeyuf/rfsoc4x2_mts | Native | Buffer play over JTAG (tone/chirp) | Buffer capture over JTAG | 4 GSPS two-channel loopback\[15\] | Bare-metal MTS + chirp-training skew estimation: ~100 → −1.00 (MTS) → −0.002 samples\[15\] | Bare-metal, no PYNQ; updated Oct 1, 2026 |
| strath-sdr/rfsoc_qsfp_offload — https://github.com/strath-sdr/rfsoc_qsfp_offload | Native | GRCon'23 extension: host GNU Radio → UDP → DUC; 61.44 MSPS ×80 to 4915.2 MSPS on DAC B; unlimited length\[6\] | Streaming over 100GbE UDP (Network Layer IP + CMAC); one ADC selected via AXI-Stream switch\[3\]\[6\] | ADC 307.2–2457.5 MSPS; "up to 2 GHz of instantaneous bandwidth"; ~80 Gbit/s sustained\[6\]\[16\] | None documented | BSD-3; RFSoC4x2; PYNQ v2.7, Vivado/Vitis 2022.1; pip-installable overlay; updated Jul 7, 2026\[3\]\[17\] |
| Forks of qsfp_offload: LyceanEM/RFSoC-Sensing; spectrumx/rfsoc_qsfp_10g | Native | Sensing fork claims "multichannel reception and transmission" (*unverified* detail)\[18\] | 10G fork: lower rates, selectable ADC over XXV Ethernet to SFP+\[19\] | 10G fork limited to 10 GbE | *unverified* | 10G fork: Vivado/Vitis 2021.1\[19\] |
| Xilinx/RFSoC-PYNQ base overlay (RFSoC4x2) — https://github.com/Xilinx/RFSoC-PYNQ | Native | Transmitter/receiver hierarchies with Python drivers (rfsystem transmitter.py); TX buffer type/length *unverified*\[20\] | Receiver capture into on-chip buffers; length *unverified* | 4 ADC, 2 DAC, 100G CMAC in overlay\[21\] | MTS not in base overlay (*unverified*) | Now Vivado 2025.2 and PYNQ v4.0 (ZCU111, ZCU208, RFSoC4x2); ships on SD image\[22\]\[23\] |
| strath-sdr/rfsoc_sam — https://github.com/strath-sdr/rfsoc_sam | Native | RF data generator (BPSK/QPSK/8PSK/16QAM) to DAC for loopback\[24\] | Spectrum-analyser pipeline (FFT/PSD in PL), not raw capture | "Up to 256 MHz bandwidth for inspection" (original release); 3 selectable ADCs\[24\] | None | BSD-3; updated Jul 7, 2026\[17\] |
| strath-sdr/rfsoc_ofdm, rfsoc_radio, rfsoc_qpsk — https://github.com/strath-sdr | Native | Modem transmitters (narrowband) | Modem receivers + visualisation | Narrowband demos | None | BSD-3; rfsoc_radio: PYNQ ≥3.1.1, ZCU208/ZCU111/4x2/2x2, updated Oct 19, 2026; rfsoc_ofdm updated Oct 16, 2026\[15\]\[17\]\[25\]\[26\] |
| openquantumhardware/qick — https://github.com/openquantumhardware/qick | Native | Pulse generators from tables: 65536 samples per channel (I,Q); timed by tProcessor\[7\] | Readout DDC + ÷8 decimation; decimated buffer 1024 samples/component, accumulated 16384 (doc); DDR4 buffer driver exists\[7\]\[10\] | DAC 9.85 GSPS, ADC 5 GSPS on 4x2; standard 4x2 firmware: 2 DAC, 2 ADC\[8\]\[9\]\[27\] | tProc-timed TX/RX alignment; external 10 MHz on CLK_IN for 4x2; ADC step attenuator control\[28\] | MIT; ZCU111/ZCU216/RFSoC4x2; Vivado 2022.1 exactly; PYNQ 2.6–3.0.1 (not 3.1); prebuilt qick_4x2.bit; very active (docs v0.2.436)\[29\]\[30\]\[31\]\[32\] |
| CASPER mlib_devel + tutorials — https://casper-toolflow.readthedocs.io/projects/tutorials/en/latest/ | Native | DAC tutorial ("Using the RFDC's DAC"); buffer method *unverified*\[33\] | 100GbE yellow block streams packetised RFDC samples to a server; snapshot blocks for short captures\[34\] | 4x2: 8 samples per clock in 100GbE tutorial; CGEM (arXiv 2609.00357) samples at 4096 MS/s on a 4x2 using CASPER FIR, FFT and 100 Gb blocks | MTS/PPS *unverified* | Open source but needs MATLAB/Simulink (R2021a) + Vivado 2021.1 (m2021a branch) — proprietary dependencies\[33\]\[35\] |
| manbhattarai/awg-rfsoc (+ awg-rfsoc-trigger, AWG-MultiChannel-RFSoC) — https://github.com/manbhattarai/awg-rfsoc | Native (TX only) | Cyclic from on-chip RAM: 2^19 points at 9.34 GSps; trigger version 2^20 split into two switchable waveforms; multichannel version: 4 × 2^18 waveforms via AXI DMA\[36\]\[37\]\[38\] | None | 9.34–9.8304 GSps; 1–2 DACs\[38\] | Hardware trigger switching; MTS none | License not shown (*unverified*); Vivado TCL + RealDigital BSP; updated Oct 28, 2024\[15\] |
| nyu-wireless/mmwsdr — https://github.com/nyu-wireless | Native (*unverified* details) | *unverified* (repo not fetchable in this research) | *unverified* | *unverified* | *unverified* | MIT; Python; updated Apr 4, 2025; contains "Lessons for RFSoC" referenced by RFSoC_SDR\[2\]\[39\] |
| functions-lab SPEAR / SPEAR+ / SPEAR-BF — https://github.com/functions-lab/SPEAR_BF | Portable (ZCU216 only) | Custom FSM streaming DMA, PS-DRAM → DAC; continuous waveforms "up to 2^31 samples" (SPEAR-BF)\[11\]\[12\] | ADC → PL-DRAM via custom streaming DMA at 42.6 Gbps; DRAM-capacity-limited\[11\] | SPEAR: up to 1250 MHz IQ BW (2.5 GSPS, 2×), 1T1R; SPEAR+: 1.2 GHz aggregate, 2T2R–16T16R, demonstrated 1T1R/2T2R/4T4R\[40\]\[41\] | MTS (SPEAR+/BF) + software phase calibration; CLK104; ext ref *unverified*\[12\] | License not found (*unverified*); Vivado 2021.1; community ZCU216 PYNQ image; no prebuilt bitstreams; SPEAR_plus updated Mar 10, 2026\[12\]\[15\]\[40\] |
| LBNL QubiC 2.0 — https://lbl-qubic.gitlab.io/ | Portable in principle, low relevance | Pulse/sequence-based qubit drive | Per-shot demod readout | DSP at 500 MHz on ZCU216\[42\] | Distributed processor timing\[43\] | LBNL modified BSD; ZCU216 only; PYNQ backend; active GitLab\[44\]\[45\] |
| jorgeiglesiasc/advanced-rfsoc-front-end — https://github.com/jorgeiglesiasc/advanced-rfsoc-front-end | Native (RX only) | None | Multi-band GNSS chain (RFDC → frequency translator → FIR → DMA to PS)\[46\] | Direct RF sampling at 3–4 GSPS, one ADC\[46\] | None | GPL-3.0; RFSoC4x2 + ZCU208; Vitis HLS; updated Sep 3, 2024\[15\] |
| agrequejo/rfsoc4x2_loopback (PYNQ forum tutorial) | Native | 10 MHz tone DAC_B → ADC_B, ILA visualisation\[47\] | ILA only | Tutorial | None | Bare-metal Vitis; useful for LMK/LMX bring-up without PYNQ\[47\] |

## Ranked Shortlist for Your Use Case

**1. uceeyuf/rfsoc4x2_rdma_ofdm (+ rfsoc4x2_ernic, rfsoc4x2_mts)**
- *Beats yours:*
  - Truly continuous TX and RX with no waveform or capture-length limit, at 2.0 GSPS I/Q (about 2× your per-channel rate), with measured 6–7 ps MTS alignment.
  - Data arrives directly in host memory over 100G RDMA, with no PS involvement.
  - rfsoc4x2_ernic alone gives RDMA access to PL DDR4 at about 98 Gb/s.\[15\] That could replace whatever path you currently use to fill and drain your DDR4 buffers.
- *Falls short:*
  - Only one complex stream each way (I/Q split across DAC_A/DAC_B), so not 2 TX / 4 RX.
  - No PYNQ: JTAG/bare-metal bring-up and Vivado 2023.2.
  - Needs a licensed ERNIC IP and a tuned Linux host with isolated CPUs.\[5\]
  - Single author and closed modem.
  - The UltraRAM congestion it reports, when a DAC player and the RX core coexist, suggests resource pressure if you add your multi-channel DDR4 logic.

**2. ali-rasteh/RFSoC_SDR (NYU Wireless lineage)**
- *Beats yours:* a complete MIMO channel-sounder application stack (host-side processing, phase calibration, Pi-Radio FR3 front-end control), with prebuilt 4x2 bitstreams and an MIT licence.\[2\]
- *Falls short:*
  - Sample rates, buffer sizes and channel counts are not documented in the README (*unverified*).
  - Its own README warns of a power-cycle-dependent inter-port phase offset that must be recalibrated.\[2\]
  - It depends on RFSoC-MTS rather than adding new sync capability.\[2\]
  - It looks architecturally close to yours, so expect overlap rather than new capability.

**3. Xilinx/RFSoC-MTS**
- *Beats yours:* AMD-maintained, MIT-licensed reference for MTS on exactly this board, with prebuilt bitstreams and a PL-DRAM device-tree pattern for PYNQ.\[1\]
- *Falls short:* 64-kilosample on-chip players and captures, 4 GS real broadcast (not independent per-channel I/Q waveforms), PYNQ 3.0.1 / Vivado 2022.1, and little activity.\[1\]

**4. strath-sdr/rfsoc_qsfp_offload**
- *Beats yours:* continuous RX streaming of one ADC up to 2457.5 MSPS (about 80 Gb/s sustained) into GNU Radio on a PC, with peer-reviewed documentation (IEEE NEWCAS 2023, GRCon 2023) and a BSD-3 licence.\[4\]\[6\]\[16\]
- *Falls short:*
  - One ADC at a time.\[6\]
  - TX only at 61.44 MSPS from the host.\[6\]
  - No MTS.
  - Older PYNQ v2.7 image.\[3\]

**5. SPEAR / SPEAR+ / SPEAR-BF (ZCU216)**
- *Beats yours:* a published, multi-channel (up to 16T16R) streaming-DMA architecture with MTS. It streams PS-DRAM → DAC and ADC → PL-DRAM with deterministic hardware FSM control, and the "up to 2^31 samples" claim exceeds typical BRAM designs.\[11\]\[12\]\[41\]
- *Falls short:*
  - ZCU216 only, with no prebuilt bitstreams.
  - Licence not found.
  - Per-channel bandwidth at high channel counts is modest (1.2 GHz aggregate).\[41\]
- *Porting effort to the 4x2: substantial, roughly several engineer-weeks (my estimate).* You would need to:
  - remap RFDC tiles (ZU49DR 16×2.5 GSPS ADCs → ZU48DR 4 exposed ADCs);\[6\]\[48\]
  - replace CLK104 clocking with the 4x2's LMK04828/LMX2594 setup;
  - add the 4x2 PL-DDR4 MIG constraints;
  - rebuild in Vivado 2021.1 against a 4x2 PYNQ image.

**6. QICK**
- *Beats yours:* external 10 MHz reference support on the 4x2, cycle-accurate tProcessor timing of TX and RX events, a large and active community, and prebuilt bitstreams.\[28\]\[31\]
- *Falls short:* pulse-oriented buffers (65536-sample generator tables, a 1024-sample decimated buffer per the docs), only 2 ADCs active in standard 4x2 firmware, and strict Vivado 2022.1 / PYNQ ≤3.0.1 pinning.\[30\]\[32\]

**7. CASPER toolflow**
- *Beats yours:* mature 100GbE streaming and spectrometer libraries on the 4x2, already used in science instruments (CGEM, arXiv 2609.00357, samples at 4096 MS/s on a 4x2 using CASPER's FIR, FFT and 100 Gb blocks).
- *Falls short:* depends on MATLAB/Simulink R2021a and Vivado 2021.1, and is oriented to RX/DSP rather than long cyclic TX.\[33\]

## Details

**Why streaming does not replace your DDR4 buffer.**
- Your RX aggregate is 4 × 983.04 MSPS × 32 bits ≈ 125.8 Gb/s, which is more than a 100G QSFP28 link.
- The open designs top out around 64 Gb/s (RDMA streaming core) to about 80 Gb/s (Strathclyde UDP).\[16\]
- So real-time offload covers at most 2 of your 4 channels at your rate.
- Your TX aggregate (2 × 983.04 MSPS × 32 bits ≈ 62.9 Gb/s) would fit, so a hybrid design is plausible: host-streamed TX plus DDR4 snapshot RX.

**What can be lifted directly.**
- From rfsoc4x2_ernic: the ERNIC + CMAC + MIG block design, which writes and reads PL DDR4 at about 98 Gb/s.\[15\] It would replace PS-mediated DMA for loading multi-GB TX waveforms and pulling multi-GB RX captures.
- From RFSoC-MTS: the PYNQ device-tree overlay for PL-DRAM access and the URAM player/capture IP (MIT).\[1\]
- From MKIDGen3 (arXiv 2406.09764): its approach to the same PYNQ limitations. It reserved a device-tree segment for PL DDR4 so that pynq.allocate works, and patched the xrfdc driver to add MTS functions.\[49\]
- From SPEAR: the FSM-driven "streaming DMA" pattern, which avoids the stall and jitter of stock AXI DMA on long continuous transfers.\[11\]\[12\]\[50\]

**Academic work without fully verified code.**
- arXiv 2512.22582 describes real-time mmWave 5G NR radar on the RFSoC 4x2. It loads PDSCH waveforms into PL block RAM and uses MTS to align TX and RX.\[51\] A public code release was not confirmed (*unverified*).
- arXiv 2608.02860 describes an open-source RFSoC KID readout (KidPy, ZCU111) that uses CASPER mlib_devel and PYNQ.\[52\]

## Recommendations

1. **Keep your PL-DDR4 architecture for 4-RX snapshot capture.** It is the only approach among those surveyed that reaches 4 × ~1 GSPS I/Q simultaneously. Audit the DDR4 bandwidth budget described in Key Findings item 6.
2. **Adopt rfsoc4x2_ernic's RDMA path** if waveform load and capture readout time is your bottleneck. Budget for the ERNIC licence and the move off a pure-PYNQ flow (or wrap configuration in PYNQ yourself).
3. **Use rfsoc4x2_mts's chirp-training skew estimation**, which reports residual skew of −0.002 samples after MTS.\[15\] Combine it with RFSoC_SDR's power-cycle phase recalibration lesson to improve channel-to-channel coherence beyond plain MTS.\[2\]
4. **Borrow QICK's clocking approach** if you need a lab-wide 10 MHz reference. The QICK docs confirm the 4x2 takes 10 MHz on CLK_IN.
5. **Do not port SPEAR or QubiC** unless you are moving to a ZCU216. The porting cost is greater than the benefit over your existing design.

## Caveats
- Several cells are marked *unverified* because the repo pages or papers were not readable in this research. These include nyu-wireless/mmwsdr internals, RFSoC_SDR buffer sizes, the base-overlay buffer lengths, CASPER DAC buffering, and all SPEAR+ repo details beyond its IEEE abstract.
- The uceeyuf projects are recent, single-author and unstarred. Their figures are self-reported in READMEs and have not been independently reproduced.
- QICK buffer sizes come from firmware documentation that may describe older (tProc v1) images. Newer tProc v2 firmware may differ.
- Porting-effort estimates are my engineering judgement, not sourced figures.

## Notable Closed or Commercial Tools (excluded from table)
- MathWorks Communications Toolbox "Zynq SDR Support" (radio-in-the-loop for Zynq/RFSoC boards).\[6\]
- NI/Ettus USRP X410, which is RFSoC (ZU28DR) based. UHD/RFNoC is open source, but the hardware is commercial and is not an RFSoC 4x2 target.\[53\] Open RFNoC channel sounders exist for it (arXiv 2209.12773, 2403.11856).\[54\]
- Commercial 100GbE recorders used with the RFSoC 4x2; DAQscribe's application note says the 4x2's 100GbE "can only handle 12.5 GB/s, meaning we cannot stream all four channels at full rate without dropping data".

## Sources

1. [GitHub - Xilinx/RFSoC-MTS: A PYNQ overlay demonstrating AMD RFSoC Multi-Tile Synchronization (MTS).](https://github.com/Xilinx/RFSoC-MTS)
2. [GitHub - ali-rasteh/rfsoc-sdr](https://github.com/ali-rasteh/RFSoC_SDR)
3. [GitHub - strath-sdr/rfsoc\_qsfp\_offload: RFSoC QSFP Data Offload Design with GNU Radio · GitHub](https://github.com/strath-sdr/rfsoc_qsfp_offload)
4. [100GBit/s RF sample offload for RFSoC using GNU Radio and PYNQ - University of Strathclyde](https://pureportal.strath.ac.uk/en/publications/100gbits-rf-sample-offload-for-rfsoc-using-gnu-radio-and-pynq/)
5. [GitHub - uceeyuf/rfsoc4x2\_rdma\_ofdm: RFSoC 4x2: continuous 2 GSPS I/Q streaming over 100G RoCE v2 (AMD ERNIC) with multi-tile sync, 64 Gb/s to the DACs and from the ADCs at once; OFDM modem on the host CPU. Vivado 2023.2.](https://github.com/uceeyuf/rfsoc4x2_rdma_ofdm)
6. <https://strathprints.strath.ac.uk/87765/1/Siauciulis-etal-GNURC-2023-Ultra-wideband-SDR-architecture.pdf>
7. [Firmware — qick 0.2.376 documentation](https://docs.qick.dev/latest/firmware.html)
8. [Qibosoq: an open-source framework for quantum circuit RFSoC programming](https://arxiv.org/pdf/2310.05851)
9. [\\Qibosoq: an open-source framework for quantum circuit RFSoC programming](https://arxiv.org/html/2310.05851)
10. [qick.drivers.readout — qick 0.2.422 documentation](https://docs.qick.dev/latest/_autosummary/qick.drivers.readout.html)
11. <https://par.nsf.gov//servlets/purl/10554447>
12. [GitHub - functions-lab/SPEAR\_BF: SPEAR-BF: Wideband Multi-Channel D-Band Beamformer based on the RFSoC Platform · GitHub](https://github.com/functions-lab/SPEAR_BF)
13. [RFSoC Single DAC - Support - PYNQ](https://discuss.pynq.io/t/rfsoc-single-dac/6553)
14. [GitHub - uceeyuf/rfsoc4x2\_ernic: 100G RDMA (RoCE v2) on RFSoC 4x2 with the AMD ERNIC IP: host RDMA WRITE/READ to PL DDR4 at 98 Gbit/s, byte-exact, Linux rdma-core host · GitHub](https://github.com/uceeyuf/rfsoc4x2_ernic)
15. [rfsoc · GitHub Topics · GitHub](https://github.com/topics/rfsoc)
16. [QSFP Data Offload](https://sdr.eee.strath.ac.uk/qsfp-data-offload/)
17. [University of Strathclyde - Software Defined Radio Research Laboratory · GitHub](https://github.com/strath-sdr)
18. [GitHub - LyceanEM/RFSoC-Sensing: RFSoC QSFP Data Offload Design with GNU Radio, to enable multichannel reception and transmission.](https://github.com/LyceanEM/RFSoC-Sensing)
19. [GitHub - spectrumx/rfsoc\_qsfp\_10g: RFSoC QSFP Data Offload Design with GNU Radio · GitHub](https://github.com/spectrumx/rfsoc_qsfp_10g)
20. [Register Read/Write on RFSoC4x2 Hangs Entire Board - Support - PYNQ](https://discuss.pynq.io/t/register-read-write-on-rfsoc4x2-hangs-entire-board/7222)
21. [RFSoC-PYNQ overlays](http://www.rfsoc-pynq.io/overlays.html)
22. [RFSoC-PYNQ/docs/rfsoc\_4x2\_base\_overlay.md at master · Xilinx/RFSoC-PYNQ · GitHub](https://github.com/Xilinx/RFSoC-PYNQ/blob/master/docs/rfsoc_4x2_base_overlay.md)
23. [GitHub - Xilinx/RFSoC-PYNQ: Python productivity for RFSoC platforms · GitHub](https://github.com/Xilinx/RFSoC-PYNQ)
24. [An Open-Source Spectrum Analyser for RFSoC - Learn - PYNQ](https://discuss.pynq.io/t/an-open-source-spectrum-analyser-for-rfsoc/980)
25. [GitHub - strath-sdr/rfsoc\_radio: PYNQ example of using the RFSoC as a QPSK/BPSK radio transceiver. · GitHub](https://github.com/strath-sdr/rfsoc_radio)
26. [strath-sdr/rfsoc\_radio](https://deepwiki.com/strath-sdr/rfsoc_radio)
27. [RFSoC 4x2](https://www.realdigital.org/hardware/rfsoc-4x2)
28. [qick.qick — QICK 0.2.436 documentation](https://docs.qick.dev/latest/_autosummary/qick.qick.html)
29. [GitHub - openquantumhardware/qick: QICK: Quantum Instrumentation Control Kit · GitHub](https://github.com/openquantumhardware/qick)
30. [qick/firmware at main · openquantumhardware/qick · GitHub](https://github.com/openquantumhardware/qick/tree/main/firmware)
31. [Welcome to the QICK documentation! — QICK 0.2.436 documentation](https://docs.qick.dev/latest/)
32. [Quick-start guide — qick 0.2.422 documentation](https://docs.qick.dev/latest/quick_start.html)
33. [CASPER Tutorials — CASPER Tutorials 0.1 documentation](https://casper-tutorials.readthedocs.io/)
34. [Tutorial 4: 100GbE — CASPER Tutorials 0.1 documentation](https://casper-toolflow.readthedocs.io/projects/tutorials/en/latest/tutorials/rfsoc/tut_100g.html)
35. [100GBit/s RF sample ofﬂoad for RFSoC using GNU Radio and PYNQ](https://strathprints.strath.ac.uk/86118/1/Siauciulis_etal_NEWCAS2023_100GBit_s_RF_sample_offload_for_RFSoC.pdf)
36. [GitHub - manbhattarai/awg-rfsoc-trigger: Arbitrary waveform generator with capability to switch fast between two waveforms](https://github.com/manbhattarai/awg-rfsoc-trigger)
37. [GitHub - manbhattarai/awg-rfsoc: An arbitrary waveform generator with the Real Digital RFSoC board · GitHub](https://github.com/manbhattarai/awg-rfsoc)
38. [GitHub - CAQS-RQC/AWG-MultiChannel-RFSoC: High-sampling-rate, high-memory-density, multi-channel arbitrary waveform generator implemented on the RealDigital 4x2 RFSoC · GitHub](https://github.com/CAQS-RQC/AWG-MultiChannel-RFSoC)
39. [NYU Wireless · GitHub](https://github.com/nyu-wireless)
40. [GitHub - functions-lab/SPEAR: SPEAR: Software-defined Python-EnhAnced RFSoC for Wideband Radio Applications](https://github.com/functions-lab/SPEAR)
41. [SPEAR+: Streaming-Based Multi-Channel SDR Implementation Using the RFSoC Platform](https://ieeexplore.ieee.org/document/11310057/)
42. [QubiC 2.0: An Extensible Open-Source Qubit Control System Capable of Mid-Circuit Measurement and Feed-Forward](https://arxiv.org/html/2309.10333v1)
43. [Distributed Architecture for FPGA-based Superconducting Qubit Control](https://arxiv.org/html/2404.15260v1)
44. [QubiC](https://lbl-qubic.gitlab.io/)
45. [GitHub - al-nafisah/rfsoc-qubit-control-survey: Open-source RFSoC controllers for superconducting qubits, compared side by side from their papers and code: QubiC, QICK, FIREQ, RISC-Q, SQ-CARS, QiController · GitHub](https://github.com/al-nafisah/rfsoc-qubit-control-survey)
46. [GitHub - jorgeiglesiasc/advanced-rfsoc-front-end: Advanced multi-band GNSS SDR front-end implementation in an RFSoC integrated circuit. GPL-3.0 license · GitHub](https://github.com/jorgeiglesiasc/advanced-rfsoc-front-end)
47. [RFSoC 4x2 Bare-Metal Tutorial: 10 MHz DAC–ADC Loopback - Learn - PYNQ](https://discuss.pynq.io/t/rfsoc-4x2-bare-metal-tutorial-10-mhz-dac-adc-loopback/9063)
48. [RFSoC-based Readout and Characterization Platform Development at SLAC](https://indico.slac.stanford.edu/event/8288/contributions/7662/attachments/3645/9994/CPAD_2023_CLiu.pdf)
49. [MKIDGen3: Energy-Resolving, Single-Photon-Counting MKID Readout on an RFSoC](https://arxiv.org/pdf/2406.09764)
50. [Real-time Wideband Software-defined Radio with Python Programmability based on RFSoC](https://dl.acm.org/doi/10.1145/3636534.3698855)
51. [Real-Time Multi-Target Detection and Tracking with mmWave 5G NR Waveforms on RFSoC](https://arxiv.org/pdf/2512.22582)
52. [An Open Source RFSoC-based Readout Electronics System for Arrays of Kinetic Inductance Detectors and Superconducting Resonators](https://arxiv.org/pdf/2608.02860)
53. [A Wideband Distributed Massive MIMO Channel Sounder ...](https://arxiv.org/pdf/2403.11856)
54. [Ultrawideband USRP-Based Channel Sounding Utilizing the RFNoC Framework](https://arxiv.org/pdf/2209.12773)
