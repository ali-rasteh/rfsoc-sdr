# Deep research: open-source wideband TX/RX projects for the RFSoC 4x2

## Goal
Find every open-source project that can be built for and run on the **Real Digital / AMD Xilinx RFSoC 4x2** board (XCZU48DR, Gen 3, 4 ADC / 2 DAC) to **transmit continuous wideband signals from the DACs and capture them on the ADCs**.

For context, I already have my own design: PYNQ-controlled, multi-channel MIMO, ADCs at 3.93216 GSPS with 4x decimation (~983 MSPS I/Q per channel), PL DDR4 used as the buffer for cyclic TX playback and RX capture, multi-tile sync (MTS). I want to know what else exists, how it compares, and whether any of it is reusable.

## Scope
- Projects targeting the RFSoC 4x2 directly.
- Projects for other RFSoC boards (ZCU111, ZCU208, ZCU216, RFSoC 2x2, etc.) that could realistically be ported. State how much porting work it would take.
- Any form: Vivado/HLS designs, PYNQ overlays, HDL IP, SDR frameworks and drivers (GNU Radio, UHD-like, SoapySDR), channel sounders, radar, spectrum analyzers, quantum/instrumentation control, and academic testbeds with released code.
- Starting points to check, but don't stop at them: RFSoC-PYNQ and the Strathclyde RFSoC projects, QICK, CASPER, Xilinx RFSoC-MTS, nyu-wireless/mmwsdr. Also search GitHub/GitLab, the AMD/Xilinx forums, the Real Digital and PYNQ sites, arXiv/IEEE papers, and theses.
- Exclude closed-source or commercial-only tools. If any are notable, list them briefly at the end.

## What to evaluate for each project
- **Continuous TX:** how it's done (cyclic playback from BRAM/URAM/DDR4, or streaming from the PS or host) and the max waveform length.
- **RX capture:** buffer type and max capture length, or whether it streams continuously (and over which interface: DMA to PS DDR, 100 GbE QSFP28, etc.).
- **Bandwidth:** achieved sample rates and instantaneous bandwidth per channel, and the number of TX/RX channels used at once.
- **Synchronization:** MTS/phase coherence across channels, TX-RX timing alignment, external clock/trigger/PPS support.
- **Practicalities:** license, supported boards, Vivado/PYNQ versions, whether prebuilt bitstreams exist, documentation quality, last commit date and activity.

## Output
1. A comparison table with one row per project and a column for each criterion above, plus repo link and RFSoC 4x2 support (native / portable / no).
2. A short ranked shortlist of the projects most relevant to my use case. For each one, say where it beats my design and where it falls short.
3. Cite a source (repo URL, doc, or paper) for every claim. Mark anything you couldn't confirm from code or docs as *unverified*.
