# Plan: Switchable DAC sample rate to keep images away from the carrier

- **Date:** 2026-10-28
- **Scope:** RFSoC 4x2, `vivado/sounder_fr3_if_ddr4_mimo_4x2` + `python/rfsoc.py`

## Problem

The new RF frontend will use DAC Nyquist zones 1 and 2 for different IF (mixer) frequencies. Its programmable filters will remove the images and other unwanted components.

A real DAC output at carrier f_c has images at k·Fs ± f_c. The nearest image sits 2·d from the carrier, where d is the distance from f_c to the nearest multiple of Fs/2. Close to a zone edge, this gap shrinks to zero, and no filter can separate the signal from its image.

Today the DAC sample rate is fixed at Fs = 3932.16 MHz:

- The LMX2594 clocks the tiles directly (tile PLLs off).
- The interpolation factor is ×4.
- The PL fabric clock is 245.76 MHz.

As a result, every carrier near 1966 MHz (Fs/2) is unusable.

## Goal

Choose the DAC sample rate per carrier at runtime, without a new FPGA image per frequency. For every IF in the planned range, the carrier-to-image gap must satisfy:

> gap ≥ B + G, where B is the signal bandwidth and G is the frontend filter's transition width.

## Approach

Switch between a few preplanned rates of the form **Fs = R × 983.04 MSPS**. On each switch, change the DAC's LMX2594 output and the interpolation factor R together. Do not tune Fs continuously.

Why this approach:

- **No FPGA changes.** The baseband rate stays at 983.04 MSPS (4 samples × 245.76 MHz), so the fabric clock, MMCM, timing, DMA and loop buffer are all unchanged. A bitstream rebuild should not be needed.
- **No ADC impact.** The RFSoC 4x2 has a separate LMX2594 for the DAC tiles.
- **Same clock quality.** Every rate is integer-N from the existing 122.88 MHz LMX reference. Today's file uses N = 64 with a 7864.32 MHz VCO and ÷2. Phase noise and spur behaviour should match today's.
- **SYSREF still works.** The 7.68 MHz SYSREF divides every rate (Fs / 7.68 = 384, 512, 640, 768).

### Rate plan

| f_c range (MHz) | DAC Fs (MHz) | R | Zone | LMX VCO (MHz) / N |
|---|---|---|---|---|
| ≤ 1475      | 3932.16 (current) | 4 | 1 | 7864.32 / 64  |
| 1475 – 1966 | 4915.2            | 5 | 1 | 9830.4 / 80   |
| 1966 – 2458 | 2949.12           | 3 | 2 | 11796.48 / 96 |
| 2458 – 3441 | 3932.16           | 4 | 2 | 7864.32 / 64  |
| 3441 – 4423 | 4915.2            | 5 | 2 | 9830.4 / 80   |

With these three rates, the carrier-to-image gap is at least about 983 MHz anywhere in 0.5–4.4 GHz. Today it drops to 0 at 1966 MHz. The plan therefore works as long as **B + G ≤ ~983 MHz**.

An optional fourth rate, 5898.24 MHz (R = 6), keeps more carriers in zone 1. It is within the DAC's 7.0 GSPS limit for the current datapath mode.

### Runtime switch sequence

1. Mute TX: set `dac_enable = 0` and hold `dac_reset`.
2. Shut down the DAC tile with `dac_tile.ShutDown()`.
3. Program the DAC LMX2594 (and only that chip) with the file for the new Fs. Wait for lock.
4. Call `dac_tile.DynamicPLLConfig(0, Fs, Fs)` (0 = external clock). This updates the driver's Fs for the NCO maths and restarts the tile.
5. For each DAC block:
   - set `InterpolationFactor = R`;
   - set `NyquistZone` to 1 or 2 (zone 2 enables mix-mode) and choose the matching inverse-sinc mode;
   - set the mixer frequency to f_c and apply it with `UpdateEvent`.
6. Run `SetupFIFO`, release `dac_reset`, and replay the waveform.
7. Recalibrate the phase between the two TX channels.

## Work items

1. **LMX register files.**
   - Generate files for 2949.12 and 4915.2 MHz (optionally 5898.24) in TI TICS Pro.
   - Start from `rfsoc/rfsoc4x2_clock_configs/LMX2594_3932.16.txt` and change only the output frequency; keep the 122.88 MHz reference and integer-N.
   - Do not hand-edit N, because several LMX fields depend on it.
   - Deploy the files as the README describes.
2. **DAC-only LMX write helper.**
   - `xrfclk.set_ref_clks` writes the same file to every LMX and also reprograms the LMK, so it can't be used here.
   - Add a helper that writes to the DAC LMX's SPI device only. This first requires identifying which SPI device that is.
3. **Frequency planner.**
   - Input: f_c, B and G. Output: (Fs, R, zone).
   - Also check where the folded 2nd and 3rd harmonics land (|k·Fs ± m·f_c|).
   - Prefer the current rate, and zone 1, whenever the margins allow.
4. **`retune_dac(f_c)` in `rfsoc.py`.**
   - Implement the switch sequence above.
   - The existing hook passes `DynamicPLLConfig(0, 122.88, 3932.16)` and is forced off for the 4x2 board. For an external clock, both frequency arguments should be Fs.
5. **Calibration per (Fs, zone).** Output level differs by rate and zone, so `optimal_gains.json` and the system-response calibration each need one entry per (Fs, zone).
6. **TX phase between channels.**
   - DAC multi-tile sync (MTS) is off in this build, so the phase between the two DACs changes after every switch.
   - Either recalibrate after each switch, or move to the DAC-MTS build (`v1-0-62`); with MTS the phase is repeatable.
7. **RX (optional).**
   - If the RX IF equals the TX IF, the ADC has the same problem at the same frequencies.
   - The same rate table works for the ADC (all rates ≤ 5 GSPS), using the ADC's own LMX and `DecimationFactor`.
   - Re-run ADC MTS after each switch.

## Verify first / risks

- **Runtime 3× and 5× interpolation.** The RFSoC Gen3 converters support these factors, but they are untested on this bitstream. This is the first thing to test.
  - Fallback: switch rates by changing the fabric clock instead (MMCM dynamic reconfiguration plus a separate DAC clock). That is a much larger FPGA change.
- **Switch time.** A switch involves SPI writes, LMX lock and a tile restart. Measure the time, and switch only between measurements, not within a capture.
- **Zone-2 output level.** The DAC output rolls off strongly in zone 2 (NRZ sinc). Use mix-mode and the zone-2 inverse-sinc setting.

## Rejected alternatives

- **Continuously tunable Fs.** The fabric clock would have to follow Fs / (4R). That means MMCM reconfiguration, timing closure at the highest rate, and SYSREF constraints, with no benefit for this problem.
- **Tile internal PLLs.** These avoid needing TICS Pro files, but have worse phase noise than clocking straight from the LMX. They are also still limited to Fs = R × 983.04 MSPS.

## Inputs needed

- IF (f_c) range, signal bandwidth B, and transition width G for each frontend filter.
- Whether RX should switch together with TX.

## Reference

- RFSoC 4x2 clocking (separate ADC and DAC LMX2594): <https://casper-tutorials.readthedocs.io/en/latest/tutorials/rfsoc/platforms/rfsoc4x2.html>
