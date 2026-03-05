import os
import pickle
import time
from dataclasses import dataclass

from pyftdi.ftdi import Ftdi  # type: ignore
from pyftdi.spi import SpiController  # type: ignore

from .general import GeneralConfig  # type: ignore
from .sivers_common import int2intlist, intlist2int
from .sivers_defaults import SiversDefaults

_THIS_DIR = os.path.dirname(__file__)


@dataclass(kw_only=True)
class SiversControllerConfig(GeneralConfig):
    pass


class SiversController(SiversDefaults):
    def __init__(self, config: SiversControllerConfig, **overrides):
        super().__init__(config, **overrides)

        allDevices = Ftdi.list_devices()
        Ftdi.show_devices()
        strFTDIdesc = str(allDevices[0][0])
        snStr = strFTDIdesc[strFTDIdesc.find("sn=") + 4 : strFTDIdesc.find("sn=") + 14]
        siverEVKAddr = "ftdi://ftdi:4232:" + snStr
        self.print(f"siverEVKAddr: {siverEVKAddr}", thr=1)

        ctrlA = SpiController(cs_count=1)
        ctrlA.configure(siverEVKAddr + "/1")
        self.spiSiver = ctrlA.get_port(cs=0, freq=60e6, mode=0)

        ctrlB = SpiController()  # spi
        ctrlB.configure(siverEVKAddr + "/2")
        self.gpioB = ctrlB.get_gpio()
        self.gpioB.set_direction(0xF0, 0xF0)  # [RST_N, PWR_EN, X, X, Xspi, Xspi, Xspi, Xspi]
        self.gpioB.write(0x00)
        self.reset()

    def reset(self):
        # Reset EVK
        self.gpioB.write(0x00)  # reset:On, power: Off
        self.gpioB.write(0x40)  # reset:On, power: On
        # gpio1.write(0x80) # reset:Off, power: off
        self.gpioB.write(0xC0)  # reset:Off, power: on

    def init(self):
        self.reset()

        # Bias the circuitries
        self.wr("bias_ctrl", 0x7F)  # Enable all the biases
        self.wr("bias_pll", 0x17)  # Enable all the biases for PPL with nominal current
        self.wr("bias_lo", 0x2A)  # Set nominal bias for X3  (TX,RX,X3)
        if self.chipType == "Eder B MMF":
            self.wr("bias_tx", 0x96AA)
            self.wr("bias_rx", 0x0AA9)
            if (self.fc <= self.bias_vco_x3_lo_freq) or (
                self.frequency >= self.bias_vco_x3_hi_freq
            ):
                self.wr("bias_vco_x3", 0x02)
            else:
                self.wr("bias_vco_x3", 0x01)
        else:
            self.wr("bias_tx", 0xAEAA)
            self.wr("bias_rx", 0xAAAA)
            self.wr("bias_vco_x3", 0x00)  # X3 out bias

        # Enable reference
        self.wr("fast_clk_ctrl", 0x20)  # Ref Clk = 45 MHz, Fast CLK freq = 180 MHz

        # Initialize ADC
        self.tgl("adc_ctrl", 0x20)  # reset ADC with toggling
        adc_sample_clk = 19e6
        cycle = 10
        set_edge = 0
        fast_clk_ctrl_reg = self.rd("fast_clk_ctrl") & 0x10
        if fast_clk_ctrl_reg == 0x10:
            freq = self.freq * 5
        elif fast_clk_ctrl_reg == 0x00:
            freq = self.freq * 4
        div = int(((38 * adc_sample_clk) / freq) - 1)
        self.wr("adc_clk_div", div)
        self.wr("adc_sample_cycle", cycle)
        if set_edge == 1:
            self.set("adc_ctrl", 2)
        else:
            self.clr("adc_ctrl", 2)

        # AGC:
        self.wr("gpio_agc_gain_in_ctrl", 0x80)
        self.wr("gpio_agc_start_ctrl", 0x00)
        self.wr("gpio_tx_rx_sw_ctrl", 0x00)

        # Enable PLL/VCO
        self.wr("pll_en", 0x7B)  # Enable PLL
        self.wr("pll_ref_in_lvds_en", 0x01)  # LVDS input
        self.wr("pll_chp", 0x01)  # Set charge pump current to 600 uA
        self.wr("vco_alc_del", 0x0E)  # 311 nanoseconds
        self.wr("vco_tune_loop_del", 0x000384)  # 20 microseconds
        self.wr("vco_atc_vtune_set_del", 0x001194)  # 100 microseconds
        self.wr("vco_atc_vtune_unset_del", 0x000384)  # 20 microseconds
        self.wr("vco_override_ctrl", 0x3F)  # Internal VCO tune state machine
        self.wr("vco_vtune_ctrl", 0x20)  # ATC Low threshold mux function enabled
        self.alc_th = int(self.alc_th_v / self.dac_ref * 255)  # VCO amplitude threshold
        self.wr("vco_alc_hi_th", self.alc_th)
        self.atc_hi_th = int(self.atc_hi_th_v / self.dac_ref * 255)
        self.wr("vco_atc_hi_th", self.atc_hi_th)
        self.atc_lo_th = int(self.atc_lo_th_v / self.dac_ref * 255)
        self.wr("vco_atc_lo_th", self.atc_lo_th)
        self.wr("pll_pfd", 0x00)  # Enable PFD Test
        self.wr("vco_en", 0x3C)
        self.set("vco_tune_ctrl", 0xFF)
        self.clr("vco_tune_ctrl", 0xFF)

        restart = True
        if restart:
            self.set("vco_tune_ctrl", (1 << 2))
        else:
            self.clr("vco_tune_ctrl", (1 << 2))
        time.sleep(0.5)
        self.wr("vco_en", 0x3C)  # Enable VCO

        # Set Carrier Frequency
        self.set_frequency(self.fc)

        # Initialize RX/TX
        self.wr("trx_rx_on", 0x1FFFFF)
        self.wr("rx_gain_ctrl_mode", 0x13)  # Direct register control
        self.wr("rx_gain_ctrl_sel", 0x3FF)
        self.wr("rx_bb_biastrim", 0x00)  # nominal current bias
        self.wr("rx_bb_test_ctrl", 0xD9)
        self.wr("rx_dco_en", 0x01)
        self.wr("rx_bb_i_dco", 0x101B)
        self.wr("rx_bb_q_dco", 0x1076)
        self.wr("rx_drv_dco", 0xFF0305FF)

        self.wr(
            "rx_gain_ctrl_bb1", 0x77
        )  # I[0:3]:[0,1,3,7,F]:-6:0 dB, 4 steps, Q[0:3]:[0,1,3,7,F]:-6:0 dB, 4 steps
        self.wr(
            "rx_gain_ctrl_bb2", 0x11
        )  # I[0:3]:[0,1,3,7,F]:-6:0 dB, 4 steps, Q[0:3]:[0,1,3,7,F]:-6:0 dB, 4 steps
        self.wr(
            "rx_gain_ctrl_bb3", 0x44
        )  # I[0:3]:[0-F]:0:6 dB, 16 steps, Q[0:3]:[0-F]:0:6 dB, 16 steps,
        self.wr(
            "rx_gain_ctrl_bfrf", 0x77
        )  # this is the gain before RF mixer, [0:3,RF gain]: 0-15 dB, 16 steps, [4:7, BF gain]: 0-15 dB, 16 steps
        self.wr("rx_gain_ctrl_reg_index", 0x0)

        self.wr("trx_tx_on", 0x1FFFFF)
        self.wr("tx_ctrl", 0x18)
        self.wr("tx_bb_i_dco", 0x2F)  # Nominal: 0x40
        self.wr("tx_bb_q_dco", 0x35)  # Nominal: 0x40
        self.wr("tx_alc_ctrl", 0x80)  # adjust RF gain before BF gain (ALC is off)
        self.wr(
            "tx_bb_gain", 0x00
        )  # tx_ctrl bit 3 (BB Ibias set) = 0: 0x00  = 0 dB, 0x01  = 6 dB, 0x02  = 6 dB, 0x03  = 9.5 dB
        # tx_ctrl bit 3 (BB Ibias set) = 1, 0x00  = 0 dB, 0x01  = 3.5 dB, 0x02  = 3.5 dB, 0x03  = 6 dB *
        self.wr("tx_bb_phase", 0x00)
        self.wr(
            "tx_bb_iq_gain", 0x44
        )  # this is the gain in BB, [0:3,I gain]: 0-6 dB, 16 steps, [4:7, Q gain]: 0-6 dB, 16 steps
        self.wr(
            "tx_bfrf_gain", 0x33
        )  # this is the gain after RF mixer, [0:3,RF gain]: 0-15 dB, 16 steps, [4:7, BF gain]: 0-15 dB, 16 steps

        # Set the AWVs
        # self.load_dumb('conf',group='bf_rx')
        # self.load_dumb('conf',group='bf_tx')

        self.set_beam_index_tx(32)
        self.set_beam_index_rx(32)

    def get_gain_rx(self):
        self.print("Getting RX gain", thr=1)
        rx_gain_ctrl_bb1 = self.rd("rx_gain_ctrl_bb1")
        rx_gain_ctrl_bb2 = self.rd("rx_gain_ctrl_bb2")
        rx_gain_ctrl_bb3 = self.rd("rx_gain_ctrl_bb3")
        rx_gain_ctrl_bfrf = self.rd("rx_gain_ctrl_bfrf")
        agc_int_bfrf_gain_lvl = self.rd("agc_int_bfrf_gain_lvl")
        agc_int_bb3_gain_lvl = self.rd("agc_int_bb3_gain_lvl")

        return (
            rx_gain_ctrl_bb1,
            rx_gain_ctrl_bb2,
            rx_gain_ctrl_bb3,
            rx_gain_ctrl_bfrf,
            agc_int_bfrf_gain_lvl,
            agc_int_bb3_gain_lvl,
        )

    def set_gain_rx(self, rx_gain_ctrl_bb1, rx_gain_ctrl_bb2, rx_gain_ctrl_bb3, rx_gain_ctrl_bfrf):
        self.print("Setting RX gain", thr=1)
        if (
            rx_gain_ctrl_bb1 > 255
            or rx_gain_ctrl_bb1 < 0
            or rx_gain_ctrl_bb2 > 255
            or rx_gain_ctrl_bb2 < 0
            or rx_gain_ctrl_bb3 > 255
            or rx_gain_ctrl_bb3 < 0
            or rx_gain_ctrl_bfrf > 255
            or rx_gain_ctrl_bfrf < 0
        ):
            success = False
            status = "Gain values should between 0 and 255."
            return success, status

        self.wr("rx_gain_ctrl_bb1", rx_gain_ctrl_bb1)
        self.wr("rx_gain_ctrl_bb2", rx_gain_ctrl_bb2)
        self.wr("rx_gain_ctrl_bb3", rx_gain_ctrl_bb3)
        self.wr("rx_gain_ctrl_bfrf", rx_gain_ctrl_bfrf)
        success = True
        status = "Success"
        return success, status

    def get_gain_tx(self):
        self.print("Getting TX gain", thr=1)
        tx_bb_gain = self.rd("tx_bb_gain")
        tx_bb_phase = self.rd("tx_bb_phase")
        tx_bb_iq_gain = self.rd("tx_bb_iq_gain")
        tx_bfrf_gain = self.rd("tx_bfrf_gain")
        tx_ctrl = self.rd("tx_ctrl")
        return tx_bb_gain, tx_bb_phase, tx_bb_iq_gain, tx_bfrf_gain, tx_ctrl

    def set_gain_tx(self, tx_bb_gain, tx_bb_phase, tx_bb_iq_gain, tx_bfrf_gain):
        self.print("Setting TX gain", thr=1)
        if (
            tx_bb_gain > 255
            or tx_bb_gain < 0
            or tx_bb_phase > 255
            or tx_bb_phase < 0
            or tx_bb_iq_gain > 255
            or tx_bb_iq_gain < 0
            or tx_bfrf_gain > 255
            or tx_bfrf_gain < 0
        ):
            success = False
            status = "Gain values should between 0 and 255."
            return success, status

        self.wr("tx_bb_gain", tx_bb_gain)
        self.wr("tx_bb_phase", tx_bb_phase)
        self.wr("tx_bb_iq_gain", tx_bb_iq_gain)
        self.wr("tx_bfrf_gain", tx_bfrf_gain)
        success = True
        status = "Success"
        return success, status

    def get_mode(self):
        self.print("Getting RX/TX mode", thr=1)
        self.mode = self.rd("trx_ctrl") & 0x03
        if self.mode == 0:
            mode = "RXen0_TXen0"
        elif self.mode == 1:
            mode = "RXen1_TXen0"
        elif self.mode == 2:
            mode = "RXen0_TXen1"
        elif self.mode == 3:
            mode = "RXen1_TXen1"
        return mode

    def set_mode(self, mode):
        self.print(f"Setting RX/TX mode to {mode}", thr=1)

        if mode == "RXen0_TXen0":
            self.wr("trx_ctrl", 0x00)
            success = True
            status = "Success"
        elif mode == "RXen1_TXen0":
            self.wr("trx_ctrl", 0x01)
            success = True
            status = "Success"
        elif mode == "RXen0_TXen1":
            self.wr("trx_ctrl", 0x02)
            success = True
            status = "Success"
        elif mode == "RXen1_TXen1":
            # self.wr('trx_ctrl',0x03) # dangerous case
            self.print(f"Error: set_mode {mode} Not performed.", thr=0)
            success = False
            status = "Not implemented"
        else:
            success = False
            status = (
                "Error: No such mode. Only available modes are RXen0_TXen0,RXen1_TXen0,RXen0_TXen1."
            )

        return success, status

    def freq_to_divn(self, freq):
        self.print("Converting frequency to divn", thr=2)
        return int(freq / 6 / self.freq - 36)

    def divn_to_freq(self, divn):
        self.print("Converting divn to frequency", thr=2)
        return (divn + 36) * 6 * self.freq

    def get_frequency(self):
        self.print("Getting frequency", thr=1)
        divn = self.rd("pll_divn")
        return self.divn_to_freq(divn)

    def set_frequency(self, frequency):
        if frequency > 70e9 or frequency < 57.51e9:
            self.print("Error: Carrier frequency should be between 57.51 GHz and 70 GHz", thr=0)
            success = False
            status = "ERROR: Invalid carrier frequency"
            return success, status

        self.print(f"Setting frequency to {frequency / 1e9} GHz", thr=1)
        # start_time = time.time()

        self.t = self.get_temperature("K") - 273
        # Set vco amplitude according to temperature
        self.alc_th = int(
            (self.alc_th_v + (25 - self.t) * 2.4e-3) / self.dac_ref * 255
        )  # VCO amplitude threshold
        self.wr("vco_alc_hi_th", self.alc_th)
        if self.chipType == "Eder B MMF":
            # Set vtune_th according to temperature
            self.vtune_th = int((self.t * 9e-3 + 1.166) * 255 / self.dac_ref)
            # pll_chp is set to 0x01 before lock
            self.clr("pll_chp", 0x03)
            self.set("pll_chp", 0x01)
        else:
            self.vtune_th = int((self.t * 67e-4 + 1.066) * 255 / self.dac_ref)
        self.print(f"Temperature: {self.t:.3f} C", thr=1)
        self.print(
            "vco_vtune_atc_lo_th: "
            + hex(self.vtune_th)
            + " ("
            + "%1.3f" % (self.vtune_th * self.dac_ref / 255)
            + " V)",
            thr=1,
        )
        self.print("vco_tune_ctrl: " + hex(self.rd("vco_tune_ctrl")), thr=1)
        self.wr("vco_vtune_atc_lo_th", self.vtune_th)
        self.wr("pll_divn", self.freq_to_divn(frequency))
        self.tgl("vco_tune_ctrl", 0x02)
        self.tgl("vco_tune_ctrl", 0x01)
        time.sleep(0.002)  # Increased to 2 ms from 0.5 ms
        vco_tune_status = self.rd("vco_tune_status")
        vco_tune_det_status = self.rd("vco_tune_det_status")
        vco_tune_freq_cnt = self.rd("vco_tune_freq_cnt")
        self.print("vco_tune_status [0x7e]: " + hex(vco_tune_status), thr=1)
        self.print("vco_tune_det_status[0] [1]: " + hex(vco_tune_det_status), thr=1)
        self.print("vco_tune_freq_cnt [0x7ff +/-11]: " + hex(vco_tune_freq_cnt), thr=1)
        if self.chipType == "Eder B MMF":
            # Set pll_chp to 0x00 if digtune between 28 and 64 or 92 and 128
            digtune = self.rd("vco_tune_dig_tune")
            if (digtune > 0x5C) or (0x1D < digtune < 0x40):
                self.clr("pll_chp", 0x03)
        # Check if tuning has succeeded
        if (
            (vco_tune_status != 0x7E)
            or (vco_tune_det_status & 0x01 != 0x01)
            or (vco_tune_freq_cnt > 0x80A)
            or (vco_tune_freq_cnt < 0x7F4)
        ):
            self.print("VCO tune FAILED", thr=0)
        else:
            self.print("VCO tune OK.", thr=1)
            self.set("vco_tune_ctrl", 0x04)

        success = True
        status = "Success"
        return success, status

    def set_beam_index_tx(self, index):
        if index > 63 or index < 0:
            self.print("Error: TX beam index should be between 0 and 63", thr=0)
            success = False
            status = "ERROR: TX beam index should be between 0 and 63"
            return success, status
        else:
            self.print(f"Setting TX beam index to {index}", thr=1)
            ptr = 0x80 | index
            success = True
            status = "Success"
            self.wr("bf_tx_awv_ptr", ptr)
            return success, status

    def set_beam_index_rx(self, index):
        if index > 63 or index < 0:
            self.print("Error: RX beam index should be between 0 and 63", thr=0)
            success = False
            status = "ERROR: RX beam index should be between 0 and 63"
            return success, status
        else:
            self.print(f"Setting RX beam index to {index}", thr=1)
            ptr = 0x80 | index
            success = True
            status = "Success"
            self.wr("bf_rx_awv_ptr", ptr)
            return success, status

    def get_beam_index_tx(self):
        self.print("Getting TX beam index", thr=1)
        return 0x7F & self.rd("bf_tx_awv_ptr")

    def get_beam_index_rx(self):
        self.print("Getting RX beam index", thr=1)
        return 0x7F & self.rd("bf_rx_awv_ptr")

    ## ADC
    def get_temperature_raw(self):
        self.print("Getting raw temperature", thr=5)
        self.start_adc(src1=0x83, src2=None, log2_nsamples=4)
        temp = self.mean_adc()
        self.stop_adc()
        return temp

    def get_temperature(self, unit="C"):
        self.print("Getting temperature", thr=2)
        return (
            self.get_temperature_raw() * self.temp_scale
            - self.temp_comp
            + self.unit_offs[unit]
            + self.temp_calib_offset
        )

    def start_adc(self, src1, src2=None, log2_nsamples=4):
        # self.lock.acquire()
        if (self.rd("adc_ctrl") & 0x80) == 0x80:
            self.print(
                "ADC has been started and is ready, but has not been stopped/reseted.", thr=1
            )
            return
        self.print("Starting ADC", thr=1)
        self.__src_1, self.__src_2 = self.get_amux()
        self.set_amux(src1, src2)
        self.wr("adc_num_samples", log2_nsamples)
        self.tgl("adc_ctrl", 0x10)
        while (self.rd("adc_ctrl") & 0x80) == 0:
            pass

    def stop_adc(self):
        self.print("Stopping ADC", thr=1)
        self.tgl("adc_ctrl", 0x20)
        self.set_amux(self.__src_1, self.__src_2)

    def mean_adc(self):
        return self.rd("adc_mean") & 0x0FFF

    def max_adc(self):
        return self.rd("adc_max") & 0x0FFF

    def min_adc(self):
        return self.rd("adc_min") & 0x0FFF

    def diff_adc(self):
        return self.rd("adc_diff") & 0x0FFF

    ## AMUX
    def set_amux(self, src=None, src_2=None):
        """Enables output of source "src" on AMUX-pin.
        src : source for AMUX output
        Example:
        amux.set(dbg.amux_vco)
        """
        self.print("Setting AMUX", thr=1)
        if src is not None:
            self.wr("bist_amux_ctrl", src)

        if src_2 is not None:
            if (self.rd("bist_amux_ctrl") & 0x7F) == self.amux_rx_bb:
                self.wr("rx_bb_test_ctrl", src_2)
            elif (self.rd("bist_amux_ctrl") & 0x7F) == self.amux_vco:
                self.wr("vco_amux_ctrl", src_2)
            elif (self.rd("bist_amux_ctrl") & 0x7F) == self.amux_otp:
                self.wr("bist_ot_ctrl", src_2)
            elif (self.rd("bist_amux_ctrl") & 0x7F) == self.amux_tx_pdet or (
                self.rd("bist_amux_ctrl") & 0x7F
            ) == self.amux_tx_env_pdet:
                self.wr("tx_bf_pdet_mux", src_2)

    def get_amux(self):
        self.print("Getting AMUX", thr=1)
        src = self.rd("bist_amux_ctrl")
        src_2 = None
        if (0x7F & src) == (0x7F & self.amux_rx_bb):
            src_2 = self.rd("rx_bb_test_ctrl")
        elif (0x7F & src) == (0x7F & self.amux_vco):
            src_2 = self.rd("vco_amux_ctrl")
        elif (0x7F & src) == (0x7F & self.amux_otp):
            src_2 = self.rd("bist_ot_ctrl")
        elif (0x7F & src) == (0x7F & self.amux_tx_pdet) or (0x7F & src) == (
            0x7F & self.amux_tx_env_pdet
        ):
            src_2 = self.rd("tx_bf_pdet_mux")
        return src, src_2

    def enable_amux(self):
        """Enable output on AMUX-pin."""
        self.print("Enabling AMUX", thr=1)
        self.set("bist_amux_ctrl", 0x80)

    def disable_amux(self):
        """Disable output on AMUX-pin."""
        self.print("Disabling AMUX", thr=1)
        self.clr("bist_amux_ctrl", 0x80)

    def clr_amux(self):
        """Disable output on AMUX-pin."""
        self.print("Clearing AMUX", thr=1)
        self.clr("bist_amux_ctrl", 0x80)

    ## SPI
    def get_address_and_size(self, reg_key):
        address = self.regs[reg_key]["addr"]
        size = self.regs[reg_key]["size"]
        return address, size

    def rd(self, reg_key):
        self.print(f"Reading register {reg_key}", thr=5)
        address, size = self.get_address_and_size(reg_key)
        command = int2intlist((address << 3) + self.SPI_RD, 256, 2)
        read_buf = self.spiSiver.exchange(command, 2 + size, start=True, stop=True, duplex=True)
        read_buf = read_buf[2:]
        # self.print(self.fhex(self.intlist2int(read_buf),size), thr=1)
        # self.print(''.join('{:02X}'.format(a) for a in list(read_buf)), thr=1)
        # self.print(hex(int.from_bytes(read_buf,'big', signed=False)), thr=1)
        return intlist2int(list(read_buf))

    def wr(self, reg_key, data):
        self.print(f"Writing register {reg_key} with data {hex(data)}", thr=5)
        address, size = self.get_address_and_size(reg_key)
        data = int2intlist(data, num_ints=size)
        command = int2intlist((address << 3) + self.SPI_WR_RAW, 256, 2)
        data.append(0)
        self.spiSiver.write(command + data, start=True, stop=True)

    def clr(self, reg_key, data):
        self.print(f"Clearing register {reg_key} with data {hex(data)}", thr=5)
        address, size = self.get_address_and_size(reg_key)
        command = int2intlist((address << 3) + self.SPI_WR_CLR, 256, 2)
        data = int2intlist(data, num_ints=size)
        data.append(0)
        self.spiSiver.write(command + data, start=True, stop=True)

    def set(self, reg_key, data):
        self.print(f"Setting register {reg_key} with data {hex(data)}", thr=5)
        address, size = self.get_address_and_size(reg_key)
        command = int2intlist((address << 3) + self.SPI_WR_SET, 256, 2)
        data = int2intlist(data, num_ints=size)
        data.append(0)
        self.spiSiver.write(command + data, start=True, stop=True)

    def tgl(self, reg_key, data):
        self.print(f"Toggling register {reg_key} with data {hex(data)}", thr=5)
        address, size = self.get_address_and_size(reg_key)
        command = int2intlist((address << 3) + self.SPI_WR_TGL, 256, 2)
        data = int2intlist(data, num_ints=size)
        data.append(0)
        self.spiSiver.write(command + data, start=True, stop=True)

    def dump(self, group="", is_detailed="False"):
        self.print("Dumping registers", thr=2)
        selectedDict = self.regs
        self.print("===========================", thr=2)
        if group != "":
            for reg_key in selectedDict:
                if selectedDict[reg_key]["group"] == group:
                    data = self.rd(reg_key)
                    self.print(reg_key + ":" + hex(data), thr=2)
                    if is_detailed:
                        self.print(selectedDict[reg_key]["doc"], thr=2)
                        self.print("\n", thr=2)
        else:
            for reg_key in selectedDict:
                data = self.rd(reg_key)
                self.print(reg_key + ":" + hex(int.from_bytes(data, "big", signed=False)), thr=2)
                if is_detailed:
                    self.print(selectedDict[reg_key]["doc"], thr=2)
                    self.print("\n", thr=2)
        self.print("\n", thr=2)

    def dump_save(self, filename, group=""):
        self.print("Dumping registers to file", thr=2)
        save_list = []
        if group != "":
            for reg_key in self.regs:
                if self.regs[reg_key]["group"] == group:
                    data = self.rd(reg_key)
                    save_list.append([reg_key, data])
            with open(filename + "_" + group, "wb") as f:
                pickle.dump(save_list, f)
        else:
            for reg_key in self.regs:
                data = self.rd(reg_key)
                save_list.append([reg_key, data])
            with open(filename + "_allRegs", "wb") as f:
                pickle.dump(save_list, f)

    def load_dumb(self, filename, group=""):
        self.print("Loading registers from file", thr=2)
        if group != "":
            with open(os.path.join(_THIS_DIR, filename + "_" + group), "rb") as f:
                load_list = pickle.load(f)  # +'.pkl'
        else:
            with open(os.path.join(_THIS_DIR, filename + "_allRegs"), "rb") as f:
                load_list = pickle.load(f)
        for reg_key in load_list:
            self.wr(reg_key[0], reg_key[1])
