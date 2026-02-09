import time
from dataclasses import dataclass
import numpy as np
from numpy.fft import fft, ifft

from pynq import Overlay, allocate, GPIO
# from pynq.lib import dma
import xrfclk
import xrfdc

from sigcom_toolkit.general import GeneralConfig, General
from tcp_comm import Tcp_Comm_RFSoC
try:
    from Sivers.siversController import *
except Exception as e:
    print("Error importing siversController class: ", e)



@dataclass
class RFSoCConfig(GeneralConfig):
    beam_test: np.ndarray = np.array([1, 5, 9, 13, 17, 21, 25, 29, 32, 35, 39, 43, 47, 51, 55, 59, 63])
    project: str = 'sounder_if_ddr4'
    board: str = 'rfsoc_4x2'
    RFFE: str = 'none'
    rfsoc_lmk_freq_mhz: float = 122.88
    rfsoc_lmx_freq_mhz: float = 3932.16
    dac_fs: float = 245.76e6 * 4
    adc_fs: float = 245.76e6 * 4
    bit_file_path: str = 'project_v1-0-58_20241001-150336.bit'          # Path to the bit file for the RFSoC (Without DAC MTS)
    # bit_file_path: str = 'project_v1-0-62_20241019-173825.bit'          # Path to the bit file for the RFSoC (With DAC MTS)
    mix_freq_dac: float = 1000e6
    mix_freq_adc: float = 1000e6
    rfsoc_mix_phase_off: float = 0.0
    DynamicPLLConfig: list = None
    do_rfsoc_mixer_settings: bool = False
    do_rfsoc_pll_settings: bool = False
    run_tcp_server: bool = True
    n_frame_wr: int = 1
    n_frame_rd: int = 2
    n_tx_ant: int = 2
    n_rx_ant: int = 2

    n_skip: int = 0
    CLOCKWIZARD_LOCK_ADDRESS: int = 0x0004
    CLOCKWIZARD_RESET_ADDRESS: int = 0x0000
    CLOCKWIZARD_RESET_TOKEN: int = 0x000A

    def __post_init__(self):
        super().__post_init__()

        if self.DynamicPLLConfig is None:
            self.DynamicPLLConfig = (0, self.rfsoc_lmk_freq_mhz, self.rfsoc_lmx_freq_mhz)

        if 'sounder_bbf' in self.project:
            self.do_rfsoc_mixer_settings=False
            self.do_rfsoc_pll_settings=False
        if self.board == "rfsoc_4x2":
            self.do_rfsoc_pll_settings=False

        if self.board=='rfsoc_2x2':
            self.adc_bits = 12
            self.dac_bits = 14
            self.adc_max_fs = 4096e6
            self.dac_max_fs = 6554e6
        elif self.board=='rfsoc_4x2':
            self.adc_bits = 14
            self.dac_bits = 14
            self.adc_max_fs = 5000e6
            self.dac_max_fs = 9850e6

        if 'ddr4' in self.project:
            if self.board=='rfsoc_2x2':
                if 'sounder_bbf' in self.project:
                    self.dac_tile_block_dic = {0: [0], 1:[0]}
                    self.adc_tile_block_dic = {0: [0], 2:[0]}
                    self.dac_tiles_sync = [0,1]
                    self.adc_tiles_sync = [0,2]
                elif 'sounder_if' in self.project:
                    if self.n_tx_ant==1:
                        self.dac_tile_block_dic = {1: [0]}
                        self.dac_tiles_sync = [0]
                    elif self.n_tx_ant==2:
                        self.dac_tile_block_dic = {0: [0], 1:[0]}
                        self.dac_tiles_sync = [0,1]
                    if self.n_rx_ant==1:
                        self.adc_tile_block_dic = {2: [0]}
                        self.adc_tiles_sync = [0,2]
                    elif self.n_rx_ant==2:
                        self.adc_tile_block_dic = {0: [0], 2:[0]}
                        self.adc_tiles_sync = [0,2]
            elif self.board=='rfsoc_4x2':
                if 'sounder_bbf' in self.project:
                    self.dac_tile_block_dic = {0: [0], 2:[0]}
                    self.adc_tile_block_dic = {0: [0], 2:[0]}
                    self.dac_tiles_sync = [0,2]
                    self.adc_tiles_sync = [0,2]
                elif 'sounder_if' in self.project:
                    if self.n_tx_ant==1:
                        self.dac_tile_block_dic = {2: [0]}
                        self.dac_tiles_sync = []
                    elif self.n_tx_ant==2:
                        self.dac_tile_block_dic = {0: [0], 2:[0]}
                        self.dac_tiles_sync = []
                    if self.n_rx_ant==1:
                        self.adc_tile_block_dic = {2: [0]}
                        self.adc_tiles_sync = [0,2]
                    elif self.n_rx_ant==2:
                        self.adc_tile_block_dic = {0: [0], 2:[0]}
                        self.adc_tiles_sync = [0,2]
                    
        else:
            if self.board=='rfsoc_2x2':
                if 'sounder_bbf' in self.project:
                    self.dac_tile_block_dic = {0: [0], 1:[0]}
                    self.adc_tile_block_dic = {0: [0], 2:[0]}
                    self.dac_tiles_sync = []
                    self.adc_tiles_sync = []
                elif 'sounder_if' in self.project:
                    self.dac_tile_block_dic = {1: [0]}
                    self.adc_tile_block_dic = {2: [0]}
                    self.dac_tiles_sync = []
                    self.adc_tiles_sync = []
            elif self.board=='rfsoc_4x2':
                if 'sounder_bbf' in self.project:
                    self.dac_tile_block_dic = {0: [0], 2:[0]}
                    self.adc_tile_block_dic = {0: [0], 2:[0]}
                    self.dac_tiles_sync = []
                    self.adc_tiles_sync = []
                elif 'sounder_if' in self.project:
                    self.dac_tile_block_dic = {2: [0]}
                    self.adc_tile_block_dic = {2: [0]}
                    self.dac_tiles_sync = []
                    self.adc_tiles_sync = []

        if 'ddr4' in self.project:
            if 'sounder_bbf' in self.project:
                self.n_par_strms_tx = 4
                self.n_par_strms_rx = 4
            elif 'sounder_if' in self.project:
                self.n_par_strms_tx = 8
                self.n_par_strms_rx = 4
        else:
            if 'sounder_bbf' in self.project:
                self.n_par_strms_tx = 4
                self.n_par_strms_rx = 4
            elif 'sounder_if' in self.project:
                self.n_par_strms_tx = 4
                self.n_par_strms_rx = 4
        
        if 'ddr4' in self.project:
            if 'sounder_bbf' in self.project:
                self.tx_mode = 2
                self.rx_mode = 1
            elif 'sounder_if' in self.project:
                self.tx_mode = 1
                self.rx_mode = 1
        else:
            if 'sounder_bbf' in self.project:
                self.tx_mode = 1
                self.rx_mode = 1
            elif 'sounder_if' in self.project:
                self.tx_mode = 1
                self.rx_mode = 1



class RFSoC(General):
    def __init__(self, config: RFSoCConfig, **overrides):
        super().__init__(config, **overrides)

        self.txtd = None
        self.rxtd = None

        if self.config.RFFE=='sivers':
            self.init_sivers(config=config)
        elif self.config.RFFE=='piradio':
            pass
        elif self.config.RFFE=='none':
            pass

        self.load_bit_file()
        self.allocate_input(n_frame=self.config.n_frame_rd)
        self.allocate_output(n_frame=self.config.n_frame_wr)
        self.gpio_init()
        self.clock_init()
        self.verify_clock_tree()
        self.init_rfdc()
        if 'ddr4' in self.config.project:
            self.config.dac_tiles_sync_hex = 0x0
            for id in self.config.dac_tiles_sync:
                self.config.dac_tiles_sync_hex += 0x1 << id
            self.config.adc_tiles_sync_hex = 0x0
            for id in self.config.adc_tiles_sync:
                self.config.adc_tiles_sync_hex += 0x1 << id
            self.init_tile_sync()
            self.sync_tiles(dacTiles=self.config.dac_tiles_sync_hex, adcTiles=self.config.adc_tiles_sync_hex)
        self.init_dac()
        self.init_adc()
        if 'sounder_if' in self.config.project:
            self.set_dac_mixer()
            self.set_adc_mixer()
        self.dma_init()
        if self.config.run_tcp_server:
            self.tcp_comm = Tcp_Comm_RFSoC(config)
            self.tcp_comm.init_tcp_server()

        self.print("rfsoc initialization done", thr=1)


    def load_bit_file(self, verbose=False):
        self.print("Starting to load the bit-file", thr=1)

        self.ol = Overlay(self.config.bit_file_path)
        if verbose:
            self.ol.ip_dict
            # ol?

        self.print("Bit-file loading done", thr=1)


    def init_sivers(self, config=None):
        self.print("Starting Sivers EVK controller", thr=1)
        self.siversControllerObj = siversController(config)
        self.siversControllerObj.init()
        self.print("Sivers EVK controller is loaded", thr=1)


    def run_tcp(self):
        self.tcp_comm.obj_rfsoc = self
        self.tcp_comm.run_tcp_server(self.tcp_comm.parse_and_execute)
    

    def allocate_input(self, n_frame=1):
        size = self.config.n_rx_ant * n_frame * self.n_samples * 2
        if 'ddr4' in self.config.project:
            self.adc_rx_buffer = allocate(shape=(size,), target=self.ol.ddr4_0, dtype=np.int16)
            # self.adc_rx_buffer = allocate(shape=(size,), dtype=np.int16)
        else:
            self.adc_rx_buffer = allocate(shape=(size,), dtype=np.int16)
        self.print("Input buffers allocation done", thr=1)


    def allocate_output(self, n_frame=1):
        size = self.config.n_tx_ant * n_frame * self.n_samples * 2
        self.dac_tx_buffer = allocate(shape=(size,), dtype=np.int16)
        self.print("Output buffers allocation done", thr=1)


    def gpio_init(self):
        self.gpio_dic = {}

        if 'ddr4' in self.config.project:
            if self.config.board=='rfsoc_2x2':
                self.gpio_dic['lmk_reset'] = GPIO(GPIO.get_gpio_pin(84), 'out')
            elif self.config.board=='rfsoc_4x2':
                self.gpio_dic['lmk_reset'] = GPIO(GPIO.get_gpio_pin(-78+7), 'out')
            self.gpio_dic['dac_mux_sel'] = GPIO(GPIO.get_gpio_pin(3), 'out')
            self.gpio_dic['adc_enable'] = GPIO(GPIO.get_gpio_pin(34), 'out')
            self.gpio_dic['dac_enable'] = GPIO(GPIO.get_gpio_pin(2), 'out')
            self.gpio_dic['adc_reset'] = GPIO(GPIO.get_gpio_pin(32), 'out')
            self.gpio_dic['dac_reset'] = GPIO(GPIO.get_gpio_pin(0), 'out')
            self.gpio_dic['led'] = GPIO(GPIO.get_gpio_pin(80), 'out')
        else:
            self.gpio_dic['dac_mux_sel'] = GPIO(GPIO.get_gpio_pin(0), 'out')
            self.gpio_dic['adc_enable'] = GPIO(GPIO.get_gpio_pin(3), 'out')
            self.gpio_dic['dac_enable'] = GPIO(GPIO.get_gpio_pin(1), 'out')
            self.gpio_dic['adc_reset'] = GPIO(GPIO.get_gpio_pin(2), 'out')
            self.gpio_dic['dac_reset'] = GPIO(GPIO.get_gpio_pin(4), 'out')

        if 'ddr4' in self.config.project:
            self.gpio_dic['led'].write(0)
            self.gpio_dic['dac_mux_sel'].write(1)
            self.gpio_dic['adc_enable'].write(0)
            self.gpio_dic['dac_enable'].write(0)
            self.gpio_dic['adc_reset'].write(1)
            self.gpio_dic['dac_reset'].write(1)
        else:
            self.gpio_dic['dac_mux_sel'].write(0)
            self.gpio_dic['adc_enable'].write(0)
            self.gpio_dic['dac_enable'].write(0)
            self.gpio_dic['adc_reset'].write(0)
            self.gpio_dic['dac_reset'].write(0)

        self.print("PS-PL GPIOs initialization done", thr=1)


    def clock_init(self):
        if 'ddr4' in self.config.project:
            self.gpio_dic['lmk_reset'].write(1)
            self.gpio_dic['lmk_reset'].write(0)

        xrfclk.set_ref_clks(lmk_freq=self.config.rfsoc_lmk_freq_mhz, lmx_freq=self.config.rfsoc_lmx_freq_mhz)
        self.print("Xrfclk initialization done", thr=1)


    def verify_clock_tree(self):
        if 'ddr4' in self.config.project:
            status = self.ol.clocktreeMTS.clk_wiz_0.read(self.config.CLOCKWIZARD_LOCK_ADDRESS)
            if (status != 1):
                raise Exception("The MTS ClockTree has failed to LOCK. Please verify board clocking configuration")
        self.print("Verifying clock tree done", thr=1)


    def init_rfdc(self):
        self.rfdc = self.ol.usp_rf_data_converter_0
        self.print("RFDC initialization done", thr=1)


    def init_tile_sync(self):
        dacTiles = min(self.config.dac_tiles_sync_hex, 0x1)
        adcTiles = min(self.config.adc_tiles_sync_hex, 0x1)
        self.sync_tiles(dacTiles=dacTiles, adcTiles=adcTiles)
        self.ol.clocktreeMTS.clk_wiz_0.mmio.write_reg(self.config.CLOCKWIZARD_RESET_ADDRESS, self.config.CLOCKWIZARD_RESET_TOKEN)
        time.sleep(0.1)

        # for id in self.config.dac_tile_block_dic:
        for id in list(set(list(self.config.dac_tile_block_dic.keys()) + self.config.dac_tiles_sync)):
            self.rfdc.dac_tiles[id].Reset()

        for toggleValue in range(0,1):
            # for id in self.config.adc_tile_block_dic:
            for id in list(set(list(self.config.adc_tile_block_dic.keys()) + self.config.adc_tiles_sync)):
                self.rfdc.adc_tiles[id].SetupFIFO(toggleValue)
        self.print("Tiles sync initialization done", thr=1)
    

    def sync_tiles(self, dacTiles = 0, adcTiles = 0):
        self.rfdc.mts_dac_config.RefTile = 0  # MTS starts at DAC Tile 228
        self.rfdc.mts_adc_config.RefTile = 0  # MTS starts at ADC Tile 224
        self.rfdc.mts_dac_config.Target_Latency = -1
        self.rfdc.mts_adc_config.Target_Latency = -1
        if dacTiles > 0:
            self.rfdc.mts_dac_config.Tiles = dacTiles # group defined in binary 0b1111
            self.rfdc.mts_dac_config.SysRef_Enable = 1
            self.rfdc.mts_dac()
        else:
            self.rfdc.mts_dac_config.Tiles = 0x0
            self.rfdc.mts_dac_config.SysRef_Enable = 0

        if adcTiles > 0:
            self.rfdc.mts_adc_config.Tiles = adcTiles
            self.rfdc.mts_adc_config.SysRef_Enable = 1
            self.rfdc.mts_adc()
        else:
            self.rfdc.mts_adc_config.Tiles = 0x0
            self.rfdc.mts_adc_config.SysRef_Enable = 0
        self.print("Tiles sync done", thr=1)
    

    def init_dac(self):
        if 'sounder_if' in self.config.project and not 'ddr4' in self.config.project:
            # self.dac_tile.Reset()
            # self.dac_tile.SetupFIFO(True)

            for id in self.config.dac_tile_block_dic:
                self.rfdc.dac_tiles[id].Reset()
            for id in self.config.dac_tile_block_dic:
                self.rfdc.dac_tiles[id].SetupFIFO(True)
        self.print("DAC init and reset done", thr=1)


    def init_adc(self):
        if 'sounder_if' in self.config.project and not 'ddr4' in self.config.project:
            # # self.adc_tile.Reset()
            # # self.adc_tile.SetupFIFO(True)
            # for toggleValue in range(0, 1):
            #     self.adc_tile.SetupFIFO(toggleValue)

            # for id in self.config.adc_tile_block_dic:
            #     self.rfdc.adc_tiles[id].Reset()
            for toggleValue in range(0,1):
                for id in self.config.adc_tile_block_dic:
                    self.rfdc.adc_tiles[id].SetupFIFO(toggleValue)
        self.print("ADC init and reset done", thr=1)


    def set_dac_mixer(self, mix_freq=None, do_rfsoc_mixer_settings=None):
        result = True

        if mix_freq is None:
            mix_freq = self.config.mix_freq_dac
        if do_rfsoc_mixer_settings is None:
            do_rfsoc_mixer_settings = self.config.do_rfsoc_mixer_settings

        cofig_str = 'DAC configs: mix_freq: {}, rfsoc_mix_phase_off: {:.3f}'.format(mix_freq, self.config.rfsoc_mix_phase_off)
        cofig_str += ', DynamicPLLConfig: ' + str(self.config.DynamicPLLConfig)
        cofig_str += ', do_rfsoc_mixer_settings: ' + str(do_rfsoc_mixer_settings)
        self.print(cofig_str, thr=2)

        for tile_id in self.config.dac_tile_block_dic:
            for block_id in self.config.dac_tile_block_dic[tile_id]:
                dac_tile = self.rfdc.dac_tiles[tile_id]
                dac_block = dac_tile.blocks[block_id]

                if self.config.do_rfsoc_pll_settings:
                    dac_tile.DynamicPLLConfig(self.config.DynamicPLLConfig[0], self.config.DynamicPLLConfig[1], self.config.DynamicPLLConfig[2])
                # print(dac_block.MixerSettings)
                if do_rfsoc_mixer_settings:
                    dac_block.MixerSettings['Freq'] = mix_freq/1e6
                    dac_block.MixerSettings['PhaseOffset'] = self.config.rfsoc_mix_phase_off
                    # dac_block.MixerSettings['EventSource'] = xrfdc.EVNT_SRC_IMMEDIATE
                    dac_block.MixerSettings['EventSource'] = xrfdc.EVNT_SRC_TILE
                    dac_block.UpdateEvent(xrfdc.EVNT_SRC_TILE)

        self.print("DAC Mixer Settings done", thr=1)
        
        return result


    def set_adc_mixer(self, mix_freq=None, do_rfsoc_mixer_settings=None):
        result = True
        
        if mix_freq is None:
            mix_freq = self.config.mix_freq_adc
        if do_rfsoc_mixer_settings is None:
            do_rfsoc_mixer_settings = self.config.do_rfsoc_mixer_settings

        cofig_str = 'ADC configs: mix_freq: {:.2e}, rfsoc_mix_phase_off: {:.2f}'.format(mix_freq, self.config.rfsoc_mix_phase_off)
        cofig_str += ', DynamicPLLConfig: ' + str(self.config.DynamicPLLConfig)
        cofig_str += ', do_rfsoc_mixer_settings: ' + str(do_rfsoc_mixer_settings)
        self.print(cofig_str, thr=2)

        for tile_id in self.config.adc_tile_block_dic:
            for block_id in self.config.adc_tile_block_dic[tile_id]:
                adc_tile = self.rfdc.adc_tiles[tile_id]
                adc_block = adc_tile.blocks[block_id]

                if self.config.do_rfsoc_pll_settings:
                    adc_tile.DynamicPLLConfig(self.config.DynamicPLLConfig[0], self.config.DynamicPLLConfig[1], self.config.DynamicPLLConfig[2])
                if do_rfsoc_mixer_settings:
                    # adc_block.NyquistZone = 1
                    # adc_block.MixerSettings = {
                    #     'CoarseMixFreq'  : xrfdc.COARSE_MIX_BYPASS,
                    #     'EventSource'    : xrfdc.EVNT_SRC_TILE,
                    #     'FineMixerScale' : xrfdc.MIXER_SCALE_1P0,
                    #     'Freq'           : -1*mix_freq/1e6,
                    #     'MixerMode'      : xrfdc.MIXER_MODE_R2C,
                    #     'MixerType'      : xrfdc.MIXER_TYPE_FINE,
                    #     'PhaseOffset'    : 0.0
                    # }
                    adc_block.MixerSettings['Freq'] = -1*mix_freq/1e6
                    adc_block.MixerSettings['PhaseOffset'] = self.config.rfsoc_mix_phase_off
                    # adc_block.MixerSettings['EventSource'] = xrfdc.EVNT_SRC_IMMEDIATE
                    adc_block.MixerSettings['EventSource'] = xrfdc.EVNT_SRC_TILE
                    # adc_block.UpdateEvent(xrfdc.EVENT_MIXER)
                    adc_block.UpdateEvent(xrfdc.EVNT_SRC_TILE)
                    # adc_block.MixerSettings['Freq'] = -1*mix_freq/1e6
            
        self.print("ADC Mixer Settings done", thr=1)

        return result


    def dma_init(self):
        if 'ddr4' in self.config.project:
            self.ol.dac_path.axi_dma_0.set_up_tx_channel()
            self.dma_tx = self.ol.dac_path.axi_dma_0.sendchannel
        else:
            self.dma_tx = self.ol.TX_loop.axi_dma_tx.sendchannel
        self.print("TX DMA setup done", thr=1)

        if 'ddr4' in self.config.project:
            self.ol.adc_path.axi_dma_0.set_up_rx_channel()
            self.dma_rx = self.ol.adc_path.axi_dma_0.recvchannel
            self.rx_reg = self.ol.adc_path.axis_flow_ctrl_0
        else:
            self.dma_rx = self.ol.RX_Logic.axi_dma_rx.recvchannel
        self.print("RX DMA setup done", thr=1)


    def load_data_to_tx_buffer(self, txtd):
        self.txtd = txtd
        txtd_dac = self.txtd * (2 ** (self.config.dac_bits + 1) - 1)

        if 'sounder_if' in self.config.project:
            txtd_dac_interleaved = np.zeros(np.prod(txtd_dac.shape)*2, dtype='int16').reshape(-1, self.config.n_par_strms_tx//2, 2)
            for i in range(self.config.n_tx_ant):
                txtd_dac_ant = txtd_dac[i].reshape(-1,self.config.n_par_strms_tx//2)
                if self.config.tx_mode==1:
                    txtd_dac_interleaved[i::self.config.n_tx_ant,:,0] = np.int16(txtd_dac_ant.real)
                    txtd_dac_interleaved[i::self.config.n_tx_ant,:,1] = np.int16(txtd_dac_ant.imag)
                elif self.config.tx_mode==2:
                    txtd_dac_interleaved[i::self.config.n_tx_ant,:,0] = np.int16(txtd_dac_ant.imag)
                    txtd_dac_interleaved[i::self.config.n_tx_ant,:,1] = np.int16(txtd_dac_ant.real)
                else:
                    raise ValueError('Unsupported TX mode: %d' %(self.config.tx_mode))
            
        else:
            txtd_dac = txtd_dac.reshape(txtd_dac.shape[0], -1, self.config.n_par_strms_tx)
            txtd_dac_interleaved = np.zeros((np.prod(txtd_dac.shape)*2//self.config.n_par_strms_tx, self.config.n_par_strms_tx), dtype='int16')
            for i in range(self.config.n_tx_ant):
                if self.config.tx_mode==1:
                    txtd_dac_interleaved[i*2::self.config.n_tx_ant*2,:] = np.int16(txtd_dac[i].real)
                    txtd_dac_interleaved[i*2+1::self.config.n_tx_ant*2,:] = np.int16(txtd_dac[i].imag)
                elif self.config.tx_mode==2:
                    txtd_dac_interleaved[i*2::self.config.n_tx_ant*2,:] = np.int16(txtd_dac[i].imag)
                    txtd_dac_interleaved[i*2+1::self.config.n_tx_ant*2,:] = np.int16(txtd_dac[i].real)
                else:
                    raise ValueError('Unsupported RX mode: %d' %(self.config.rx_mode))

        self.dac_tx_buffer[:] = txtd_dac_interleaved.flatten()[:]

        self.print("Loading txtd data to DAC TX buffer done", thr=1)


    def load_data_from_rx_buffer(self):
        rx_data = np.array(self.adc_rx_buffer).astype('int16') / (2 ** (self.config.adc_bits + 1) - 1)
        self.rxtd = []
        
        rx_data = rx_data.reshape(-1, self.config.n_par_strms_rx)
        for i in range(self.config.n_rx_ant):
            if self.config.rx_mode==1:
                rx_data_ant = rx_data[i*2::self.config.n_rx_ant*2,:] + 1j * rx_data[i*2+1::self.config.n_rx_ant*2,:]
            elif self.config.rx_mode==2:
                rx_data_ant = rx_data[i*2+1::self.config.n_rx_ant*2,:] + 1j * rx_data[i*2::self.config.n_rx_ant*2,:]
            else:
                raise ValueError('Unsupported RX mode: %d' %(self.config.rx_mode))
            self.rxtd.append(rx_data_ant.flatten())
        
        self.rxtd = np.array(self.rxtd)
        self.print("Loading rxtd data from ADC RX buffer done", thr=5)


    def send_frame(self, txtd):
        self.load_data_to_tx_buffer(txtd)

        self.gpio_dic['dac_mux_sel'].write(0)
        self.gpio_dic['dac_enable'].write(0)
        if 'ddr4' in self.config.project:
            self.gpio_dic['dac_reset'].write(1)  # Reset ON
        else:
            self.gpio_dic['dac_reset'].write(0)
        time.sleep(0.5)
        if 'ddr4' in self.config.project:
            self.gpio_dic['dac_reset'].write(0)  # Reset OFF
        else:
            self.gpio_dic['dac_reset'].write(1)
        self.dma_tx.transfer(self.dac_tx_buffer)
        self.dma_tx.wait()
        self.gpio_dic['dac_mux_sel'].write(1)
        self.gpio_dic['dac_enable'].write(1)

        # self.dma_tx.wait()
        time.sleep(0.1)
        self.print("Frame sent via DAC", thr=1)


    def recv_frame_one(self, n_frame=1):
        # if 'ddr4' in self.config.project:
        #     self.gpio_dic['led'].write(1)
        if 'ddr4' in self.config.project:
            # Suspicous code
            self.rx_reg.write(0, self.config.n_rx_ant * self.n_samples // self.config.n_par_strms_rx)
            # self.rx_reg.write(0, self.n_samples // self.config.n_par_strms_rx)
            self.rx_reg.write(4, self.config.n_skip // self.config.n_par_strms_rx)
            self.rx_reg.write(8, self.config.n_rx_ant * n_frame * self.n_samples * 4)      # Must have self.config.n_rx_ant multiplier to work correctly

            self.gpio_dic['adc_reset'].write(0)
        else:
            self.gpio_dic['adc_enable'].write(0)
            self.gpio_dic['adc_reset'].write(0)  # Reset ON
            time.sleep(0.01)
            self.gpio_dic['adc_reset'].write(1)  # Reset OFF

        self.dma_rx.transfer(self.adc_rx_buffer)
        self.gpio_dic['adc_enable'].write(1)
        self.dma_rx.wait()

        self.gpio_dic['adc_enable'].write(0)

        if 'ddr4' in self.config.project:
            self.gpio_dic['adc_reset'].write(1)
        else:
            self.gpio_dic['adc_reset'].write(0)

        self.load_data_from_rx_buffer()
        # if 'ddr4' in self.config.project:
        #     self.gpio_dic['led'].write(0)
        self.print("Frames received from ADC", thr=5)

        return self.rxtd
    

    def recv_frame(self, n_frame=1):
        rxtd = np.zeros((len(self.config.beam_test), self.config.n_rx_ant*n_frame*self.n_samples), dtype='complex')

        for i, beam_index in enumerate(self.config.beam_test):
            if self.config.RFFE=='sivers':
                self.siversControllerObj.setBeamIndexRX(beam_index)
            self.recv_frame_one(n_frame=n_frame)
            rxtd[i,:] = self.rxtd.flatten()

        rxfd = fft(rxtd, axis=1)
        rxfd = np.roll(rxfd, 1, axis=1)
        Hest = rxfd * np.conj(self.txfd)
        hest = ifft(Hest, axis=1)
        hest = hest.flatten()
        # re = hest.real.astype(np.int16)
        # im = hest.imag.astype(np.int16)

        self.print("Frames received from ADC", thr=5)

        # return np.concatenate((re, im))
        return hest


