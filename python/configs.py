import os
import copy
from dataclasses import dataclass
import numpy as np
from sounder import SounderConfig



@dataclass
class Configs_Class(SounderConfig):
    def __init__(self):
        super().__init__()

        measurement_type=''            # Type of the measurement

        self.init()
        self.populate_measurement_parameters()


    def init(self):

        self.ant_d_m = [0.026]               # Antenna spacing in meters
        self.n_rx_ch_eq=1
        self.wb_sc_range=[-260,260]
        self.rx_same_delay=False
        # self.sparse_ch_samp_range=[-5,100]
        # self.sparse_ch_n_ignore=5
        self.n_frame_rd=32
        self.n_rd_rep=1
        self.anim_interval = 100
        self.save_parameters=True
        self.plot_configs = {'title_size': 11, 'title_max_chars': 35, 'xaxis_size': 10, 'yaxis_size': 10, 'ticks_size': 10, 'legend_size': 10, 'line_width': 1.0, 'marker_size': 8, 'hspace': 0.5, 'wspace': 0.5}

        # self.overwrite_level=False
        # self.plot_level=0
        # self.verbose_level=3


        # self.update_rfsoc_files = True
        # self.host_files_base_addr = "/home/wirelesslab914/ali/sounder_rfsoc/RFSoC_SDR/python/"
        self.host_ip = '192.168.2.1'
        # self.host_username = 'wirelesslab914'
        self.host_username = 'alira'
        self.host_password = ''



        # self.measurement_type = 'plot_saved_signal'
        # self.measurement_type = 'RFSoC_demo_simple'
        # self.measurement_type = 'mmw_demo_simple'
        self.measurement_type = 'FR3_spectrum_sweep'
        # self.measurement_type = 'FR3_demo_simple'
        # self.measurement_type = 'FR3_demo_multi_freq'
        # self.measurement_type = 'FR3_nyu_3state'
        # self.measurement_type = 'FR3_nyu_13state'
        # self.measurement_type = 'FR3_ant_calib'
        # self.measurement_type = 'FR3_beamforming'
        # self.measurement_type = 'FR3_cfo'
        # self.measurement_type = 'stream_to_matlab'
        # self.measurement_type = 'turtlebot_demo'

        self.host_role = 'client'
        # self.host_role = 'client_master'
        # self.host_role = 'client_slave'

        self.transmit_signal=True


    def populate_measurement_parameters(self):

        if self.host_role == 'client':
            self.transmit_signal=True
        elif self.host_role == 'client_master':
            self.transmit_signal=False
        elif self.host_role == 'client_slave':
            self.transmit_signal=True


        h00 = "h|0|0|circshift|mag|dbmag"
        h01 = "h|0|1|circshift|mag|dbmag"
        h10 = "h|1|0|circshift|mag|dbmag"
        h11 = "h|1|1|circshift|mag|dbmag"

        rxfd00 = "rxtd|0|0|fft|fftshift|mag|dbmag"
        rxfd10 = "rxtd|1|0|fft|fftshift|mag|dbmag"
        rxfd00_ph = "rxtd|0|0|fft|fftshift|phase/2pi"
        rxfd10_ph = "rxtd|1|0|fft|fftshift|phase/2pi"
        rxfd_ph_diff = [rxfd00_ph, '-', rxfd10_ph]

        rxtd00 = "rxtd|0|0|mag"
        rxtd10 = "rxtd|1|0|mag"
        rxtd00_ph = "rxtd|0|0|phase/2pi"
        rxtd10_ph = "rxtd|1|0|phase/2pi"
        rxtd00_r = "rxtd|0|0|real"
        rxtd00_i = "rxtd|0|0|imag"
        rxtd10_r = "rxtd|1|0|real"
        rxtd10_i = "rxtd|1|0|imag"
        rxtd_ph_diff = [rxtd00_ph, '-', rxtd10_ph]

        IQ00 = ["rxtd|0|0|fft|fftshift|IQ"]
        aoa_gauge = ["aoa_gauge|0|0"]
        


        if self.measurement_type == 'plot_saved_signal':
            self.saved_sig_plot = ['signal']
            self.sig_save_path=os.path.join(self.sig_dir, '0_tx1_rx1_rx_rotate.npz')
            self.wb_sc_range=[-260,260]
            self.animate_plot_mode=[[h00], [rxtd00_r, rxtd00_i], [rxfd00]]
            self.rx_chain = ['sync_time', 'channel_est']
            self.freq_hop_config['list'] = [6.5e9, 10e9, 15.0e9, 20.0e9]

            self.tx_sig_sim = 'shifted'
            self.sig_gen_mode = 'ZadoffChu'


        elif self.measurement_type == 'mmw_demo_simple':
            self.host_role = 'client'
            self.RFFE='sivers'
            self.wb_sc_range=[-300,-100]
            self.transmit_signal=False
            self.animate_plot_mode=[[h00], [rxtd00_r, rxtd00_i], [rxfd00]]
            self.rx_chain = ['sync_time', 'channel_est']
            # self.rx_chain = ['sync_time', 'channel_est', 'channel_eq']
            self.freq_hop_config['list'] = [60.0e9]
            # self.tx_sig_sim = 'orthogonal'
            # self.sig_gen_mode = 'ZadoffChu'


        elif self.measurement_type == 'RFSoC_demo_simple':
            # self.mix_freq_adc=0e6
            # self.do_rfsoc_mixer_settings=True
            # self.wb_sc_range = [0,300]

            # self.animate_plot_mode = [[h00], [rxtd00_r, rxtd00_i], [rxfd00, rxfd10]]
            self.animate_plot_mode = [[h00], [rxfd00, rxfd10], aoa_gauge]
            # self.animate_plot_mode = [[rxtd00_r, rxtd10_r], [rxtd10_r, rxtd10_i], [rxfd00, rxfd10]]
            # self.animate_plot_mode=[[rxtd00_r, rxtd00_i], rxtd_ph_diff, rxfd_ph_diff]

            self.rx_chain = ['sync_time', 'channel_est']
            # self.rx_chain = []

            self.tx_sig_sim = 'same'
            # self.sig_gen_mode = 'ZadoffChu'
            self.sig_gen_mode = 'fft'
            self.sig_modulation = '4qam'


        elif self.measurement_type == 'FR3_spectrum_sweep':
            self.animate_plot_mode=[[rxfd00]]
            self.rx_chain = []

            self.freq_hop_config['list'] = [10e9]
            self.tx_sig_sim = 'same'
            self.sig_gen_mode = 'fft'
            self.sig_mode = 'wideband'
            self.sig_modulation = '4qam'

            # self.save_list = ['signal']
            self.n_save = 32
            self.measurement_configs = []

            self.network_topology = {
                'rfsoc_tx': {'type': 'rfsoc', 'role': 'tx', 'ip': '192.168.3.1', 'protocol': 'tcp'},
                'rfsoc_rx': {'type': 'rfsoc', 'role': 'rx', 'ip': '192.168.3.1', 'protocol': 'tcp'},
                'lintrack': {'type': 'lintrack', 'role': 'rx', 'ip': '192.168.137.100', 'protocol': 'tcp'},
                'turntable': {'type': 'turntable', 'role': 'rx', 'port': '/dev/ttyACM0', 'baudrate': 115200, 'protocol': 'tcp'},
                'piradio_tx': {'type': 'piradio', 'role': 'tx', 'ip': '192.168.137.51', 'protocol': 'http', 'username': 'ubuntu', 'password': 'temppwd'},
                'piradio_rx': {'type': 'piradio', 'role': 'rx', 'ip': '192.168.137.51', 'protocol': 'http', 'username': 'ubuntu', 'password': 'temppwd'},
                'controller_slave': {'type': 'controller', 'role': 'tx', 'ip': '192.168.1.1', 'protocol': 'tcp'},
                'host': {'type': 'host', 'role': 'rx', 'ip': '192.168.3.100', 'protocol': 'ssh'},
            }

            self.action_loop = [
                                f'piradio_rx/set_gain_db_rx/-3:20:20/',
                                f'self/switch_sig_size/[8,16,32,128]/',
                                # f'piradio_rx/set_gain_db_rx/[3,7,10,17]/',
                                # f'self/switch_sig_size/1:256:20:log/',
                                f'self/switch_sig_ss/1:10:10/',
                                # f'self/wait/[1]/',
                                f'rfsoc_rx/capture/[10]/',
                                f'self/save/[1]/["signal"]/m'
                                ]


        elif self.measurement_type == 'FR3_demo_simple':
            self.animate_plot_mode=[[h00], [rxtd00_r, rxtd00_i], [rxfd00]]
            self.rx_chain = ['sync_time', 'channel_est']
            self.freq_hop_config['list'] = [6.5e9]
            self.tx_sig_sim = 'orthogonal'
            # self.sig_gen_mode = 'ZadoffChu'
            
            # self.save_list = ['signal']
            # self.measurement_configs = ["test"]
            # self.n_save = 256


        elif self.measurement_type == 'FR3_demo_multi_freq':
            self.animate_plot_mode=[[h00], [rxtd00_r, rxtd00_i], [rxfd00]]
            self.rx_chain = ['sync_time', 'channel_est']
            self.freq_hop_config['list'] = [6.5e9, 8.75e9, 10.0e9]
            self.tx_sig_sim = 'orthogonal'
            # self.sig_gen_mode = 'ZadoffChu'



        elif self.measurement_type == 'FR3_ant_calib':
            self.animate_plot_mode=[[h00], [rxtd00_r, rxtd00_i], [rxfd00]]
            self.rx_chain = ['sync_time', 'channel_est']
            self.use_turntable = True
            self.rotation_range_deg = [-90,90]
            self.rotation_step_deg = 1
            self.rotation_delay = 0.5

            # self.freq_hop_config['list'] = [10e9, 15.0e9]
            self.freq_hop_config['mode'] = 'sweep'
            self.freq_hop_config['range'] = [6.0e9, 22.5e9]
            self.freq_hop_config['step'] = 0.5e9

            self.tx_sig_sim = 'shifted'
            self.sig_gen_mode = 'ZadoffChu'

            self.save_list = ['signal']
            self.n_save = 32
            self.measurement_configs = []
            # self.measurement_configs.append("tx1_rx1_rx_rotate")
            self.measurement_configs.append("tx2_rx2_rx_rotate")


        elif self.measurement_type == 'FR3_beamforming':
            self.animate_plot_mode=[[h00], [rxtd00_r, rxtd00_i], [rxfd00]]
            self.rx_chain = ['sync_time', 'channel_est']
            self.use_turntable = True
            self.rotation_range_deg = [-90,90]
            self.rotation_step_deg = 2
            self.rotation_delay = 0.5

            self.freq_hop_config['list'] = [10e9]

            self.sig_gen_mode = 'fft'
            self.tx_sig_sim = 'same'

            self.beamforming=True
            self.steer_rad = [0, 0]

            self.save_list = ['signal']
            self.n_save = 32
            self.measurement_configs = [6.5]
            self.measurement_configs.append("bf_phi_{}".format(self.steer_rad[0]))



        elif self.measurement_type == 'FR3_nyu_3state':
            self.animate_plot_mode=[[h00], [rxtd00_r, rxtd00_i], [rxfd00]]
            self.save_format = 'mat'
            self.rx_chain = ['sync_time', 'channel_est']
            self.use_turntable = True
            self.rotation_range_deg = [-45,45]
            self.rotation_step_deg = 45
            self.rotation_delay = 0.5
            self.freq_hop_config['list'] = [6.5e9, 8.75e9, 10.0e9, 15.0e9, 21.7e9]
            self.tx_sig_sim = 'shifted'
            self.sig_gen_mode = 'ZadoffChu'

            self.save_list = ['signal']
            self.n_save = 256
            
            # Naming: _Position_TX-Orient_RX-Orient_Reflect/NoReflect(r/n)-Blockage/NoBlockage(b/n)
            # Orientations: alpha: 0, beta: 45, gamma: -45
            # Good Pi-radio gains for OTA: 20dB for TX channels and 21dB for RX channels
            # Good Pi-radio gains for cabled calibration: 10dB for TX channels and 15dB for RX channels

            self.measurement_configs = []
            # self.measurement_configs.append('calib_1-1_2-2')
            # self.measurement_configs.append('calib_1-2_2-1')

            self.measurement_configs.append('A_beta_<rxorient>_n')
            self.measurement_configs.append('A_alpha_<rxorient>_n')
            self.measurement_configs.append('A_gamma_<rxorient>_n')
            # self.measurement_configs.append('B_alpha_<rxorient>_n')
            # self.measurement_configs.append('B_gamma_<rxorient>_n')
            # self.measurement_configs.append('B_beta_<rxorient>_n')
            # self.measurement_configs.append('C_beta_<rxorient>_n')
            # self.measurement_configs.append('C_alpha_<rxorient>_n')
            # self.measurement_configs.append('C_gamma_<rxorient>_n')
            # self.measurement_configs.append('D_gamma_<rxorient>_n')
            # self.measurement_configs.append('D_alpha_<rxorient>_n')
            # self.measurement_configs.append('D_beta_<rxorient>_n')
            # self.measurement_configs.append('E_beta_<rxorient>_n')
            # self.measurement_configs.append('E_alpha_<rxorient>_n')
            # self.measurement_configs.append('E_gamma_<rxorient>_n')

            # self.measurement_configs.append('A_beta_beta_n')
            # self.measurement_configs.append('A_beta_alpha_n')
            # self.measurement_configs.append('A_beta_gamma_n')
            # self.measurement_configs.append('A_alpha_gamma_n')
            # self.measurement_configs.append('A_alpha_alpha_n')
            # self.measurement_configs.append('A_alpha_beta_n')
            # self.measurement_configs.append('A_gamma_beta_n')
            # self.measurement_configs.append('A_gamma_alpha_n')
            # self.measurement_configs.append('A_gamma_gamma_n')


        elif self.measurement_type == 'FR3_nyu_13state':
            self.animate_plot_mode=[[h00], [rxtd00_r, rxtd00_i], [rxfd00]]

            self.rx_chain = ['sync_time', 'channel_est']
            self.use_turntable = True
            self.rotation_range_deg = [-60,60]
            self.rotation_step_deg = 10
            self.rotation_delay = 0.5

            self.freq_hop_config['list'] = [6.5e9, 8.75e9, 10.0e9, 15.0e9, 21.7e9]

            self.tx_sig_sim = 'shifted'
            self.sig_gen_mode = 'ZadoffChu'

            # self.save_list = ['signal']
            self.save_format = 'mat'
            self.n_save = 256
            
            # Naming: _Position_TX-Orient_RX-Orient_Reflect/NoReflect(r/n)-Blockage/NoBlockage(b/n)
            self.measurement_configs = []
            # self.measurement_configs.append('calib_1-1_2-2')
            # self.measurement_configs.append('calib_1-2_2-1')

            # self.measurement_configs.append('C_alpha_<rxorient>_n')
            # self.measurement_configs.append('C_alpha_<rxorient>_r')
            # self.measurement_configs.append('C_alpha_<rxorient>_b')


        elif self.measurement_type == 'FR3_cfo':
            self.animate_plot_mode=[[h10], [rxtd10_r, rxtd10_i], [rxfd00, rxfd10]]
            self.rx_chain = ['sync_time', 'channel_est']
            
            self.freq_hop_config['list'] = [10.0e9]
            cfo_ppm = -100

            if self.host_role == 'client_master':
                cfo = cfo_ppm * self.freq_hop_config['list'][0] / 1e6
                self.mix_freq_adc += cfo
                self.do_rfsoc_mixer_settings=True

            self.sig_gen_mode = 'fft'
            self.tx_sig_sim = 'orthogonal'
            self.sig_modulation = '4qam'

            self.save_list = ['signal']
            self.n_save = 256
            self.measurement_configs = []
            self.measurement_configs.append("{}GHz_{}ppm".format(self.freq_hop_config['list'][0]/1e9, cfo_ppm))


        elif self.measurement_type == 'turtlebot_demo':

            self.animate_plot_mode = [[h00, h10], [rxfd00, rxfd10], aoa_gauge]

            self.freq_hop_config['list'] = [10.0e9]
            self.rx_chain = ['sync_time', 'channel_est']

            self.tx_sig_sim = 'same'
            self.sig_gen_mode = 'ZadoffChu'

