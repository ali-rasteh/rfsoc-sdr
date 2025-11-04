from backend import *
from backend import be_np as np, be_scp as scipy





class Params_Class_Default(object):
    def __init__(self):
        # Constant parameters
        self.c = constants.c
        self.seed=100

        # Board and RFSoC FPGA project parameters
        self.project='sounder_if_ddr4'      # Type of the project, sounder_bbf_ddr4 or sounder_if_ddr4 or sounder_bbf or sounder_if
        self.measurement_type=''            # Type of the measurement
        self.board='rfsoc_4x2'              # Type of the RFSoC board, rfsoc_4x2 or rfsoc_2x2
        self.bit_file_path=os.path.join(os.getcwd(), 'project_v1-0-58_20241001-150336.bit')       # Path to the bit file for the RFSoC (Without DAC MTS)
        # self.bit_file_path=os.path.join(os.getcwd(), 'project_v1-0-62_20241019-173825.bit')     # Path to the bit file for the RFSoC (With DAC MTS)
        self.mode='client'                  # Mode of operation, server or client or client_master or client_slave
        self.run_tcp_server=True            # If True, runs the TCP server
        self.send_signal=False               # If True, sends TX signal
        self.recv_signal=True               # If True, receives and plots RX signal

        # Plots and logs parameters
        self.plt_frame_id = 0               # Frame ID to plot
        self.overwrite_level=True           # If True, overwrites the plot and verbose levels
        self.plot_level=0                   # Level of plotting outputs
        self.verbose_level=0                # Level of printing output
        self.anim_interval=500              # Animation interval in ms
        self.animate_plot_mode=['h', 'rxfd']        # List of plots to animate
        # dictionary of plot fonts configurations
        self.plot_fonts_dict = {'title_size': 15, 'xaxis_size': 17, 'yaxis_size': 15, 'ticks_size': 15, 'legend_size': 15, 'line_width': 1.2, 'marker_size': 8, 'hspace': 0.4, 'wspace': 0.4}

        # Mixer parameters
        self.mixer_mode='analog'            # Mixer mode, analog or digital
        self.mix_freq=1000e6                # Mixer carrier frequency
        self.mix_phase_off=0.0              # Mixer's phase offset
        self.do_mixer_settings=False        # If True, performs mixer settings
        self.do_pll_settings=False          # If True, performs PLL settings
        self.lmk_freq_mhz=122.88            # LMK frequency in MHz
        self.lmx_freq_mhz=3932.16           # LMX frequency in MHz

        # RFFE and antennas parameters
        self.RFFE='piradio'                 # RF front end to use, piradio or sivers
        self.ant_dim = 1                    # Antenna dimension, 1 or 2
        self.n_tx_ant=2                     # Number of transmitter antennas
        self.n_rx_ant=2                     # Number of receiver antennas
        self.ant_dx_m = 0.02                # Antenna x axis spacing in meters
        self.ant_dy_m = 0.02                # Antenna y axis spacing in meters

        # Connections parameters
        self.control_rfsoc=True             # If True, controls the RFSoC board
        self.control_piradio=False          # If True, controls the PIRadio board
        self.set_piradio_opt_gains = False  # If True, finds and sets the optimal gains for the PIRadio board
        self.set_piradio_opt_losupp = False   # If True, finds and sets the optimal LO suppression bias for the PIRadio board
        self.tcp_localIP = "0.0.0.0"        # Local IP address
        self.tcp_bufferSize=2**10           # TCP buffer size
        self.TCP_port_Cmd=8080              # TCP port for commands
        self.TCP_port_Data=8081             # TCP port for data
        self.rfsoc_server_ip='192.168.3.1'  # RFSoC board IP as the server
        self.lintrack_server_ip='192.168.137.100'   # Linear track controller board IP as the server ('10.18.242.48')
        self.turntable_port = '/dev/ttyACM0'                # Turntable serial port
        self.turntable_baudrate = 115200            # Turntable baudrate
        self.piradio_host = '192.168.137.51'        # PIRadio host IP
        self.piradio_ssh_port = '22'                # PIRadio SSH port
        self.piradio_rest_port = '5111'             # PIRadio REST port
        self.piradio_username = 'ubuntu'            # PIRadio username
        self.piradio_password = 'temppwd'           # PIRadio password
        self.piradio_rest_protocol = 'http'         # PIRadio REST protocol
        self.host_ip = '192.168.3.100'              # Host IP
        self.host_username = 'wirelesslab914'       # Host username
        self.host_password = ''                     # Host password
        self.controller_slave_ip = '192.168.1.1'    # Controller slave IP
        self.piradio_freq_sw_dly_default = 0.1       # PIRadio frequency switch delay (default)
        self.piradio_gain_sw_dly_default = 0.1      # PIRadio gain change delay (default)
        self.piradio_bias_sw_dly_default = 0.1      # PIRadio LO Suppression change delay (default)
        self.piradio_freq_sw_dly = self.piradio_freq_sw_dly_default              # PIRadio frequency switch delay
        self.piradio_gain_sw_dly = self.piradio_gain_sw_dly_default              # PIRadio gain change delay
        self.piradio_bias_sw_dly = self.piradio_bias_sw_dly_default            # PIRadio LO Suppression change delay
        self.piradio_freq_range = [6.0, 22.5e9]     # PIRadio frequency range
        self.stable_fc_piradio = 10e9          # Most stable carrier frequency for the PIRadio board

        # File transfer parameters
        self.update_rfsoc_files = False                     # If True, updates the RFSoC files
        self.modify_rfsoc_files = True                      # If True, modifies the RFSoC files to be true for the server mode
        self.files_dwnld_target = 'rfsoc'                   # Target for file download, rfsoc or raspi
        self.host_files_base_addr = '~/RFSoC_SDR/python/'   # Base address for the host files
        self.host_ip = '192.168.3.100'                      # Host IP address
        self.host_username = 'root'                         # Host username
        self.host_password = 'root'                         # Host password
        self.local_base_addr = "./"                         # Local base address for the files
        
        # Signals information
        self.freq_hop_config = {'mode': 'discrete', 'list': [10.0e9], 'range': [10.0e9, 10.0e9], 'step': 1.0e9}    # Frequency hopping configuration, modes: discrete or sweep
        self.fs=245.76e6 * 4                        # Sampling frequency in RFSoC
        self.fs_tx=self.fs                          # DAC sampling frequency
        self.fs_rx=self.fs                          # ADC sampling frequency
        self.fs_trx=self.fs                         # Sampling frequency used for operations involving both TX and RX
        self.n_samples=1024                         # Number of samples
        self.nfft=self.n_samples                    # Number of FFT points
        self.sig_gen_mode = 'fft'                   # Signal generation mode, time, or fft or ofdm, or ZadoffChu
        self.sig_mode='wideband_null'               # Signal mode, tone_1 or tone_2 or wideband or wideband_null or load
        self.sig_modulation = '4qam'                # Signal modulation type for sounding, 4qam, 16qam, etc
        self.tx_sig_sim = 'same'                    # TX signal similarity between antennas, same or orthogonal or shifted
        self.sig_gain_db=0                          # Transmitter Signal gain in dB
        self.n_frame_wr=1                           # Number of frames to write
        self.n_frame_rd=2                           # Number of frames to read
        self.n_rd_rep=8                             # Number of read repetitions for RX signal
        self.snr_est_db=40                          # SNR for signal estimation
        self.wb_bw_mode='sc'                        # Wideband signal bandwidth mode, sc or freq
        self.wb_sc_range=[-250,250]                 # Wideband signal subcarrier range, used when wb_bw_mode is sc
        self.wb_bw_range=[-250e6,250e6]             # Wideband signal bandwidth range, used when wb_bw_mode is freq
        self.wb_null_sc=0                           # Number of carriers to null in the wideband signal
        self.tone_f_mode='sc'                       # Tone signal frequency mode, sc or freq
        self.sc_tone=10                             # Tone signal subcarrier
        self.f_tone=10.0 * self.fs_tx / self.nfft   # Tone signal frequency
        self.filter_bw_range=[-450e6,450e6]         # Final filter BW range on the RX signal
        self.n_rx_ch_eq=1                           # Number of RX chains for channel equalization
        self.sparse_ch_samp_range=[-6,20]           # Range of samples around the strongest peak to consider for channel estimation
        self.sparse_ch_n_ignore=-1                  # Number of samples to ignore around the strongest peak
        self.rx_same_delay=True                     # If True, all applies the same time shift to all RX antennas
        self.rx_chain=['sync_time', 'channel_est']  # The chain of operations to perform on the RX signal, filter, integrate, sync_time, sync_time_frac, sync_freq, pilot_separate, sys_res_deconv, channel_est, sparse_est, channel_eq
        self.channel_limit = True                   # If True, limits the channel to a specific range in the frequency domain

        # Save parameters
        self.calib_params_dir=os.path.join(os.getcwd(), 'calib/')                           # Calibration parameters directory
        self.calib_params_path=os.path.join(self.calib_params_dir, 'calib_params.npz')      # Calibration parameters path
        self.optimal_gains_path=os.path.join(self.calib_params_dir, 'optimal_gains.json')   # Calibration parameters path
        self.sig_dir=os.path.join(os.getcwd(), 'sigs/')                             # Signals directory
        self.sig_path=os.path.join(self.sig_dir, 'txtd.npz')                        # Signal load path
        self.sig_save_path=os.path.join(self.sig_dir, 'trx.npz')                    # Signal save path
        self.measurement_configs = []                                               # List of measurement configurations
        self.channel_dir=os.path.join(os.getcwd(), 'channels/')                     # Channel directory
        self.channel_save_path=os.path.join(self.channel_dir, 'channel.npz')        # Channel save path
        self.sys_response_path=os.path.join(self.channel_dir, 'sys_response.npz')   # System response save path
        self.figs_dir=os.path.join(os.getcwd(), 'figs/')                            # Figures directory
        self.figs_save_path=os.path.join(self.figs_dir, 'plot.pdf')                 # Figures save path
        self.n_save = 100                                                           # Number of samples to save
        self.save_list = ['', '']                                                   # List of items to save, signal or channel
        self.save_format = 'npz'                                                    # Format to save the data, npz or mat (for MATLAB)
        self.saved_sig_plot = []                                                    # List of saved signal plots
        self.params_dir = os.path.join(os.getcwd(), 'params/')                      # Parameters directory
        self.params_path = os.path.join(self.params_dir, 'params.json')             # Parameters load path
        self.params_save_path = os.path.join(self.params_dir, 'params.json')        # Parameters save path
        self.save_parameters=False                                                  # If True, saves current parameters
        self.load_parameters=False                                                  # If True, loads parameters from the file

        # Calibration parameters
        self.calib_iter = 100           # Number of iterations for calibration

        # Beamforming parameters
        self.beamforming=False          # If True, performs beamforming
        self.steer_theta_deg = 0        # Desired steering elevation in degrees
        self.steer_phi_deg = 30         # Desired steering azimuth in degrees

        # Near field measurements parameters
        self.nf_param_estimate = False                  # If True, performs near field parameter estimation
        self.use_linear_track = False                   # If True, uses the linear track for near field measurements
        self.nf_walls = np.array([[-5,4], [-1,6]])      # Near field walls coordinates in meters
        self.nf_rx_sep_dir = np.array([1,0])            # Direction of the RX antenna separation
        self.nf_tx_sep_dir = np.array([1,0])            # Direction of the TX antenna separation
        self.nf_npath_max = [20,5]                      # 1st number is the maximum number to extract at the 1st round, 2nd number is the maximum number to extract at the 2nd round
        self.nf_stop_thr = 0.03                         # Stopping threshold for the near field parameter estimation
        self.nf_tx_loc = np.array([[0.3,1.0]])          # TX antenna location in meters
        self.nf_rx_loc_sep = np.array([0,0.2,0.4])      # RX locations separation in meters
        self.nf_tx_ant_sep = 0.5                        # TX antenna separation in meters
        self.nf_rx_ant_sep = 0.5 * np.array([1,2,4])    # RX antenna separation in meters

        # Antenna calibration parameters
        self.use_turntable = False                      # If True, uses the turntable for calibration
        self.calibrate_turntable = False                # If True, calibrates the turntable
        self.rotation_range_deg = [-90,90]              # Turntable Rotation range in degrees
        self.rotation_step_deg = 1                      # Turntable Rotation step in degrees
        self.rotation_delay = 0.0                       # Turntable between rotations delay in seconds

        # matlab data transfer parameters
        self.enable_matlab_stream = False          # If True, enables the MATLAB data transfer
        # self.matlab_stream_ip = '10.20.9.65'     # Roberto IP address for the MATLAB data transfer
        self.matlab_stream_ip = '10.20.38.213'     # IP address for the MATLAB data transfer
        self.matlab_stream_port = 50007             # Port for the MATLAB data transfer
        # self.calc_params()



    def calc_params(self):

        if 'h_sparse' in self.animate_plot_mode and 'sparse_est' not in self.rx_chain:
            self.rx_chain.append('sparse_est')

        system_info = platform.uname()
        if "pynq" in system_info.node.lower():
            self.mode = 'server'


        if self.overwrite_level:
            if self.mode == 'server':
                self.plot_level=4
                self.verbose_level=4
            elif 'slave' in self.mode:
                self.plot_level=0
                self.verbose_level=4
            else:
                self.plot_level=0
                self.verbose_level=1


        if self.mode == 'server':
            self.nf_param_estimate=False
            self.control_rfsoc=False
            self.control_piradio=False
            self.use_linear_track=False
            self.use_turntable=False
        elif self.mode == 'client':
            pass
        elif self.mode == 'client_master':
            self.piradio_freq_sw_dly = 0.0
            self.piradio_gain_sw_dly = 0.0
            self.piradio_bias_sw_dly = 0.0
            # self.send_signal=False
            pass
        elif self.mode == 'client_slave':
            self.control_rfsoc=False
            self.use_linear_track=False
            self.use_turntable=False
            # self.piradio_freq_sw_dly = 0.0
            # self.piradio_gain_sw_dly = 0.0
            # self.piradio_bias_sw_dly = 0.0


        if self.mixer_mode=='digital' and self.mix_freq!=0:
            self.mix_freq_dac = 0
            self.mix_freq_adc = 0
        elif self.mixer_mode == 'analog':
            self.mix_freq_dac = self.mix_freq
            self.mix_freq_adc = self.mix_freq
        else:
            self.mix_freq_dac = 0
            self.mix_freq_adc = 0
            
        if 'sounder_bbf' in self.project:
            self.do_mixer_settings=False
            self.do_pll_settings=False
            self.n_tx_ant=1
            self.n_rx_ant=1
        if self.board == "rfsoc_4x2":
            self.do_pll_settings=False

        if self.n_tx_ant==1 and self.n_rx_ant==1:
            self.ant_dim = 1
            self.beamforming = False


        if self.freq_hop_config['mode']=='discrete':
            self.freq_hop_list = self.freq_hop_config['list']
        elif self.freq_hop_config['mode']=='sweep':
            self.freq_hop_list = np.arange(self.freq_hop_config['range'][0], self.freq_hop_config['range'][1]+self.freq_hop_config['step'], self.freq_hop_config['step'])
        else:
            raise ValueError('Invalid freq_hop_config mode: ' + self.freq_hop_config['mode'])
        
        
        if self.save_format != 'npz':
            self.sig_save_path = self.sig_save_path.replace('.npz', '.'+self.save_format)
            self.channel_save_path = self.channel_save_path.replace('.npz', '.'+self.save_format)
        

        self.fc = self.freq_hop_list[0]
        self.wl = self.c / self.fc
        self.ant_dx = self.ant_dx_m/self.wl             # Antenna spacing in wavelengths (lambda)
        self.ant_dy = self.ant_dy_m/self.wl

        if self.board=='rfsoc_2x2':
            self.adc_bits = 12
            self.dac_bits = 14
        elif self.board=='rfsoc_4x2':
            self.adc_bits = 14
            self.dac_bits = 14

        if self.tx_sig_sim=='same':
            self.seed_list = [self.seed for i in range(self.n_tx_ant)]
        elif self.tx_sig_sim=='orthogonal':
            self.seed_list = [self.seed*i+i for i in range(self.n_tx_ant)]
        elif self.tx_sig_sim=='shifted':
            self.seed_list = [self.seed for i in range(self.n_tx_ant)]

        self.server_ip = None
        self.steer_phi_rad = np.deg2rad(self.steer_phi_deg)
        self.steer_theta_rad = np.deg2rad(self.steer_theta_deg)
        self.n_samples_tx = self.n_frame_wr*self.n_samples
        self.n_samples_rx = self.n_frame_rd*self.n_samples
        self.n_samples_trx = min(self.n_samples_tx, self.n_samples_rx)
        self.nfft_tx = self.n_frame_wr*self.nfft
        self.nfft_rx = self.n_frame_rd*self.nfft
        self.nfft_trx = min(self.nfft_tx, self.nfft_rx)
        self.freq = np.linspace(-0.5, 0.5, self.nfft, endpoint=True) * self.fs / 1e6
        self.freq_tx = np.linspace(-0.5, 0.5, self.nfft_tx, endpoint=True) * self.fs_tx / 1e6
        self.freq_rx = np.linspace(-0.5, 0.5, self.nfft_rx, endpoint=True) * self.fs_rx / 1e6
        self.freq_trx = np.linspace(-0.5, 0.5, self.nfft_trx, endpoint=True) * self.fs_trx / 1e6

        self.beam_test = np.array([1, 5, 9, 13, 17, 21, 25, 29, 32, 35, 39, 43, 47, 51, 55, 59, 63])
        self.DynamicPLLConfig = (0, self.lmk_freq_mhz, self.lmx_freq_mhz)

        if self.tone_f_mode=='sc':
            self.f_tone = self.sc_tone * self.fs_tx/self.nfft_tx
        elif self.tone_f_mode=='freq':
            self.sc_tone = int(np.round((self.f_tone)*self.nfft_tx/self.fs_tx))
        else:
            raise ValueError('Invalid tone_f_mode mode: ' + self.tone_f_mode)
        
        if self.wb_bw_mode=='sc':
            self.wb_bw_range = [self.wb_sc_range[0]*self.fs_tx/self.nfft_tx, self.wb_sc_range[1]*self.fs_tx/self.nfft_tx]
        elif self.wb_bw_mode=='freq':
            self.wb_sc_range = [int(np.round(self.wb_bw_range[0]*self.nfft_tx/self.fs_tx)), int(np.round(self.wb_bw_range[1]*self.nfft_tx/self.fs_tx))]
        else:
            raise ValueError('Invalid wb_bw_mode mode: ' + self.tone_f_mode)

        if 'tone' in self.sig_mode:
            self.f_max = abs(self.f_tone)
            if self.sig_mode == 'tone_1':
                self.sc_range = [self.sc_tone, self.sc_tone]
                self.filter_bw_range = [self.f_tone-50e6, self.f_tone+50e6]
            elif self.sig_mode == 'tone_2':
                self.sc_range = [-1*self.sc_tone, self.sc_tone]
                self.filter_bw_range = [-1*self.f_tone-50e6, self.f_tone+50e6]
            self.null_sc_range = [0, 0]
        elif 'wideband' in self.sig_mode or self.sig_mode == 'load':
            self.f_max = max(abs(self.wb_bw_range[0]), abs(self.wb_bw_range[1]))
            self.sc_range = self.wb_sc_range
            self.filter_bw_range = [self.wb_bw_range[0]-50e6, self.wb_bw_range[1]+50e6]
            self.null_sc_range = [-1*self.wb_null_sc, self.wb_null_sc]
        else:
            raise ValueError('Unsupported signal mode: ' + self.sig_mode)
        
        if ('channel' in self.save_list):
            self.channel_limit = False
        if self.channel_limit:
            self.sc_range_ch = self.sc_range
            self.n_samples_ch = self.sc_range_ch[1] - self.sc_range_ch[0] + 1
            self.nfft_ch = self.n_samples_ch
            self.freq_ch = self.freq_trx[(self.sc_range_ch[0]+self.nfft_trx//2):(self.sc_range_ch[1]+self.nfft_trx//2+1)]
        else:
            self.sc_range_ch = [-1*self.nfft_trx//2, self.nfft_trx//2-1]
            self.n_samples_ch = self.n_samples_trx
            self.nfft_ch = self.nfft_trx
            self.freq_ch = self.freq_trx



        self.nf_n_rx_loc_sep = len(self.nf_rx_loc_sep)
        self.nf_n_ant_sep = len(self.nf_rx_ant_sep)
        self.nf_n_meas = self.nf_n_rx_loc_sep * self.nf_n_ant_sep
        p = len(self.nf_rx_sep_dir)
        # Generate the RX antenna positions
        self.nf_rx_ant_loc = np.zeros((self.n_rx_ant, self.nf_n_meas, p))
        self.nf_tx_ant_loc = np.zeros((self.n_tx_ant, self.nf_n_meas, p))
        for k in range(self.nf_n_rx_loc_sep):
            for i in range(self.nf_n_ant_sep):
                m = k*self.nf_n_ant_sep + i
                # Linear distance of the RX antennas from the origin
                t = self.nf_rx_loc_sep[k] + self.nf_rx_ant_sep[i]*np.arange(self.n_rx_ant)*self.wl
                # Position of the RX antennas
                self.nf_rx_ant_loc[:,m,:] = t[:,None]*self.nf_rx_sep_dir[None,:]

                t = self.ant_dx_m * np.arange(self.n_tx_ant)
                self.nf_tx_ant_loc[:,m,:] = self.nf_tx_loc + t[:,None]*self.nf_tx_sep_dir[None,:]

        if self.use_turntable:
            self.rotation_angles = np.arange(self.rotation_range_deg[0], self.rotation_range_deg[1]+self.rotation_step_deg, self.rotation_step_deg)
        else:
            self.rotation_angles = [0]

        for f in [self.calib_params_dir, self.sig_dir, self.channel_dir, self.figs_dir, self.params_dir]:
            if not os.path.exists(f):
                os.makedirs(f)


        if self.files_dwnld_target == 'rfsoc':
            self.files_to_download = ["*.py", "*.txt", "sigcom_toolkit/*.py"]
            # self.files_to_download.extend(["../vivado/sounder_fr3_if_ddr4_mimo_4x2/builds/project_v1-0-58_20241001-150336.bit", 
            #                 "../vivado/sounder_fr3_if_ddr4_mimo_4x2/builds/project_v1-0-58_20241001-150336.hwh"])
        elif self.files_dwnld_target == 'raspi':
            self.files_to_download = ["*.py", "*.txt", "sigcom_toolkit/*.py", "linear_track/*.py", "linear_track/*.txt"]
        
        if self.files_dwnld_target == 'rfsoc':
            self.params_to_modify = {"backend.py": {"import_pynq": True, "import_torch": False,
                                "import_sklearn": False, "import_cv2": False, "import_sivers": False, "import_adafruit": False}}  
        elif self.files_dwnld_target == 'raspi':
            self.params_to_modify = {"backend.py": {"import_pynq": False, "import_torch": False,
                            "import_sklearn": False, "import_cv2": False, "import_sivers": False, "import_adafruit": True}}
            
        if self.files_dwnld_target == 'rfsoc':
            self.files_to_convert = {"rfsoc_test.py": "rfsoc_test.ipynb"}

        


    def copy(self):
        return copy.deepcopy(self)
    






class Params_Class(Params_Class_Default):
    def __init__(self):
        super().__init__()

        self.init()
        self.populate_measurement_parameters()
        self.calc_params()


    
    def init(self):

        # parser = argparse.ArgumentParser()
        # parser.add_argument("--bit_file_path", type=str, default="./rfsoc.bit", help="Path to the bit file")
        # params = parser.parse_args()

        self.controller_slave_ip = '10.18.251.169'
        self.turntable_port = '/dev/ttyACM0'
        # self.turntable_port = 'COM4'

        # self.set_piradio_opt_gains = True
        self.set_piradio_opt_losupp = True
        self.piradio_freq_sw_dly_default = 0.1
        self.piradio_gain_sw_dly_default = 0.1
        self.piradio_bias_sw_dly_default = 0.1
        self.ant_dx_m = 0.02               # Antenna spacing in meters
        self.n_rx_ch_eq=1
        self.wb_sc_range=[-260,260]
        self.rx_same_delay=False
        # self.sparse_ch_samp_range=[-5,100]
        # self.sparse_ch_n_ignore=5
        self.n_frame_rd=32
        self.n_rd_rep=1
        self.anim_interval = 100
        self.save_parameters=True
        # self.load_parameters=True
        # self.plot_fonts_dict = {'title_size': 15, 'xaxis_size': 17, 'yaxis_size': 15, 'ticks_size': 15, 'legend_size': 15, 'line_width': 1.2, 'marker_size': 8, 'hspace': 0.4, 'wspace': 0.4}
        self.plot_fonts_dict = {'title_size': 11, 'title_max_chars': 35, 'xaxis_size': 10, 'yaxis_size': 10, 'ticks_size': 10, 'legend_size': 10, 'line_width': 1.0, 'marker_size': 8, 'hspace': 0.5, 'wspace': 0.5}
        # self.calibrate_turntable = True

        # self.overwrite_level=False
        # self.plot_level=0
        # self.verbose_level=3


        # self.update_rfsoc_files = True
        # self.host_files_base_addr = "/home/wirelesslab914/ali/sounder_rfsoc/RFSoC_SDR/python/"
        self.host_files_base_addr = "/Users/alira/OneDrive/Desktop/Current_works/Channel_sounding/RFSoC_SDR_copy/"
        self.host_ip = '192.168.2.1'
        # self.host_username = 'wirelesslab914'
        self.host_username = 'alira'
        self.host_password = ''



        # self.measurement_type = 'plot_saved_signal'
        self.measurement_type = 'RFSoC_demo_simple'
        # self.measurement_type = 'mmw_demo_simple'
        # self.measurement_type = 'FR3_demo_simple'
        # self.measurement_type = 'FR3_demo_multi_freq'
        # self.measurement_type = 'FR3_nyu_3state'
        # self.measurement_type = 'FR3_nyu_13state'
        # self.measurement_type = 'FR3_ant_calib'
        # self.measurement_type = 'FR3_beamforming'
        # self.measurement_type = 'FR3_cfo'
        # self.measurement_type = 'stream_to_matlab'

        self.mode = 'client'
        # self.mode = 'client_master'
        # self.mode = 'client_slave'






    def populate_measurement_parameters(self):

        if self.mode == 'client':
            self.send_signal=True
            self.rfsoc_server_ip='192.168.185.4'
        elif self.mode == 'client_master':
            self.send_signal=False
            # self.rfsoc_server_ip='192.168.2.99'
        elif self.mode == 'client_slave':
            self.send_signal=True
            # self.rfsoc_server_ip='192.168.2.99'


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
            self.control_rfsoc=False
            self.freq_hop_config['list'] = [6.5e9, 10e9, 15.0e9, 20.0e9]

            self.tx_sig_sim = 'shifted'
            self.sig_gen_mode = 'ZadoffChu'


        elif self.measurement_type == 'mmw_demo_simple':
            self.mode = 'client'
            self.RFFE='sivers'
            self.wb_sc_range=[-300,-100]
            self.send_signal=False
            self.recv_signal=True
            self.animate_plot_mode=[[h00], [rxtd00_r, rxtd00_i], [rxfd00]]
            self.rx_chain = ['sync_time', 'channel_est']
            # self.rx_chain = ['sync_time', 'channel_est', 'channel_eq']
            self.freq_hop_config['list'] = [60.0e9]
            # self.tx_sig_sim = 'orthogonal'
            # self.sig_gen_mode = 'ZadoffChu'



        elif self.measurement_type == 'RFSoC_demo_simple':
            # self.mix_freq=0e6
            # self.do_mixer_settings=True
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



        elif self.measurement_type == 'FR3_demo_simple':
            self.animate_plot_mode=[[h00], [rxtd00_r, rxtd00_i], [rxfd00]]
            self.rx_chain = ['sync_time', 'channel_est']
            self.control_piradio=True
            self.freq_hop_config['list'] = [6.5e9]
            self.tx_sig_sim = 'orthogonal'
            # self.sig_gen_mode = 'ZadoffChu'
            
            # self.save_list = ['signal']
            # self.measurement_configs = ["test"]
            # self.n_save = 256



        elif self.measurement_type == 'FR3_demo_multi_freq':
            self.animate_plot_mode=[[h00], [rxtd00_r, rxtd00_i], [rxfd00]]
            self.rx_chain = ['sync_time', 'channel_est']
            self.control_piradio=True
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

            self.control_piradio=True
            # self.freq_hop_config['list'] = [6.5e9, 10e9, 15.0e9, 20.0e9]
            # self.freq_hop_config['list'] = [10e9, 15.0e9]
            self.freq_hop_config['mode'] = 'sweep'
            self.freq_hop_config['range'] = [6.0e9, 22.5e9]
            # self.freq_hop_config['range'] = [20.0e9, 21.0e9]
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

            self.control_piradio=True
            self.freq_hop_config['list'] = [10e9]
            # self.freq_hop_config['mode'] = 'sweep'
            # self.freq_hop_config['range'] = [6.0e9, 22.5e9]
            # self.freq_hop_config['step'] = 0.5e9

            self.sig_gen_mode = 'fft'
            self.tx_sig_sim = 'same'

            self.beamforming=True
            self.steer_theta_deg = 0
            self.steer_phi_deg = 0

            self.save_list = ['signal']
            self.n_save = 32
            self.measurement_configs = [6.5]
            # self.measurement_configs.append("tx1_rx1_rx_rotate")
            self.measurement_configs.append("bf_phi_{}".format(self.steer_phi_deg))



        elif self.measurement_type == 'FR3_nyu_3state':
            self.animate_plot_mode=[[h00], [rxtd00_r, rxtd00_i], [rxfd00]]
            self.save_format = 'mat'
            self.rx_chain = ['sync_time', 'channel_est']
            self.use_turntable = True
            self.rotation_range_deg = [-45,45]
            self.rotation_step_deg = 45
            self.rotation_delay = 0.5
            self.control_piradio=True
            self.freq_hop_config['list'] = [6.5e9, 8.75e9, 10.0e9, 15.0e9, 21.7e9]
            # self.freq_hop_config['list'] = [10.0e9]
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

            self.control_piradio=True
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
            self.control_piradio=True
            
            self.freq_hop_config['list'] = [10.0e9]
            cfo_ppm = -100

            if self.mode == 'client_master':
                cfo = cfo_ppm * self.freq_hop_config['list'][0] / 1e6
                self.mix_freq += cfo
                self.do_mixer_settings=True

            # self.sig_gen_mode = 'ZadoffChu'
            # self.tx_sig_sim = 'shifted'
            self.sig_gen_mode = 'fft'
            self.tx_sig_sim = 'orthogonal'
            self.sig_modulation = '4qam'

            self.save_list = ['signal']
            self.n_save = 256
            self.measurement_configs = []
            self.measurement_configs.append("{}GHz_{}ppm".format(self.freq_hop_config['list'][0]/1e9, cfo_ppm))

        elif self.measurement_type == 'stream_to_matlab':
            self.animate_plot_mode=[[h00], [rxtd00_r, rxtd00_i], [rxfd00]]
            self.rx_chain = ['sync_time', 'channel_est']
            self.control_piradio = True
            self.freq_hop_config['list'] = [6.5e9]
     
            self.enable_matlab_stream = True
            self.sig_gen_mode = 'ZadoffChu'
            self.tx_sig_sim = 'shifted'

            
            # self.save_list = ['signal']
            # self.measurement_configs = ["test"]
            # self.n_save = 256   






