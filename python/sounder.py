import os
import platform
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import constants

from file_utils import File_Utils
from signal_utilsrfsoc import SignalUtilsRfsoc, SignalUtilsRFSoCConfig


@dataclass
class SounderConfig(SignalUtilsRFSoCConfig):
    # Constant parameters
    seed = 100

    # Board and RFSoC FPGA project parameters
    measurement_configs = []  # List of measurement configurations
    host_role = "client"  # Mode of operation, client or client_master or client_slave
    transmit_signal = False  # If True, sends TX signal

    # Plots and logs parameters
    overwrite_level = True  # If True, overwrites the plot and verbose levels
    plot_level = 0  # Level of plotting outputs
    verbose_level = 0  # Level of printing output
    animate_plot_mode = []  # List of plots to animate
    # dictionary of plot fonts configurations
    plot_configs = {
        "title_size": 15,
        "xaxis_size": 17,
        "yaxis_size": 15,
        "ticks_size": 15,
        "legend_size": 15,
        "line_width": 1.2,
        "marker_size": 8,
        "hspace": 0.4,
        "wspace": 0.4,
    }

    # Mixer parameters
    rfsoc_mixer_mode = "analog"  # Mixer mode, analog or digital

    # RFFE and antennas parameters
    RFFE = "piradio"  # RF front end to use, piradio or sivers
    n_tx_ant = 2  # Number of transmitter antennas
    n_rx_ant = 2  # Number of receiver antennas
    ant_d_m = [
        0.02
    ]  # Antenna axis spacing in meters, a list of spacing for each axis, for example [0.02, 0.02] for a 2D array with 2 cm spacing in both axes

    # Connections parameters
    network_topology = {
        "rfsoc_tx": {"type": "rfsoc", "role": "tx", "ip": "192.168.3.1", "protocol": "tcp"},
        "rfsoc_rx": {"type": "rfsoc", "role": "rx", "ip": "192.168.3.1", "protocol": "tcp"},
        "lintrack": {"type": "lintrack", "role": "rx", "ip": "192.168.137.100", "protocol": "tcp"},
        "turntable": {
            "type": "turntable",
            "role": "rx",
            "port": "/dev/ttyACM0",
            "baudrate": 115200,
            "protocol": "serial",
        },
        "piradio_tx": {
            "type": "piradio",
            "role": "tx",
            "ip": "192.168.137.51",
            "protocol": "http",
            "username": "ubuntu",
            "password": "temppwd",
        },
        "piradio_rx": {
            "type": "piradio",
            "role": "rx",
            "ip": "192.168.137.51",
            "protocol": "http",
            "username": "ubuntu",
            "password": "temppwd",
        },
        "controller_slave": {
            "type": "controller",
            "role": "tx",
            "ip": "192.168.1.1",
            "protocol": "tcp",
        },
        "host": {"type": "host", "role": "rx", "ip": "192.168.3.100", "protocol": "ssh"},
    }

    # File transfer parameters
    update_rfsoc_files = False  # If True, updates the RFSoC files
    modify_rfsoc_files = True  # If True, modifies the RFSoC files to be true for the server mode
    files_dwnld_target = "rfsoc"  # Target for file download, rfsoc or raspi
    host_files_base_addr = "~/RFSoC_SDR/python/"  # Base address for the host files
    host_ip = "192.168.3.100"  # Host IP address
    host_username = "root"  # Host username
    host_password = "root"  # Host password
    local_base_addr = "./"  # Local base address for the files

    # Signals information
    fs = 245.76e6 * 4  # Sampling frequency in RFSoC
    n_samples = 1024  # Number of samples
    sig_gen_mode = "fft"  # Signal generation mode, time, or fft or ofdm, or ZadoffChu
    sig_mode = "wideband_null"  # Signal mode, tone_1 or tone_2 or wideband or wideband_null or load
    sig_modulation = "4qam"  # Signal modulation type for sounding, 4qam, 16qam, etc
    tx_sig_sim = "same"  # TX signal similarity between antennas, same or orthogonal or shifted
    sig_gain_db = 0  # Transmitter Signal gain in dB
    n_frame_wr = 1  # Number of frames to write
    n_frame_rd = 2  # Number of frames to read
    n_rd_rep = 8  # Number of read repetitions for RX signal
    snr_est_db = 40  # SNR for signal estimation
    wb_bw_mode = "sc"  # Wideband signal bandwidth mode, sc or freq
    wb_sc_range = [-250, 250]  # Wideband signal subcarrier range, used when wb_bw_mode is sc
    wb_bw_range = [-250e6, 250e6]  # Wideband signal bandwidth range, used when wb_bw_mode is freq
    wb_null_sc = 0  # Number of carriers to null in the wideband signal
    tone_f_mode = "sc"  # Tone signal frequency mode, sc or freq
    sc_tone = 10  # Tone signal subcarrier
    f_tone = 250e6  # Tone signal frequency
    filter_bw_range = [-450e6, 450e6]  # Final filter BW range on the RX signal
    n_rx_ch_eq = 1  # Number of RX chains for channel equalization
    sparse_ch_samp_range = [
        -6,
        20,
    ]  # Range of samples around the strongest peak to consider for channel estimation
    sparse_ch_n_ignore = 5  # Number of samples to ignore around the strongest peak
    rx_same_delay = True  # If True, all applies the same time shift to all RX antennas
    rx_chain = [
        "sync_time",
        "channel_est",
    ]  # The chain of operations to perform on the RX signal, filter, integrate, sync_time, sync_time_frac, sync_freq, pilot_separate, sys_res_deconv, channel_est, sparse_est, channel_eq
    channel_limit = True  # If True, limits the channel to a specific range in the frequency domain
    npath_max = [
        20,
        5,
    ]  # 1st number is the maximum number to extract at the 1st round, 2nd number is the maximum number to extract at the 2nd round

    # Save parameters
    calib_config_dir = os.path.join(os.getcwd(), "calib/")  # Calibration parameters directory
    sig_dir = os.path.join(os.getcwd(), "sigs/")  # Signals directory
    channel_dir = os.path.join(os.getcwd(), "channels/")  # Channel directory
    figs_dir = os.path.join(os.getcwd(), "figs/")  # Figures directory
    config_dir = os.path.join(os.getcwd(), "config/")  # Configuration directory
    n_save = 100  # Number of samples to save
    save_format = "npz"  # Format to save the data, npz or mat (for MATLAB)
    save_parameters = False  # If True, saves current parameters
    load_parameters = False  # If True, loads parameters from the file

    # Calibration parameters
    calib_iter = 100  # Number of iterations for calibration

    # Beamforming parameters
    beamforming = False  # If True, performs beamforming
    steer_rad = [
        np.deg2rad(0.0),
        np.deg2rad(0.0),
    ]  # Desired steering angles in radians [azimuth, elevation]

    # Action parameters
    action_loop = []

    def detect_running_platform(self):
        system_info = platform.uname()
        if "pynq" in system_info.node.lower():
            self.running_platform = "rfsoc"
        else:
            self.running_platform = "host"

    def __post_init__(self):
        super().__post_init__()

        self.detect_running_platform()

        self.calib_config_path = os.path.join(
            self.calib_config_dir, "calib_config.npz"
        )  # Calibration parameters path
        self.optimal_gains_path = os.path.join(
            self.calib_config_dir, "optimal_gains.json"
        )  # Calibration parameters path
        self.sig_path = os.path.join(self.sig_dir, "txtd.npz")  # Signal load path
        self.sys_response_path = os.path.join(
            self.channel_dir, "sys_response.npz"
        )  # System response save path
        self.figs_save_path = os.path.join(self.figs_dir, "plot.pdf")  # Figures save path
        self.config_path = os.path.join(self.config_dir, "config.json")  # Configuration load path
        self.config_save_path = os.path.join(
            self.config_dir, "config.json"
        )  # Configuration save path

        self.n_samples_tx = self.n_frame_wr * self.n_samples
        self.n_samples_rx = self.n_frame_rd * self.n_samples
        self.nfft_tx = self.n_frame_wr * self.nfft
        self.nfft_rx = self.n_frame_rd * self.nfft

        if self.overwrite_level:
            if self.running_platform == "rfsoc":
                self.plot_level = 4
                self.verbose_level = 4
            elif "slave" in self.host_role:
                self.plot_level = 0
                self.verbose_level = 4
            else:
                self.plot_level = 0
                self.verbose_level = 1

        if self.n_tx_ant == 1 and self.n_rx_ant == 1:
            self.beamforming = False

        self.fc = self.freq_hop_list[0]
        self.wl = constants.c / self.fc
        self.ant_d = [
            d / self.wl for d in self.ant_d_m
        ]  # Antenna axis spacing in wavelengths (lambda)

        if self.tx_sig_sim == "same":
            self.seed_list = [self.seed for i in range(self.n_tx_ant)]
        elif self.tx_sig_sim == "orthogonal":
            self.seed_list = [self.seed * i + i for i in range(self.n_tx_ant)]
        elif self.tx_sig_sim == "shifted":
            self.seed_list = [self.seed for i in range(self.n_tx_ant)]

        if self.tone_f_mode == "sc":
            self.f_tone = self.sc_tone * self.fs_tx / self.nfft_tx
        elif self.tone_f_mode == "freq":
            self.sc_tone = int(np.round((self.f_tone) * self.nfft_tx / self.fs_tx))
        else:
            raise ValueError("Invalid tone_f_mode mode: " + self.tone_f_mode)

        if self.wb_bw_mode == "sc":
            self.wb_bw_range = [
                self.wb_sc_range[0] * self.fs_tx / self.nfft_tx,
                self.wb_sc_range[1] * self.fs_tx / self.nfft_tx,
            ]
        elif self.wb_bw_mode == "freq":
            self.wb_sc_range = [
                int(np.round(self.wb_bw_range[0] * self.nfft_tx / self.fs_tx)),
                int(np.round(self.wb_bw_range[1] * self.nfft_tx / self.fs_tx)),
            ]
        else:
            raise ValueError("Invalid wb_bw_mode mode: " + self.tone_f_mode)

        if "tone" in self.sig_mode:
            self.f_max = abs(self.f_tone)
            if self.sig_mode == "tone_1":
                self.sc_range = [self.sc_tone, self.sc_tone]
                self.filter_bw_range = [self.f_tone - 50e6, self.f_tone + 50e6]
            elif self.sig_mode == "tone_2":
                self.sc_range = [-1 * self.sc_tone, self.sc_tone]
                self.filter_bw_range = [-1 * self.f_tone - 50e6, self.f_tone + 50e6]
            self.null_sc_range = [0, 0]
        elif "wideband" in self.sig_mode or self.sig_mode == "load":
            self.f_max = max(abs(self.wb_bw_range[0]), abs(self.wb_bw_range[1]))
            self.sc_range = self.wb_sc_range
            self.filter_bw_range = [self.wb_bw_range[0] - 50e6, self.wb_bw_range[1] + 50e6]
            self.null_sc_range = [-1 * self.wb_null_sc, self.wb_null_sc]
        else:
            raise ValueError("Unsupported signal mode: " + self.sig_mode)

        if self.channel_limit:
            self.sc_range_ch = self.sc_range
            self.n_samples_ch = self.sc_range_ch[1] - self.sc_range_ch[0] + 1
            self.nfft_ch = self.n_samples_ch
            self.freq_ch = self.freq_trx[
                (self.sc_range_ch[0] + self.nfft_trx // 2) : (
                    self.sc_range_ch[1] + self.nfft_trx // 2 + 1
                )
            ]
        else:
            self.sc_range_ch = [-1 * self.nfft_trx // 2, self.nfft_trx // 2 - 1]
            self.n_samples_ch = self.n_samples_trx
            self.nfft_ch = self.nfft_trx
            self.freq_ch = self.freq_trx

        if self.files_dwnld_target == "rfsoc":
            self.files_to_download = ["*.py", "*.txt", "sigcom_toolkit/*.py"]
            # self.files_to_download.extend(["../vivado/sounder_fr3_if_ddr4_mimo_4x2/builds/project_v1-0-58_20241001-150336.bit",
            #                 "../vivado/sounder_fr3_if_ddr4_mimo_4x2/builds/project_v1-0-58_20241001-150336.hwh"])
        elif self.files_dwnld_target == "raspi":
            self.files_to_download = [
                "*.py",
                "*.txt",
                "sigcom_toolkit/*.py",
                "linear_track/*.py",
                "linear_track/*.txt",
            ]

        if self.files_dwnld_target == "rfsoc" or self.files_dwnld_target == "raspi":
            self.configs_to_modify = {}

        if self.files_dwnld_target == "rfsoc":
            self.files_to_convert = {"sounder.py": "sounder.ipynb"}


class Sounder(SounderConfig):
    def __init__(self, config: SounderConfig, **overrides: Any):
        super().__init__(config, **overrides)

        if self.config.save_parameters:
            self.save_class_attributes_to_json(self.config, self.config.config_save_path)
            self.config.save_parameters = False
        if self.config.load_parameters:
            self.load_class_attributes_from_json(self.config, self.config.config_path)

        self.create_dirs(
            [
                self.config.calib_config_dir,
                self.config.sig_dir,
                self.config.channel_dir,
                self.config.figs_dir,
                self.config.config_dir,
            ]
        )
        self.print(f"Running the code as {self.config.host_role}", thr=1)

        if self.config.running_platform == "rfsoc" and (
            self.config.update_rfsoc_files or self.config.modify_rfsoc_files
        ):
            self.update_rfsoc_files()

        self.signals_inst = SignalUtilsRfsoc(self.config)
        self.signals_inst.gen_tx_signal()

    def update_rfsoc_files(self):
        file_utils = File_Utils(self.config, scp_connect=self.config.update_rfsoc_files)
        changed_1 = False
        changed_2 = False
        changed_3 = False

        if self.config.update_rfsoc_files:
            changed_1 = file_utils.download_files()
        if self.config.update_rfsoc_files or self.config.modify_rfsoc_files:
            changed_2 = file_utils.modify_files()
        if self.config.update_rfsoc_files:
            changed_3 = file_utils.convert_files()

        if changed_1:
            print("Some files were updated from the Host server ...")
        if changed_2:
            print("To handle pre-requisites some files were modified ...")
        if changed_3:
            print("Some files were converted ...")
        if changed_1 or changed_2 or changed_3:
            print("Please run the script again ...")
            return

    def run_rfsoc(self):
        from rfsoc import RFSoC

        rfsoc_inst = RFSoC(self.config)
        rfsoc_inst.txtd = self.signals_inst.txtd
        if self.config.transmit_signal:
            rfsoc_inst.send_frame(self.signals_inst.txtd)

        # Receiving a test frame to verify connection
        rfsoc_inst.recv_frame_one(n_frame=self.config.n_frame_rd)
        rfsoc_inst.run_tcp()

    def run(self):

        if self.config.running_platform == "rfsoc":
            self.run_rfsoc()

        if "client" in self.config.host_role:
            self.signals_inst.operator()
