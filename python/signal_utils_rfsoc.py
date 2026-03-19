import ast
import contextlib
import itertools
import os
import time
from dataclasses import dataclass
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from lin_track_cntrl import LinearTrackController, LinearTrackControllerConfig
from matplotlib import cycler  # type: ignore
from numpy.fft import fft, fftshift, ifft, ifftshift
from scipy import constants
from serial_comm import (
    SerialComD48PTU,
    SerialComD48PTUConfig,
    SerialComTurnTable,
    SerialComTurnTableConfig,
)
from sigcom_toolkit.general import General, GeneralConfig
from sigcom_toolkit.plot_utils import PlotUtils, PlotUtilsConfig
from sigcom_toolkit.signal_utils import SignalUtils, SignalUtilsConfig
from sigcom_toolkit.specsense_utils import SpecSenseUtils
from sigcom_toolkit.utils import get_viewing_angle_range
from tcp_comm import (
    RESTComPiradio,
    RestComPiradioConfig,
    TCPComControllerConfig,
    TCPComLinTrackConfig,
    TcpCommController,
    TcpCommLinTrack,
    TcpCommRFSoC,
    TCPComRFSoCConfig,
)


@dataclass(kw_only=True)
class PlotChart:
    plot_signals: dict = None
    title: str = ""
    x_label: str = ""
    y_label: str = ""


@dataclass(kw_only=True)
class PlotSignal:
    signal_name: str = ""
    trx_id: tuple = None
    process_list: tuple = None
    x: np.ndarray = None
    data: np.ndarray = None
    label: str = ""


@dataclass(kw_only=True)
class SparseEstParams:
    h_tr_mat: np.ndarray = None
    dly_est_mat: np.ndarray = None
    peaks_mat: np.ndarray = None
    npaths_est_mat: np.ndarray = None


@dataclass(kw_only=True)
class RxSignal:
    # Shape: [n_frame_rd, n_rx_ant, n_samples_rx]
    rxtd: np.ndarray = None
    # Shape: [n_frame_rd, n_rx_ant, n_samples_rx]
    rxtd_base: np.ndarray = None
    # Shape: [n_frame_rd, n_rx_ant, n_tx_ant, n_samples_ch]
    h_est: np.ndarray = None
    sparse_est_params: SparseEstParams = None


@dataclass(kw_only=True)
class TxSignal:
    # Shape: [n_frame_rd, n_tx_ant, n_samples_tx]
    txtd: np.ndarray = None
    # Shape: [n_frame_rd, n_tx_ant, n_samples_tx]
    txtd_base: np.ndarray = None


@dataclass(kw_only=True)
class ClientRFSoCConfig(TCPComRFSoCConfig):
    calib_config_dir: str = os.path.join(os.getcwd(), "calib/")
    calib_iter: int = 100  # Number of iterations for calibration

    def __post_init__(self):
        super().__post_init__()
        self.calib_config_path: str = os.path.join(self.calib_config_dir, "calib_config.npz")


class ClientRFSoC(TcpCommRFSoC):
    def __init__(self, config: ClientRFSoCConfig, **overrides: Any):
        super().__init__(config, **overrides)

        self.rx_phase_offset = 0
        self.rx_delay_offset = 0

    def calibrate_rx_phase_offset(self):
        """
        This function calibrates the phase offset between the receivers ports in RFSoCs
        """
        input_ = input(
            "Press Y for phase offset calibration (and position the TX/RX at AoA = 0) or any key to use the saved phase offset: "
        )

        if input_.lower() != "y":
            if os.path.exists(self.config.calib_config_path):
                self.rx_phase_offset = np.load(self.config.calib_config_path)["rx_phase_offset"]
                self.rx_delay_offset = np.load(self.config.calib_config_path)["rx_delay_offset"]
                self.print(
                    f"Using saved phase offset between RX ports: {self.rx_phase_offset:0.3f} Rad",
                    thr=1,
                )
                # self.print("Using saved delay offset between RX ports: {:0.3f} s".format(self.rx_delay_offset), thr=1)
            else:
                self.print("No saved calibration found, please calibrate the phase offset", thr=0)
                self.rx_phase_offset = 0
                self.rx_delay_offset = 0
            return
        else:
            phase_diff_list = []
            delay_list = []
            for _ in range(self.config.calib_iter):
                rxtd = self.receive_data(mode="once")
                phase_diff = SignalUtils.calc_phase_offset(rxtd[0, 0, :], rxtd[0, 1, :])
                delay = phase_diff / (2 * np.pi * self.config.fc)
                phase_diff_list.append(phase_diff)
                delay_list.append(delay)

            self.rx_phase_offset = np.mean(phase_diff_list)
            self.rx_delay_offset = np.mean(delay_list)
            np.savez(
                self.config.calib_config_path,
                rx_phase_offset=self.rx_phase_offset,
                rx_delay_offset=self.rx_delay_offset,
                fc=self.config.fc,
            )
            self.print(
                f"Calibrated and saved phase offset between RX ports: {self.rx_phase_offset:0.3f} Rad",
                thr=1,
            )
            # self.print(f"Calibrated and saved delay offset between RX ports: {self.rx_delay_offset:0.3f} s", thr=1)


@dataclass(kw_only=True)
class PiRadioConfig(RestComPiradioConfig):
    calib_config_dir: str = os.path.join(os.getcwd(), "calib/")  # Calibration parameters directory
    stable_fc_piradio: float = 10.0e9
    piradio_gain_sw_dly_default: float = 0.1
    freq_range: tuple = (6.0, 22.5)

    def __post_init__(self):
        super().__post_init__()
        self.optimal_gains_path: str = os.path.join(self.calib_config_dir, "optimal_gains.json")


class PiRadioFR3Trx(RESTComPiradio):
    def __init__(self, config: PiRadioConfig, **overrides: Any):
        super().__init__(config, **overrides)

        self.fc = None

        if os.path.exists(self.config.optimal_gains_path):
            self.optimal_gains = self.load_dict_from_json(
                self.config.optimal_gains_path, convert_values=True
            )
        else:
            self.optimal_gains = {}

    def hop_freq(self, fc, set_opt_losupp=False):
        if self.fc is None or self.fc != fc:
            self.set_frequency(fc=fc)
            if set_opt_losupp:
                self.set_optimal_losupp(fc=fc)
            self.fc = fc
            self.wl = constants.c / self.fc

    def set_optimal_losupp(self, fc=None):
        if fc is None:
            fc = self.fc

        self.print("Setting optimal LO suppression for TX and RX in Pi-Radio", thr=1)

        lo_supp_lut = {
            6.5: [-0.026, -0.021],
            7.5: [-0.025, -0.016],
            8.5: [-0.001, -0.036],
            9.5: [0.078, -0.045],
            10.5: [0.192, -0.146],
            11.5: [0.113, -0.08],
            12.5: [0.055, -0.03],
            13.5: [0.04, 0.008],
            14.5: [0.016, -0.002],
            15.5: [-0.002, -0.022],
            16.5: [0.004, -0.065],
            17.5: [0.034, -0.065],
            18.5: [0.049, -0.005],
            19.5: [0.075, 0.003],
            20.5: [0.116, 0.049],
            21.5: [0.07, 0.027],
            22.5: [-0.025, -0.027],
        }

        nearest_fc = min(lo_supp_lut.keys(), key=lambda x: abs(x - fc / 1e9))
        optimal_lo_supp = lo_supp_lut[nearest_fc]

        self.print(
            f"Nearest frequency: {nearest_fc} GHz, Optimal LO suppression: {optimal_lo_supp}",
            thr=1,
        )
        self.set_bias(chan=0, iq="I", bias_voltage=optimal_lo_supp[0])
        self.set_bias(chan=0, iq="Q", bias_voltage=optimal_lo_supp[1])
        self.set_bias(chan=1, iq="I", bias_voltage=optimal_lo_supp[0])
        self.set_bias(chan=1, iq="Q", bias_voltage=optimal_lo_supp[1])

    def set_optimal_gain(self, side="both", tx_rx_distance=3.0):
        self.print("Setting optimal TX/RX gains in Pi-Radio", thr=0)

        freq_list = list(self.optimal_gains[tx_rx_distance].keys())
        nearest_fc = min(freq_list, key=lambda x: abs(x - self.config.stable_fc_piradio / 1e9))

        if side == "rx" or side == "both":
            rx_gain_optimal = self.optimal_gains[tx_rx_distance][nearest_fc]["rx_gain"]
            self.set_gain(trx="rx", chan=0, gain_db=rx_gain_optimal)
            self.set_gain(trx="rx", chan=1, gain_db=rx_gain_optimal)
        if side == "tx" or side == "both":
            tx_gain_optimal = self.optimal_gains[tx_rx_distance][nearest_fc]["tx_gain"]
            self.set_gain(trx="tx", chan=0, gain_db=tx_gain_optimal)
            self.set_gain(trx="tx", chan=1, gain_db=tx_gain_optimal)


@dataclass(kw_only=True)
class TurtlebotConfig(GeneralConfig):
    cmd_topic: str = ("/cmd_vel_unstamped",)
    odom_topic: str = ("/odom",)
    rate: float = (10.0,)
    max_linear: float = (0.50,)
    max_angular: float = (0.25,)
    target_frame: str = ("map",)
    source_frame: str = ("base_link",)
    tf_timeout: float = (20.0,)
    lin_accel_limit: float = (0.05,)
    ang_accel_limit: float = (0.8,)


class Turtlebot(General):
    def __init__(self, config: TurtlebotConfig, **overrides: Any):
        super().__init__(config, **overrides)

        # from turtlebot.map_motion_api import MapMotionAPI
        # self.map_motion_api = MapMotionAPI(
        #     cmd_topic=self.config.cmd_topic,
        #     odom_topic=self.config.odom_topic,
        #     rate=self.config.rate,
        #     max_linear=self.config.max_linear,
        #     max_angular=self.config.max_angular,
        #     target_frame=self.config.target_frame,
        #     source_frame=self.config.source_frame,
        #     tf_timeout=self.config.tf_timeout,
        #     lin_accel_limit=self.config.lin_accel_limit,
        #     ang_accel_limit=self.config.ang_accel_limit,
        # )

        self.init()
        exit()

    def close(self):
        self.map_motion_api.shutdown()

    def __del__(self):
        self.close()

    def move_to(self, position):
        api = self.map_motion_api
        cur_x, cur_y, yaw = api.read_pos()
        mv_yaw, mv_dis = api.compute_yaw_distance_to_target([cur_x, cur_y], position)
        api.move(yaw=mv_yaw, distance=mv_dis)
        self.turtlebot_pos = position

    def rotate_to(self, position):
        api = self.map_motion_api
        cur_x, cur_y, yaw = api.read_pos()
        mv_yaw, _ = api.compute_yaw_distance_to_target([cur_x, cur_y], position)
        api.move(yaw=mv_yaw, distance=0.0)
        self.turtlebot_orientation = mv_yaw

    def init(self):
        # Origin is the point that turtlebot is powered on on the corner of the room
        lintrack_length = 1.2
        # Moving room size in meters [length, width]
        self.moving_room_size = [2.0, -3.0]
        # Grid size for the moving room in meters [length, width]
        moving_room_grid_size = [0.2, -0.2]
        # Offset of the linear track from the origin point in meters [length, width]
        self.lintrack_offset = np.array([1.0, 1.0])
        # Tilt of the linear track in degrees
        self.lintrack_tilt_deg = 180.0 + 10.0
        # Grid size for the linear track in meters
        lintrack_grid_size = 0.05
        # Offset of the gimbal azimuth angle in degrees
        # To compensate for the linear track tilt and point the gimbal towards the center of the room
        self.gimbal_az_offset_deg = -45.0 + (self.lintrack_tilt_deg - 180.0)
        # Grid size for the gimbal azimuth angles in degrees
        self.gimbal_az_grid_size_deg = 5.0
        # TX beam width in degrees, used to limit the gimbal angles range
        self.tx_beam_width_deg = 60.0
        # Height difference between the TX and RX in meters, used to calculate the gimbal elevation angle
        self.tx_rx_height_diff = -0.5

        self.moving_room_grid = np.mgrid[
            0 : self.moving_room_size[0] : moving_room_grid_size[0],
            0 : self.moving_room_size[1] : moving_room_grid_size[1],
        ].reshape(2, -1).T

        # Sort the grid points in a snake-like pattern to minimize the movement distance of the turtlebot
        def zigzag_sort(arr):
            ys = np.unique(arr[:, 1])
            ys.sort()
            rows = []
            for i, y in enumerate(ys):
                row = arr[arr[:, 1] == y]
                row = row[np.argsort(row[:, 0])]
                if i % 2 == 0:
                    row = row[::-1]
                rows.append(row)
            return np.vstack(rows)

        self.moving_room_grid = zigzag_sort(self.moving_room_grid)
        self.lintrack_grid = np.linspace(0, lintrack_length,\
                            int(lintrack_length / lintrack_grid_size))

        self.turtlebot_pos = None
        self.tx_pos = None
        self.lintrack_grid_id = 0
        self.gimbal_az_grid = None
        self.gimbal_az_grid_id = 0
        self.turtlebot_orientation = None
        self.tx_orientation = None

    def get_next_turtlebot_position(self):
        # This function should return the next position of the turtlebot in the room grid
        for pos in self.moving_room_grid:
            self.reset_lintrack_position()
            yield pos

    def reset_lintrack_position(self):
        self.reset_gimbal_position()
        self.lintrack_grid_id = 0

    def get_next_lintrack_position(self):
        # This function should return the next position of the linear track
        if self.lintrack_grid_id >= len(self.lintrack_grid):
            raise StopIteration("No more linear track positions available")
        pos = self.lintrack_grid[self.lintrack_grid_id]
        self.tx_pos = self.lintrack_offset + pos * np.array([
                    np.cos(np.deg2rad(self.lintrack_tilt_deg)),
                    np.sin(np.deg2rad(self.lintrack_tilt_deg))])
        self.lintrack_grid_id += 1
        self.reset_gimbal_position()
        return pos

    def reset_gimbal_position(self):
        min_angle, max_angle, exact_angle = get_viewing_angle_range(
            ref_x=self.tx_pos[0],
            ref_y=self.tx_pos[1],
            obj_x=self.turtlebot_pos[0],
            obj_y=self.turtlebot_pos[1],
            alpha_deg=self.tx_beam_width_deg / 2,
        )
        # self.gimbal_az_grid = np.linspace(min_angle, max_angle,
        #                     int((max_angle - min_angle) / self.gimbal_az_grid_size_deg))  # Gimbal azimuth angles grid in degrees
        self.gimbal_az_grid = np.array([exact_angle-self.gimbal_az_offset_deg])  # Only use the exact angle for the gimbal azimuth
        self.gimbal_az_grid_id = 0

    def get_next_gimbal_angle(self):
        # This function should return the next angle of the gimbal
        if self.gimbal_az_grid_id >= len(self.gimbal_az_grid):
            raise StopIteration("No more gimbal angles available")
        az = self.gimbal_az_grid[self.gimbal_az_grid_id]
        tx_rx_dist = np.linalg.norm(self.tx_pos - self.turtlebot_pos)
        el = np.arctan2(self.tx_rx_height_diff, tx_rx_dist)
        self.tx_orientation = [az+self.gimbal_az_offset_deg, el]
        self.gimbal_az_grid_id += 1
        return (az, el)


@dataclass(kw_only=True)
class SignalUtilsRFSoCConfig(SignalUtilsConfig):
    freq_hop_list: tuple = (10.0e9,)
    seed: int = None
    seed_list: tuple = None
    ant_d_m: tuple = (0.026,)
    n_tx_ant: int = 2
    n_rx_ant: int = 2

    # Mixer parameters
    rfsoc_mixer_mode: str = "analog"  # Mixer mode, analog or digital

    # Signals information
    sig_gen_mode: str = "fft"  # Signal generation mode, time, or fft or ofdm, or ZadoffChu
    sig_mode: str = (
        "wideband_null"  # Signal mode, tone_1 or tone_2 or wideband or wideband_null or load
    )
    sig_modulation: str = "4qam"  # Signal modulation type for sounding, 4qam, 16qam, etc
    tx_sig_sim: str = "same"  # TX signal similarity between antennas, same or orthogonal or shifted
    sig_gain_db: float = 0  # Transmitter Signal gain in dB
    n_frame_wr: int = 1  # Number of frames to write
    n_frame_rd: int = 2  # Number of frames to read
    snr_est_db: float = 40  # SNR for signal estimation
    wb_bw_mode: str = "sc"  # Wideband signal bandwidth mode, sc or freq
    wb_sc_range: tuple = (-250, 250)  # Wideband signal subcarrier range, used when wb_bw_mode is sc
    wb_bw_range: tuple = (
        -250e6,
        250e6,
    )  # Wideband signal bandwidth range, used when wb_bw_mode is freq
    wb_null_sc: int = 0  # Number of carriers to null in the wideband signal
    tone_f_mode: str = "sc"  # Tone signal frequency mode, sc or freq
    sc_tone: int = 10  # Tone signal subcarrier
    f_tone: float = 250e6  # Tone signal frequency
    filter_bw_range: tuple = (-450e6, 450e6)  # Final filter BW range on the RX signal
    n_rx_ch_eq: int = 1  # Number of RX chains for channel equalization
    sparse_ch_samp_range: tuple = (
        -6,
        20,
    )  # Range of samples around the strongest peak to consider for channel estimation
    sparse_ch_n_ignore: int = 5  # Number of samples to ignore around the strongest peak
    rx_same_delay: bool = True  # If True, all applies the same time shift to all RX antennas
    rx_chain: tuple = (
        "sync_time",
        "channel_est",
    )  # The chain of operations to perform on the RX signal, filter, integrate, sync_time, sync_time_frac, sync_freq, pilot_separate, sys_res_deconv, channel_est, estimate_sparse_params, channel_eq
    channel_limit: bool = (
        True  # If True, limits the channel to a specific range in the frequency domain
    )
    npath_max: tuple = (
        20,
        5,
    )  # 1st number is the maximum number to extract at the 1st round, 2nd number is the maximum number to extract at the 2nd round

    # Save parameters
    calib_config_dir: str = os.path.join(os.getcwd(), "calib/")  # Calibration parameters directory
    sig_dir: str = os.path.join(os.getcwd(), "sigs/")  # Signals directory
    channel_dir: str = os.path.join(os.getcwd(), "channels/")  # Channel directory
    figs_dir: str = os.path.join(os.getcwd(), "figs/")  # Figures directory

    save_format: str = "npz"  # Format to save the data, npz or mat (for MATLAB)

    # Beamforming parameters
    beamforming: bool = False  # If True, performs beamforming
    steer_rad: tuple = (
        np.deg2rad(0.0),
        np.deg2rad(0.0),
    )  # Desired steering angles in radians [azimuth, elevation]

    def __post_init__(self):
        super().__post_init__()

        self.optimal_gains_path = os.path.join(
            self.calib_config_dir, "optimal_gains.json"
        )  # Calibration parameters path
        self.sig_path = os.path.join(self.sig_dir, "txtd.npz")  # Signal load path
        self.sys_response_path = os.path.join(
            self.channel_dir, "sys_response.npz"
        )  # System response save path
        self.figs_save_path = os.path.join(self.figs_dir, "plot.pdf")  # Figures save path

        self.n_samples_tx = self.n_frame_wr * self.n_samples
        self.n_samples_rx = self.n_frame_rd * self.n_samples
        self.nfft_tx = self.n_frame_wr * self.nfft
        self.nfft_rx = self.n_frame_rd * self.nfft

        if self.n_tx_ant == 1 and self.n_rx_ant == 1:
            self.beamforming = False

        self.fc = 10.0e9  # Carrier frequency in Hz
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


class SignalUtilsRfsoc(SignalUtils):
    def __init__(self, config: SignalUtilsRFSoCConfig, **overrides):
        super().__init__(config, **overrides)

        self.rx_phase_list = []
        self.aoa_list = []
        self.tx_rx_distance = 3.0
        self.tx_signal = None
        self.rx_signal = None

    def gen_tx_signal(self):
        self.print("Generating TX signal", thr=3)

        txtd_base = []
        txtd = []
        for ant_id in range(self.config.n_tx_ant):
            if "tone" in self.config.sig_mode:
                if self.config.sig_mode == "tone_1":
                    nsc = 1
                elif self.config.sig_mode == "tone_2":
                    nsc = 2
                txtd_base_s = self.generate_tone(
                    freq_mode=self.config.tone_f_mode,
                    sc=self.config.sc_tone,
                    f=self.config.f_tone,
                    sig_mode=self.config.sig_mode,
                    gen_mode=self.config.sig_gen_mode,
                )
            elif "wideband" in self.config.sig_mode:
                nsc = self.config.wb_sc_range[1] - self.config.wb_sc_range[0] + 1
                txtd_base_s = self.generate_wideband(
                    bw_mode=self.config.wb_bw_mode,
                    sc_range=self.config.wb_sc_range,
                    bw_range=self.config.wb_bw_range,
                    modulation=self.config.sig_modulation,
                    sig_mode=self.config.sig_mode,
                    gen_mode=self.config.sig_gen_mode,
                    seed=self.config.seed_list[ant_id],
                )
            elif self.config.sig_mode == "load":
                txtd_base_s = np.load(self.config.sig_path)
            else:
                raise ValueError("Unsupported signal mode: " + self.config.sig_mode)
            txtd_base_s /= np.max([np.abs(txtd_base_s.real), np.abs(txtd_base_s.imag)])
            txtd_base_s *= self.db_to_lin(self.config.sig_gain_db, mode="mag")
            txtd_base.append(txtd_base_s)

            self.config.sig_pow_dbm = (
                self.lin_to_db(0.5 * 1000, mode="pow") + self.config.sig_gain_db
            )
            bw = (nsc / self.config.nfft_tx) * self.config.fs_tx
            self.config.sig_psd_dbm = self.config.sig_pow_dbm - self.lin_to_db(bw, mode="pow")
            self.config.sig_psd_dbm_sc = self.config.sig_pow_dbm - self.lin_to_db(nsc, mode="pow")
            self.print(
                f"TX Signal power for antenna {ant_id}: {self.config.sig_pow_dbm:0.3f} dbm",
                thr=4,
            )
            self.print(
                "TX Signal PSD for antenna {}: {:0.3f} dBm/Hz = {:0.3f} dBm/MHz = {:0.3f} dBm/sc".format(
                    ant_id,
                    self.config.sig_psd_dbm,
                    self.config.sig_psd_dbm + self.lin_to_db(1e6, mode="pow"),
                    self.config.sig_psd_dbm_sc,
                ),
                thr=4,
            )

        txtd_base = np.array(txtd_base)

        if self.config.tx_sig_sim == "shifted":
            if self.config.n_tx_ant < 2:
                raise ValueError("tx_sig_sim='shifted' requires at least two TX antennas")
            txtd_base[1, :] = np.roll(txtd_base[0, :], shift=(384), axis=-1)

        if self.config.rfsoc_mixer_mode == "digital" and self.config.mix_freq_dac != 0:
            for ant_id in range(self.config.n_tx_ant):
                txtd_s = self.shift_freq(
                    txtd_base[ant_id], shift=self.config.mix_freq_dac, fs=self.config.fs_tx
                )
                txtd.append(txtd_s)
        else:
            txtd = txtd_base.copy()

        txtd = np.array(txtd)
        txtd_base = np.expand_dims(txtd_base, axis=0)
        txtd = np.expand_dims(txtd, axis=0)

        if self.config.beamforming:
            txtd_base = self.beam_form(txtd_base)
            txtd = self.beam_form(txtd)

        if self.config.n_tx_ant > 1:
            self.print(
                f"Dot product of transmitted signals: {np.abs(np.vdot(txtd_base[0, 1], txtd_base[0, 0]))}",
                thr=4,
            )
        # self.plotter.plot_signal(sigs = np.abs(np.correlate(txtd_base[0, 1], txtd_base[0, 0], mode='full')))

        self.tx_signal = TxSignal(txtd=txtd, txtd_base=txtd_base)
        return self.tx_signal

    def validate_saved_signals(self, rxtd, txtd=None, thr=1e-8):
        self.print("Sanity check for saved signals", thr=2)

        mses = []
        n_frames = rxtd.shape[0]
        for i in range(n_frames):
            mse = self.mse(rxtd[i, 0], rxtd[i, 1])
            mses.append(mse)
            mse = self.mse(rxtd[i - n_frames // self.config.n_frame_rd, 0], rxtd[i, 0])
            mses.append(mse)
            mse = self.mse(rxtd[i - n_frames // self.config.n_frame_rd, 1], rxtd[i, 1])
            mses.append(mse)
            mse = self.mse(rxtd[i - 1, 0], rxtd[i, 0])
            mses.append(mse)
            mse = self.mse(rxtd[i - 1, 1], rxtd[i, 1])
            mses.append(mse)

        if np.min(mses) < thr:
            self.print("RX signals are not saved correctly", thr=0)
            raise ValueError("RX signals are not saved correctly")

        if txtd is not None:
            offset = np.argmax(np.abs(txtd[0, 0])) - np.argmax(np.abs(txtd[0, 1]))
            self.print(f"Offset between TX signals: {offset}", thr=0)

        self.print("Sanity check passed", thr=3)

    def process_sys_response(self):
        self.print("Processing system response", thr=5)
        self.sys_response = np.load(self.config.sys_response_path)["h_est_full_avg"]
        self.sys_response /= np.max(np.abs(self.sys_response))
        return self.sys_response

    def compute_sys_response(self):
        self.print("Computing system response", thr=5)
        n_rx = 1

        if n_rx == 1:
            sys_response_folder = os.path.join(os.getcwd(), "sigs_tx1_rx1_rx_rotate/")
        elif n_rx == 2:
            sys_response_folder = os.path.join(os.getcwd(), "sigs_tx2_rx2_rx_rotate/")

        n_tx = n_rx
        postfix = f"{n_rx}x{n_tx}"
        sys_response_path = os.path.join(self.config.channel_dir, f"sys_response_{postfix}.npz")
        if not os.path.exists(sys_response_folder):
            os.makedirs(sys_response_folder)

        if not os.path.exists(sys_response_path):
            sys_response = {}

            for file_name in os.listdir(sys_response_folder):
                if file_name.endswith(".npz") or file_name.endswith(".mat"):
                    self.print(f"Processing file: {file_name}", thr=0)
                    file_path = os.path.join(sys_response_folder, file_name)
                    if file_name.endswith(".npz"):
                        data = np.load(file_path)
                        data_dict = {key: data[key] for key in data.files}
                    elif file_name.endswith(".mat"):
                        data = scipy.io.loadmat(file_path)
                        data_dict = {
                            key: value for key, value in data.items() if not key.startswith("__")
                        }

                    spec_list = file_name.split("_")
                    angle = float(spec_list[0])
                    sys_response[angle] = {}

                    txtd_base = data_dict["txtd"]

                    rxtd_dict = {}
                    for key, value in data_dict.items():
                        if "rxtd_" not in key:
                            continue
                        frequency = float(key.split("_")[-1])
                        rxtd_dict[frequency] = value

                    for frequency, rxtd in rxtd_dict.items():
                        rxtd = np.mean(rxtd, axis=0)
                        rx_signal = self.rx_operations(txtd_base, rxtd)
                        max_gain = np.max(np.abs(rx_signal.h_est), axis=-1)
                        sys_response[angle][frequency] = max_gain

            angles = [float(angle) for angle in sys_response]
            angles = np.array(angles)
            angles = np.sort(angles)
            frequencies = np.array(list(sys_response[angles[0]].keys()))
            frequencies = np.sort(frequencies)

            n_rx_ = np.shape(sys_response[angles[0]][frequencies[0]])[0]
            n_tx_ = np.shape(sys_response[angles[0]][frequencies[0]])[1]
            sys_response_matrix = np.zeros((len(angles), len(frequencies), n_rx_, n_tx_))
            for i, angle in enumerate(angles):
                for j, frequency in enumerate(frequencies):
                    sys_response_matrix[i, j] = sys_response[angle][frequency]

            np.savez(
                sys_response_path,
                sys_response_matrix=sys_response_matrix,
                angles=angles,
                frequencies=frequencies,
            )

        else:
            data = np.load(sys_response_path)
            sys_response_matrix = data["sys_response_matrix"]
            angles = data["angles"]
            frequencies = data["frequencies"]

        sys_response_matrix /= np.max(sys_response_matrix)
        sys_response_matrix = self.lin_to_db(sys_response_matrix, mode="mag")

        plot_params_dict = {
            "title_size": 18,
            "title_weight": "bold",
            "title_max_chars": 45,
            "xaxis_size": 16,
            "yaxis_size": 16,
            "ticks_size": 14,
            "legend_size": 16,
            "line_width": 2.0,
            "marker_size": 8,
            "hspace": 0.5,
            "wspace": 0.5,
        }

        fixed_angles = [-90, -30, 0, 30, 90]
        if n_rx > 1:
            fixed_angles = [-90, 0, 30, 90]
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        lines = []
        for fixed_angle in fixed_angles:
            angle_id = np.where(angles == fixed_angle)[0][0]
            for rx_id in range(n_rx):
                (line,) = ax.plot(
                    frequencies,
                    sys_response_matrix[angle_id, :, rx_id, 0],
                    label=f"Angle {fixed_angle}, RX {rx_id}",
                )
                lines.append(line)
        plot_params_dict["title"] = "System Response vs Frequency at Different Angles"
        plot_params_dict["xlabel"] = "Frequency (GHz)"
        plot_params_dict["ylabel"] = "Normalized Response (dB)"
        self.plotter.set_plot_params(ax, lines, plot_params_dict)
        plt.savefig(os.path.join(self.config.figs_dir, f"sys_response_vs_freq_{postfix}.pdf"))

        fixed_freqs = [6.0, 8.0, 10.0, 15.0, 20.0]
        if n_rx > 1:
            fixed_freqs = [6.0, 10.0, 15.0, 20.0]
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        lines = []
        for fixed_freq in fixed_freqs:
            freq_id = np.where(frequencies == fixed_freq)[0][0]
            for rx_id in range(n_rx):
                (line,) = ax.plot(
                    angles,
                    sys_response_matrix[:, freq_id, rx_id, 0],
                    label=f"Fc {fixed_freq}GHz, RX {rx_id}",
                )
                lines.append(line)
        plot_params_dict["title"] = "System Response vs Angle at Different Frequencies"
        plot_params_dict["xlabel"] = "Angle (Deg)"
        plot_params_dict["ylabel"] = "Normalized Response (dB)"
        self.plotter.set_plot_params(ax, lines, plot_params_dict)
        plt.show()
        plt.savefig(os.path.join(self.config.figs_dir, f"sys_response_vs_angle_{postfix}.pdf"))

        rx_id = 0
        tx_id = 0
        for rx_id in range(n_rx):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            lines = []
            cax = ax.imshow(
                sys_response_matrix[:, :, rx_id, tx_id],
                extent=[frequencies[0], frequencies[-1], angles[0], angles[-1]],
                aspect="auto",
                origin="lower",
                cmap="viridis",
            )
            cbar = fig.colorbar(cax, ax=ax, label="Normalized Response (dB)")
            cbar.ax.tick_params(labelsize=plot_params_dict["ticks_size"])
            cbar.ax.yaxis.label.set_size(plot_params_dict["yaxis_size"])
            plot_params_dict["title"] = f"2D Heat Diagram of System Response for RX {rx_id}"
            plot_params_dict["xlabel"] = "Frequency (GHz)"
            plot_params_dict["ylabel"] = "Angle (Deg)"
            self.plotter.set_plot_params(ax, lines, plot_params_dict)
            plt.savefig(
                os.path.join(self.config.figs_dir, f"sys_response_2D_{postfix}_RX{rx_id}.pdf")
            )
            plt.show()

    def collect_signals(self):
        self.print("Collecting signals", thr=5)

        collect_count = 512
        ignore_less_count = False
        # input_folder = self.config.channel_dir
        input_folder = self.config.sig_dir
        # input_folder = "./sigs_tx1_rx1_rx_rotate"
        output_folder = os.path.join(input_folder, "collected")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file_name in os.listdir(input_folder):
            if file_name.endswith(".npz") or file_name.endswith(".mat"):
                self.print(f"Processing file: {file_name}", thr=0)
                file_path = os.path.join(input_folder, file_name)
                if file_name.endswith(".npz"):
                    data = np.load(file_path)
                    data_dict = {key: data[key] for key in data.files}
                elif file_name.endswith(".mat"):
                    data = scipy.io.loadmat(file_path)
                    data_dict = {
                        key: value for key, value in data.items() if not key.startswith("__")
                    }

                collected_data = {}
                for key, value in data_dict.items():
                    if (
                        not any(x in key for x in ["rxtd", "h_est"])
                        or ignore_less_count
                        and value.shape[0] < collect_count
                    ):
                        continue
                    else:
                        collect_count_ = 1 if key == "txtd" else collect_count
                        collect_count_ = min(value.shape[0], collect_count_)
                        collected_data[key] = value[:collect_count_]

                for key, value in collected_data.items():
                    if "rxtd" in key:
                        rxtd = value
                        self.validate_saved_signals(rxtd=rxtd, txtd=collected_data["txtd"])
                # output_file_path = os.path.join(output_folder, file_name)
                # np.savez(output_file_path, **collected_data)

    def find_optimal_gain_piradio(self, client_rfsoc_rx, client_piradio_rx, client_piradio_tx):
        self.print("Finding optimal TX/RX gains in Pi-Radio", thr=1)

        if os.path.exists(client_piradio_rx.config.optimal_gains_path):
            optimal_gains = self.load_dict_from_json(
                client_piradio_rx.config.optimal_gains_path, convert_values=True
            )
        else:
            optimal_gains = {}

        input_ = input(
            "Press Y for TX/RX optimal gains calibration or any key to use the saved data: "
        )
        if input_.lower() != "y":
            self.print("Using saved TX/RX optimal gains...", thr=0)
            return

        self.print("Finding optimal gain for TX/RX in Pi-Radio", thr=1)
        tx_rx_distance = input("Enter the distance between the TX and RX in meters: ")
        if tx_rx_distance != "":
            try:
                self.tx_rx_distance = float(tx_rx_distance)
            except Exception:
                raise ValueError(f"Invalid distance value: {self.tx_rx_distance}")  # noqa: B904
        else:
            pass
        optimal_gains[self.tx_rx_distance] = {}

        max_total_gain_db = 60
        min_tx_gain_db = 10
        max_tx_gain_db = 30
        min_rx_gain_db = 10
        max_rx_gain_db = 40
        gain_step_db = 1

        tx_gain_db_list = np.arange(min_tx_gain_db, max_tx_gain_db + gain_step_db, gain_step_db)
        rx_gain_db_list = np.arange(min_rx_gain_db, max_rx_gain_db + gain_step_db, gain_step_db)

        freq_list = [client_piradio_rx.config.stable_fc_piradio]
        for frequency in freq_list:
            self.print(f"Finding gains for frequency: {frequency} GHz", thr=1)
            for client in [client_piradio_rx, client_piradio_tx]:
                client.hop_freq(fc=frequency)

            optimal_gains[self.tx_rx_distance][frequency] = {}

            snr_db_optimal = 0
            tx_gain_db_optimal = 0
            rx_gain_db_optimal = 0

            for tx_gain_db in tx_gain_db_list:
                if tx_gain_db < min_tx_gain_db or tx_gain_db > max_tx_gain_db:
                    continue
                self.print(f"Setting TX gain to {tx_gain_db} dB", thr=1)
                client_piradio_tx.set_gain_piradio(trx="tx", chan=0, gain_db=tx_gain_db)
                client_piradio_tx.set_gain_piradio(trx="tx", chan=1, gain_db=tx_gain_db)

                for rx_gain_db in rx_gain_db_list:
                    if rx_gain_db < min_rx_gain_db or rx_gain_db > max_rx_gain_db:
                        continue
                    if tx_gain_db + rx_gain_db > max_total_gain_db:
                        continue

                    self.print(f"Setting RX gain to {rx_gain_db} dB", thr=1)
                    client_piradio_rx.set_gain_piradio(trx="rx", chan=0, gain_db=rx_gain_db)
                    client_piradio_rx.set_gain_piradio(trx="rx", chan=1, gain_db=rx_gain_db)
                    if client_piradio_rx.config.gain_sw_dly == 0:
                        time.sleep(2 * client_piradio_rx.config.piradio_gain_sw_dly_default)

                    rxtd = client_rfsoc_rx.receive_data_rfsoc(mode="once")
                    snr = self.calculate_snr(
                        sig_td=rxtd[0, :, : self.config.n_samples_trx],
                        sig_sc_range=self.config.sc_range,
                    )
                    snr_db = self.lin_to_db(snr, mode="pow")
                    self.print(
                        f"SNR for TX gain {tx_gain_db} dB and RX gain {rx_gain_db} dB: {snr_db:.3f} dB",
                        thr=1,
                    )
                    if snr_db > snr_db_optimal:
                        snr_db_optimal = snr_db
                        tx_gain_db_optimal = tx_gain_db
                        rx_gain_db_optimal = rx_gain_db

            self.print(
                f"Optimal TX gain for frequency {frequency}: {tx_gain_db_optimal} dB",
                thr=1,
            )
            self.print(
                f"Optimal RX gain for frequency {frequency}: {rx_gain_db_optimal} dB",
                thr=1,
            )
            self.print(f"Optimal SNR for frequency {frequency}: {snr_db_optimal} dB", thr=1)

            optimal_gains[self.tx_rx_distance][frequency]["tx_gain"] = int(tx_gain_db_optimal)
            optimal_gains[self.tx_rx_distance][frequency]["rx_gain"] = int(rx_gain_db_optimal)

        self.save_dict_to_json(optimal_gains, client_piradio_rx.config.optimal_gains_path)
        self.print("Calculated and saved optimal TX/RX gains...", thr=1)

        return optimal_gains

    def rx_operations(self, txtd_base, rxtd):
        self.print("Performing RX operations", thr=5)

        sparse_est_params = None
        plt_frm_id = 0
        n_rd_rep = rxtd.shape[0]

        if self.config.rfsoc_mixer_mode == "digital" and self.config.mix_freq_adc != 0:
            rxtd_base = np.zeros_like(rxtd)
            for frm_id in range(n_rd_rep):
                for ant_id in range(self.config.n_rx_ant):
                    rxtd_base[frm_id, ant_id, :] = self.shift_freq(
                        rxtd[frm_id, ant_id],
                        shift=-1 * self.config.mix_freq_adc,
                        fs=self.config.fs_rx,
                    )
        else:
            rxtd_base = rxtd.copy()

        if "filter" in self.config.rx_chain:
            for frm_id in range(n_rd_rep):
                for ant_id in range(self.config.n_rx_ant):
                    cf = (self.config.filter_bw_range[0] + self.config.filter_bw_range[1]) / 2
                    cutoff = self.config.filter_bw_range[1] - self.config.filter_bw_range[0]
                    rxtd_base[frm_id, ant_id, :] = self.filter(
                        rxtd_base[frm_id, ant_id, :],
                        center_freq=cf,
                        cutoff=cutoff,
                        fil_order=64,
                        plot=False,
                    )

        for ant_id in range(self.config.n_rx_ant):
            # n_samples = min(len(txtd_base), len(rxtd_base))
            txfd_base_ = np.abs(fftshift(fft(txtd_base[0, ant_id, : self.config.n_samples])))
            rxfd_base_ = np.abs(
                fftshift(fft(rxtd_base[plt_frm_id, ant_id, : self.config.n_samples]))
            )

            scale = np.max(txfd_base_) / np.max(rxfd_base_)
            self.print(f"TX to RX spectrum scale for antenna {ant_id}: {scale:0.3f}", thr=4)
            self.print(
                "txfd_base max freq for antenna {}: {} MHz".format(
                    ant_id,
                    self.config.freq[
                        (self.config.nfft >> 1) + np.argmax(txfd_base_[self.config.nfft >> 1 :])
                    ],
                ),
                thr=4,
            )
            self.print(
                "rxfd_base max freq for antenna {}: {} MHz".format(
                    ant_id,
                    self.config.freq[
                        (self.config.nfft >> 1) + np.argmax(rxfd_base_[self.config.nfft >> 1 :])
                    ],
                ),
                thr=4,
            )

        if "pilot_separate" in self.config.rx_chain:
            n_samples_rx = self.config.n_samples_trx * 2
        else:
            n_samples_rx = self.config.n_samples_trx

        txtd_base = txtd_base[:, :, : self.config.n_samples_trx]
        if "integrate" in self.config.rx_chain:
            rxtd_base = self.integrate_signal(rxtd_base, n_samples=n_samples_rx)

        if "sync_time" in self.config.rx_chain:
            rxtd_base_s = []
            for frm_id in range(n_rd_rep):
                sync_frac = "sync_time_frac" in self.config.rx_chain
                rxtd_base_s_ = self.sync_time(
                    rxtd_base[frm_id],
                    txtd_base[0],
                    sc_range=self.config.sc_range,
                    rx_same_delay=self.config.rx_same_delay,
                    sync_frac=sync_frac,
                )
                rxtd_base_s.append(rxtd_base_s_)
            rxtd_base_s = np.array(rxtd_base_s)
        else:
            rxtd_base_s = rxtd_base.copy()
            rxtd_base_s = np.stack((rxtd_base_s, rxtd_base_s), axis=2)

        if "sync_freq" in self.config.rx_chain:
            cfo_coarse = self.estimate_cfo(
                txtd_base[0], rxtd_base_s[0], mode="coarse", sc_range=self.config.sc_range
            )
            rxtd_base_t = []
            for frm_id in range(n_rd_rep):
                rxtd_base_t_ = self.sync_frequency(rxtd_base_s[frm_id], cfo_coarse, mode="time")
                rxtd_base_t.append(rxtd_base_t_)
            rxtd_base_t = np.array(rxtd_base_t)
            cfo_fine = self.estimate_cfo(
                txtd_base[0], rxtd_base_t[0], mode="fine", sc_range=self.config.sc_range
            )
            cfo = cfo_coarse + cfo_fine
            for frm_id in range(n_rd_rep):
                rxtd_base_s[frm_id] = self.sync_frequency(rxtd_base_s[frm_id], cfo, mode="time")

        if "pilot_separate" in self.config.rx_chain:
            rxtd_pilot_s = rxtd_base_s[:, :, :, : n_samples_rx // 2]
            rxtd_base_s = rxtd_base_s[:, :, :, n_samples_rx // 2 :]
        else:
            rxtd_pilot_s = rxtd_base_s.copy()

        rxtd_base = np.stack(
            (
                rxtd_base_s[:, 0, 0, : self.config.n_samples_trx],
                rxtd_base_s[:, 1, 0, : self.config.n_samples_trx],
            ),
            axis=1,
        )
        rxtd_pilot = np.stack(
            (
                rxtd_pilot_s[:, 0, 0, : self.config.n_samples_trx],
                rxtd_pilot_s[:, 1, 0, : self.config.n_samples_trx],
            ),
            axis=1,
        )

        if "channel_est" in self.config.rx_chain:
            if "sys_res_deconv" in self.config.rx_chain:
                self.sys_response = self.process_sys_response()
            else:
                self.sys_response = None
            snr_est = self.db_to_lin(self.config.snr_est_db, mode="pow")

            if "estimate_sparse_params" in self.config.rx_chain:
                h = []
                for frm_id in range(n_rd_rep):
                    h_est = self.estimate_channel(
                        txtd_base[0],
                        rxtd_pilot_s[frm_id],
                        sys_response=self.sys_response,
                        sc_range_ch=self.config.sc_range_ch,
                        snr_est=snr_est,
                    )
                    h.append(h_est)
                h = np.array(h)
                h = h.transpose(3, 1, 2, 0)
                g = self.sys_response.copy() if self.sys_response is not None else None
                if g is not None:
                    g = g.transpose(2, 0, 1)
                ndly = 5000
                sparse_est_params = self.estimate_sparse_params(
                    h=h,
                    g=g,
                    sc_range_ch=self.config.sc_range_ch,
                    npaths=self.config.npath_max,
                    nframe_avg=1,
                    ndly=ndly,
                    drange=self.config.sparse_ch_samp_range,
                    cv=True,
                    n_ignore=self.config.sparse_ch_n_ignore,
                )
                sparse_est_params = SparseEstParams(
                    sparse_est_params[0],
                    sparse_est_params[1],
                    sparse_est_params[2],
                    sparse_est_params[3],
                )
            else:
                h_est = self.estimate_channel(
                    txtd_base[0],
                    rxtd_pilot_s[0],
                    sys_response=self.sys_response,
                    sc_range_ch=self.config.sc_range_ch,
                    snr_est=snr_est,
                )
            self.rx_phase_list, self.aoa_list = self.angle_of_arrival(
                rxtd=rxtd_pilot,
                rx_phase_list=self.rx_phase_list,
                aoa_list=self.aoa_list,
                fc=self.config.fc,
                rx_phase_offset=self.rx_phase_offset,
                rx_delay_offset=self.rx_delay_offset,
            )
        else:
            h_est = None
        if "channel_eq" in self.config.rx_chain and "channel_est" in self.config.rx_chain:
            rxtd_base = self.equalize_channel(
                txtd_base[0],
                rxtd_base[plt_frm_id],
                h_est,
                sc_range=self.config.sc_range,
                sc_range_ch=self.config.sc_range_ch,
                null_sc_range=self.config.null_sc_range,
                n_rx_ch_eq=self.config.n_rx_ch_eq,
            )

        self.rx_signal = RxSignal(
            rxtd=rxtd,
            rxtd_base=rxtd_base,
            h_est=h_est,
            sparse_est_params=sparse_est_params,
        )

        return self.rx_signal

    def process_sig(self, sig=None, process_list=()):
        self.print(f"Processing signal with operations: {process_list}", thr=5)

        if sig is None:
            return None

        sig = sig.copy()
        title = ""
        for item in process_list:
            if item in ["tx", "rx", "h", "H"]:
                continue
            elif item == "fft":
                sig = fft(sig, axis=-1)
                # title += "-FFT"
                title += "-FD"
            elif item == "psd":
                nfft = 2 ** int(np.ceil(np.log2(len(sig))))
                sig = self.psd(sig, fs=self.config.fs_rx, nfft=nfft)
            elif item == "ifft":
                sig = ifft(sig, axis=-1)
                title += "-IFFT"
            elif item == "fftshift":
                sig = fftshift(sig, axes=-1)
            elif item == "ifftshift":
                sig = ifftshift(sig, axes=-1)
            elif item == "mag":
                sig = np.abs(sig)
                title += "-Mag"
            elif item == "phase":
                sig = np.angle(sig)
                title += "-Phase"
            elif item == "phase/2pi":
                sig = np.angle(sig) / (2 * np.pi)
                title += "-Phase/2pi"
            elif item == "phase_unwrap":
                sig = np.unwrap(np.angle(sig))
                title += "-PhaseUnwrap"
            elif item == "real":
                sig = np.real(sig)
                title += "-Real"
            elif item == "imag":
                sig = np.imag(sig)
                title += "-Imag"
            elif item == "IQ":
                n_samples = sig.shape[-1]
                sig = sig[
                    self.config.sc_range[0] + n_samples // 2 : self.config.sc_range[1]
                    + n_samples // 2
                    + 1
                ]
                title += "-IQ"
            elif item == "conj":
                sig = np.conj(sig)
                title += "-Conj"
            elif item == "dbmag":
                sig = self.lin_to_db(sig, mode="mag")
                title += "-dBMag"
            elif item == "dbpow":
                sig = self.lin_to_db(sig, mode="pow")
                title += "-dBPow"
            elif item == "circshift":
                im = np.argmax(np.abs(sig), axis=-1)
                sig = np.roll(sig, -im + len(sig) // 4, axis=-1)
            elif item == "normalize":
                sig = sig / np.max(np.abs(sig))
                title += "-Norm"
            else:
                raise ValueError(f"Invalid operation: {item}")

        return sig, title


@dataclass(kw_only=True)
class ExperimentOperatorConfig(SignalUtilsRFSoCConfig):
    measurement_configs: tuple = None  # List of measurement configurations
    # host_role: str = "client"  # Mode of operation, client or client_master or client_slave
    RFFE: str = "piradio"  # RF front end to use, piradio or sivers
    network_topology: dict = None  # Network topology configuration
    action_loop: tuple = None


class ExperimentOperator(SignalUtilsRfsoc):
    def __init__(self, config: ExperimentOperatorConfig, **overrides):
        super().__init__(config, **overrides)

        self._network_topology = self.config.network_topology
        self._network_objects = {}
        self.animate_plotter = None

        self.create_dirs(
            [
                self.config.calib_config_dir,
                self.config.sig_dir,
                self.config.channel_dir,
                self.config.figs_dir,
            ]
        )

    @property
    def network_topology(self):
        return self._network_topology

    @property
    def network_objects(self):
        return self._network_objects

    @property
    def rfsoc_tx_list(self):
        rfsoc_tx_list = []
        for name in self.network_topology:
            item = self.network_topology[name]
            if item["type"] in ["rfsoc", "controller"] and item["role"] == "tx":
                rfsoc_tx_list.append(name)
        return rfsoc_tx_list

    @property
    def piradio_tx_list(self):
        piradio_tx_list = []
        for name in self.network_topology:
            item = self.network_topology[name]
            if item["type"] in ["piradio", "controller"] and item["role"] == "tx":
                piradio_tx_list.append(name)
        return piradio_tx_list

    @property
    def rfsoc_rx_list(self):
        rfsoc_rx_list = []
        for name in self.network_topology:
            item = self.network_topology[name]
            if item["type"] in ["rfsoc", "controller"] and item["role"] == "rx":
                rfsoc_rx_list.append(name)
        return rfsoc_rx_list

    @property
    def piradio_rx_list(self):
        piradio_rx_list = []
        for name in self.network_topology:
            item = self.network_topology[name]
            if item["type"] in ["piradio", "controller"] and item["role"] == "rx":
                piradio_rx_list.append(name)
        return piradio_rx_list

    def init_objects(self):
        self.print("Initializing network objects", thr=1)

        self._network_objects = {}
        self._network_objects["self"] = self

        for name in self.network_topology:
            item_type = self.network_topology[name]["type"]
            kwargs = {k: v for k, v in self.network_topology[name].items() if k != "type"}

            method_name = f"_init_{item_type.lower()}"
            init_method = getattr(self, method_name, None)

            if init_method:
                self.print(
                    f"Initializing object: {name} with type: {item_type} and params: {kwargs}",
                    thr=2,
                )
                item_object = init_method(**kwargs)
                self._network_objects[name] = item_object
            else:
                raise NotImplementedError(f"Initialization handler '{method_name}' is not defined.")

    def _init_rfsoc(self, ip, **kwargs):
        rfsoc_config = ClientRFSoCConfig(server_ip=ip).update_from_config(self.config)
        rfsoc = ClientRFSoC(rfsoc_config)
        rfsoc.init_tcp_client()
        return rfsoc

    def _init_lintrack_client(self, ip, **kwargs):
        lintrack_config = TCPComLinTrackConfig(server_ip=ip).update_from_config(self.config)
        lintrack = TcpCommLinTrack(lintrack_config)
        lintrack.init_tcp_client()
        return lintrack

    def _init_lintrack(self, run_server=False, **kwargs):
        lintrack_config = LinearTrackControllerConfig()
        lintrack = LinearTrackController(lintrack_config)
        if run_server:
            lintrack.run_tcp()
        return lintrack

    def _init_turntable(self, port="COM6", baudrate=115200, rotation_delay=0.0, **kwargs):
        turntable_config = SerialComTurnTableConfig(
            port=port, baudrate=baudrate, rotation_delay=rotation_delay
        )
        turntable = SerialComTurnTable(turntable_config)
        turntable.connect()
        turntable.move_to_position(0)
        calibrate = kwargs.get("calibrate", False)
        if calibrate:
            turntable.calibrate()
        turntable.interactive_move()
        return turntable

    def _init_d48ptu(self, port="/dev/ttyUSB0", baudrate=9600, **kwargs):
        gimbal_config = SerialComD48PTUConfig(
            port=port,
            baudrate=baudrate,
        )
        D48PTU = SerialComD48PTU(gimbal_config)
        try:
            D48PTU.connect()
        except Exception as e:
            self.print(f"Error occurred while connecting to D48PTU: {e}", thr=0)
            D48PTU.list_ports()
            self.print("Please check the connection and try again.", thr=0)
        return D48PTU

    def _init_piradio(self, ip, freq_sw_dly=0.1, gain_sw_dly=0.1, bias_sw_dly=0.1, **kwargs):
        piradio_config = PiRadioConfig(
            ip_address=ip,
            freq_sw_dly=freq_sw_dly,
            gain_sw_dly=gain_sw_dly,
            bias_sw_dly=bias_sw_dly,
        ).update_from_config(self.config)
        piradio = PiRadioFR3Trx(piradio_config)
        piradio.set_frequency_piradio(fc=self.config.fc)
        return piradio

    def _init_turtlebot(self, **kwargs):
        # try:
        #     from tb4_aoa_viz.aoa_bridge import get_publish_aoa_fn  # type: ignore  # noqa: I001
        #     from tb4_aoa_viz.snr_bridge import get_publish_snr_fn  # type: ignore  # noqa: I001

        #     self.publish_aoa_turtlebot = get_publish_aoa_fn("/aoa_angle")
        #     self.publish_snr_turtlebot = get_publish_snr_fn("/snr_db")
        # except ImportError:
        #     self.print(
        #         "tb4_aoa_viz package not found, turtlebot publishing disabled", thr=0
        #     )
        #     self.publish_aoa_turtlebot = lambda x: None
        #     self.publish_snr_turtlebot = lambda x: None

        turtlebot_config = TurtlebotConfig().update_from_config(self.config)
        turtlebot = Turtlebot(turtlebot_config)
        return turtlebot

    def _init_controller_client(self, ip, **kwargs):
        controller_config = TCPComControllerConfig(server_ip=ip).update_from_config(self.config)
        controller = TcpCommController(controller_config)
        controller.init_tcp_client()
        controller.set_frequency_piradio(self.config.fc)
        return controller

    def _init_controller_server(self, **kwargs):
        controller_config = TCPComControllerConfig().update_from_config(self.config)
        controller = TcpCommController(controller_config)
        controller.init_tcp_server()
        for item in self.network_topology.items():
            item_name, item_info = item
            if not item_info["type"] in ["controller", "controller_server"]:
                controller.__dict__[f"obj_{item_info['type']}"] = self._network_objects[
                    item_name
                ]
        self._network_objects["self"] = controller
        controller.run_tcp_server(controller.parse_and_execute)

    def _parse_action_spec(self, spec):
        """Parses dict-based action specs."""
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid action_loop spec type: {type(spec)}")

        targets = spec.get("targets", [])
        actions = spec.get("actions", [])
        values = spec.get("values", [None])
        params = spec.get("params", {})

        if isinstance(targets, str):
            targets = [targets]
        if isinstance(actions, str):
            actions = [actions]
        # if isinstance(values, str):
        #     values = [values]
        if isinstance(params, str):
            params = {"default": params}

        if not actions:
            raise ValueError(f"Action spec is missing valid action: {spec}")

        rng = self._parse_range(values)
        return targets, actions, rng, params

    def _parse_range(self, values):
        """Handles step/count/log syntaxes, lists, and scalar values safely without eval()."""
        if isinstance(values, np.ndarray):
            return values

        if isinstance(values, (list, tuple)):
            return np.array(values)

        if isinstance(values, (int, float)):
            return np.array([values])

        if isinstance(values, str):
            # Handle range syntax: start:stop:step or start:stop:count:log
            if values.count(":") >= 2:
                parts = values.split(":")
                start = float(parts[0])
                stop = float(parts[1])
                third = float(parts[2])

                if third == 0:
                    raise ValueError(f"Invalid zero step/count in values: {values}")
                count = int(third)

                if len(parts) > 3 and "log" in values:
                    return np.logspace(np.log10(start), np.log10(stop), count)
                else:
                    return np.linspace(start, stop, count)

            # Safely evaluate lists or numbers encoded as strings
            try:
                val = ast.literal_eval(values)
                if isinstance(val, (list, tuple)):
                    return np.array(val)
                if isinstance(val, (int, float)):
                    return np.array([val])
            except (ValueError, SyntaxError):
                pass

        return np.array([None])

    def _get_target_objects(self, targets_list):
        """Resolves target strings into actual objects."""
        if isinstance(targets_list, str):
            targets_list = [targets_list]

        objects = []
        for target in targets_list:
            obj = self.network_objects.get(target)
            if not obj:
                raise ValueError(f"Invalid target object: {target}")
            objects.append(obj)
        return objects

    def run_operator(self):
        self.print("Running experiment operator", thr=1)

        self.measurement = {}
        self.save_id = 1
        self.phys_config = None
        self.phys_config_id = 0
        # self.angle_id = 0
        # self.freq_id = 0
        self.read_id = 0

        loop_list = [self._parse_action_spec(spec) for spec in self.config.action_loop]

        targets = [item[0] for item in loop_list]
        actions = [item[1] for item in loop_list]
        ranges = [item[2] for item in loop_list]
        params = [item[3] for item in loop_list]

        prev = None
        default_actions = ["capture", "save", "store", "wait", "update_plot", "print_snr"]
        default_actions_contain = ["print"]

        for values in itertools.product(*ranges):
            print(f"Current Sweep Values: {values}")
            if prev is None:
                changed_idxs = range(len(values))  # first iteration: everything is "changed"
            else:
                # indices where the value differs from previous step
                changed_idxs = [
                    i
                    for i, (a, b) in enumerate(zip(prev, values, strict=False))
                    if a != b or any(action in default_actions for action in actions[i])
                    or any(action_contain in action for action in actions[i] for \
                            action_contain in default_actions_contain)
                ]
            prev = values

            # process only the actions whose value changed
            for i in changed_idxs:
                target_objects = self._get_target_objects(targets[i])
                action_names = actions[i]
                value = values[i]
                kwargs = params[i]

                if isinstance(action_names, str):
                    action_names = [action_names]

                for action_name in action_names:
                    # DYNAMIC DISPATCH: Look for a method named `action_<action_name>`
                    method_name = f"_action_{action_name.lower()}"
                    action_method = getattr(self, method_name, None)

                    if action_method:
                        self.print(
                            f"Executing action: {action_name} with value: {value} and params: {kwargs}",
                            thr=2,
                        )
                        action_method(target_objects, value, **kwargs)
                    else:
                        raise NotImplementedError(f"Action handler '{method_name}' is not defined.")

    def _action_loop(self, target_objects, value, **kwargs):
        self.print(f"Starting loop iteration with value: {value}", thr=1)

    def _action_change_phys_config(self, target_objs, value, **kwargs):
        if not self.config.measurement_configs:
            raise ValueError("measurement_configs is empty; cannot change physical configuration")
        self.phys_config = self.config.measurement_configs[self.phys_config_id]
        self.print(f"Please change the physical configuration to: {self.phys_config}", thr=0)
        self.phys_config_id = (self.phys_config_id + 1) % len(self.config.measurement_configs)

    def _action_change_tx_rx_distance(self, target_objs, value, **kwargs):
        tx_rx_distance = input("Please enter the TX to RX distance in meters (empty for default): ")
        if tx_rx_distance != "":
            try:
                tx_rx_distance = float(tx_rx_distance)
            except Exception:
                raise ValueError(f"Invalid distance value: {tx_rx_distance}") from None
            self.tx_rx_distance = tx_rx_distance

    def _action_transmit_signal(self, target_objects, value, **kwargs):
        for client_rfsoc in target_objects:
            client_rfsoc.transmit_data_rfsoc(self.tx_signal.txtd)

    def _action_capture(self, target_objects, value, process_signal=False, **kwargs):
        client_rfsoc = target_objects[0]
        n_frames = int(value)
        rxtd_save = []

        n_rd_rep = n_frames // self.config.n_frame_rd
        rxtd = client_rfsoc.receive_data_rfsoc(n_rd_rep=n_rd_rep, mode="once", verbose=False)
        self.rx_signal = RxSignal(
            rxtd=rxtd,
            rxtd_base=rxtd,
        )

        if process_signal:
            n_rd_rep = 1
            for i in range(n_frames):
                self.print(f"Channel Save Iteration: {i + 1}", thr=0)
                rxtd = client_rfsoc.receive_data_rfsoc(n_rd_rep=n_rd_rep, mode="once")

                # to handle the dimenstion needed for read repeat
                rx_signal = self.rx_operations(self.tx_signal.txtd_base, rxtd)
                self.rx_signal = rx_signal

                rxtd_save.append(rx_signal.rxtd_base)
            rxtd_save = np.array(rxtd_save)
            rxtd_save = rxtd_save.reshape(-1, *rxtd_save.shape[-2:])
        else:
            rxtd_save = np.empty(
                (n_frames, self.config.n_rx_ant, self.config.n_samples_tx),
                dtype=rxtd.dtype,
            )
            for i in range(self.config.n_frame_rd):
                rxtd_save[i :: self.config.n_frame_rd] = rxtd[
                    :,
                    :,
                    i * self.config.n_samples_tx : (i + 1) * self.config.n_samples_tx,
                ]

        self.txtd_save = self.tx_signal.txtd_base
        self.rxtd_save = rxtd_save

        # self.validate_saved_signals(rxtd=self.rxtd_save)

    def _action_calibrate_rfsoc(self, target_objects, value, **kwargs):
        for client_rfsoc in target_objects:
            client_rfsoc.calibrate_rx_phase_offset()

    def _action_set_frequency_mixer_rfsoc(self, target_objects, value, **kwargs):
        frequency = float(value)
        for client_rfsoc in target_objects:
            client_rfsoc.set_frequency_mixer_rfsoc(f_mixer_dac=frequency, f_mixer_adc=frequency)

    def _action_capture_from_file(self, target_objects, value, sig_name="", **kwargs):
        sig_name = sig_name if sig_name else value
        sig_path = os.path.join(self.config.sig_dir, sig_name)
        sigs_save = np.load(sig_path)
        n_rd_rep = 1

        rxtd = sigs_save[f"rxtd_{self.config.fc / 1e9:.1f}"][
            self.read_id * n_rd_rep : (self.read_id + 1) * n_rd_rep
        ]
        txtd_base = sigs_save["txtd"]

        rx_signal = self.rx_operations(txtd_base, rxtd)
        self.rx_signal = rx_signal
        self.tx_signal = TxSignal(
            txtd_base=txtd_base,
        )
        self.read_id += 1

    def _action_update_plot(self, target_objects, value, **kwargs):
        rx_signal = self.rx_signal
        if rx_signal is None:
            raise ValueError("update_plot requires a valid rx_signal; run capture first")
        self.animate_plotter.update_once(rx_signal)

    def _action_save(self, target_objects, value, save_list=("signal",), **kwargs):
        if "signal" in save_list:
            self.measurement["txtd"] = self.txtd_save.copy()
            self.measurement[f"rxtd_{self.config.fc / 1e9}"] = self.rxtd_save.copy()
        if "sig_interval" in save_list:
            self.measurement["sig_interval"] = [
                self.config.wb_sc_range[0] + (self.config.nfft_tx >> 1),
                self.config.wb_sc_range[1] + (self.config.nfft_tx >> 1),
            ]
        if "snr_db" in save_list:
            snr = self.calculate_snr(
                sig_td=self.rx_signal.rxtd_base[:, 0, : self.config.n_samples_trx],
                sig_sc_range=self.config.sc_range,
            )
            snr_db = self.lin_to_db(snr, mode="pow")
            self.measurement["snr_db"] = snr_db
        if "aoa" in save_list:
            aoa = self.aoa_list[-1] if len(self.aoa_list) > 0 else 0
            self.measurement["aoa"] = aoa
        if "turtlebot_info" in save_list:
            client_turtlebot = target_objects[0]
            self.measurement["tx_pos"] = client_turtlebot.tx_pos
            self.measurement["tx_orientation"] = client_turtlebot.tx_orientation
            self.measurement["rx_pos"] = client_turtlebot.turtlebot_pos
            self.measurement["rx_orientation"] = client_turtlebot.turtlebot_orientation

    def _action_store(self, target_objects, value, save_prefix="m", **kwargs):
        save_postfix = f"{self.phys_config}_" if self.phys_config is not None else ""
        save_name = f"{save_prefix}_{save_postfix}{self.save_id}.{self.config.save_format}"
        self.measurement["id"] = self.save_id

        save_path = os.path.join(self.config.sig_dir, save_name)
        if self.config.save_format == "npz":
            np.savez(save_path, **self.measurement)
        elif self.config.save_format == "mat":
            scipy.io.savemat(save_path, self.measurement)

        self.save_id += 1

    def _action_wait(self, target_objects, value, **kwargs):
        wait_time = float(value)
        time.sleep(wait_time)

    def _action_report_time(self, target_objects, value, action="start", n_rep=0, **kwargs):
        if action == "start":
            self.start_time = time.time()
        elif action == "end":
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            self.print(f"Total time elapsed from last start: {elapsed_time:0.3f} s", thr=0)
        self.print(f"Total time remaining: {n_rep * elapsed_time:0.3f} s", thr=0)

    def _action_rotate_turntable(self, target_objects, value, **kwargs):
        angle = float(value)
        for client_turntable in target_objects:
            client_turntable.move_to_position_turntable(angle)

    def _action_move_lintrack(self, target_objects, value, **kwargs):
        distance = float(value)
        for client_lintrack in target_objects:
            client_lintrack.move_lintrack(lintrack_id=0, distance=distance)

    def _action_return_lintrack_home(self, target_objects, value, **kwargs):
        for client_lintrack in target_objects:
            client_lintrack.return2home_lintrack(lintrack_id=0)

    def _action_publish_ros2(self, target_objects, value, publish_list=(), **kwargs):
        if "aoa" in publish_list:
            aoa = self.aoa_list[-1] if len(self.aoa_list) > 0 else 0
            self.publish_aoa_turtlebot(aoa)
        if "snr" in publish_list:
            snr = self.calculate_snr(
                sig_td=self.rx_signal.rxtd_base[:, 0, : self.config.n_samples_trx],
                sig_sc_range=self.config.sc_range,
            )
            snr_db = self.lin_to_db(snr, mode="pow")
            self.publish_snr_turtlebot(snr_db)

    def _action_print(self, target_objects, value, print_list=(), **kwargs):
        if "snr" in print_list:
            snr = self.calculate_snr(
                sig_td=self.rx_signal.rxtd_base[:, 0, : self.config.n_samples_trx],
                sig_sc_range=self.config.wb_sc_range,
            )
            snr_db = self.lin_to_db(snr, mode="pow")
            self.print(f"Estimated SNR: {snr_db:.2f} dB", thr=0)

    def _action_hop_freq(self, target_objects, value, **kwargs):
        frequency = float(value)
        for client in target_objects:
            if self.config.RFFE == "sivers":
                client.set_frequency_sivers(fc=frequency)
            elif self.config.RFFE == "piradio":
                client.hop_freq_piradio(fc=frequency)

    def _action_set_gain_db_tx(self, target_objects, value, **kwargs):
        gain_db = int(value)
        for client in target_objects:
            if self.config.RFFE == "sivers":
                client.set_tx_gain_sivers()
            elif self.config.RFFE == "piradio":
                client.set_gain_piradio(trx="tx", chan=0, gain_db=gain_db)
                client.set_gain_piradio(trx="tx", chan=1, gain_db=gain_db)

    def _action_set_gain_db_rx(self, target_objects, value, **kwargs):
        gain_db = round(float(value), 1)
        for client in target_objects:
            if self.config.RFFE == "sivers":
                client.set_rx_gain_sivers()
            elif self.config.RFFE == "piradio":
                client.set_gain_piradio(trx="rx", chan=0, gain_db=gain_db)
                client.set_gain_piradio(trx="rx", chan=1, gain_db=gain_db)

    def _action_find_optimal_gain_piradio(self, target_objects, value, **kwargs):
        client_rfsoc_rx, client_piradio_rx, client_piradio_tx = target_objects
        optimal_gains = self.find_optimal_gain_piradio(
            client_rfsoc_rx, client_piradio_rx, client_piradio_tx
        )
        client_piradio_rx.optimal_gains = optimal_gains
        client_piradio_tx.optimal_gains = optimal_gains

    def _action_set_optimal_gain_piradio(self, target_objects, value, **kwargs):
        client_piradio_rx, client_piradio_tx = target_objects
        client_piradio_rx.set_optimal_gain_piradio(tx_rx_distance=self.tx_rx_distance, side="rx")
        client_piradio_tx.set_optimal_gain_piradio(tx_rx_distance=self.tx_rx_distance, side="tx")

    def _action_set_optimal_losupp_piradio(self, target_objects, value, **kwargs):
        for client_piradio in target_objects:
            client_piradio.set_optimal_losupp_piradio()

    def _action_switch_sig_size(self, target_objects, value, **kwargs):
        self.sig_size = int(value)

    def _action_switch_sig_ss(self, target_objects, value, **kwargs):
        bw_limit = 390.0e6
        sc_limit = int(np.round(bw_limit * self.config.nfft_tx / self.config.fs_tx)) * 2
        region = SpecSenseUtils.generate_random_regions(
            shape=(sc_limit,), n_regions=1, min_size=[self.sig_size], max_size=[self.sig_size]
        )
        self.config.wb_sc_range = [
            region[0][0].start - (sc_limit >> 1),
            region[0][0].stop - 1 - (sc_limit >> 1),
        ]
        signal_length = self.config.wb_sc_range[1] - self.config.wb_sc_range[0] + 1
        tx_signal = self.gen_tx_signal()
        tx_signal.txtd /= (256 / signal_length) ** 0.5
        tx_signal.txtd_base /= (256 / signal_length) ** 0.5
        for client_rfsoc in target_objects:
            client_rfsoc.transmit_data_rfsoc(tx_signal.txtd)

    def _action_move_turtlebot(self, target_objects, value, **kwargs):
        for client_turtlebot in target_objects:
            position = client_turtlebot.get_next_turtlebot_position()
            client_turtlebot.move_to(position)
            client_turtlebot.rotate_to(client_turtlebot.tx_pos)

    def _action_move_lintrack_trurtlebot(self, target_objects, value, **kwargs):
        client_turtlebot, client_lintrack = target_objects
        position = client_turtlebot.get_next_lintrack_position()
        client_lintrack.go2pos_lintrack(lintrack_id=0, position=position)

    def _action_move_gimbal_trurtlebot(self, target_objects, value, **kwargs):
        client_turtlebot, client_gimbal = target_objects
        az, el = client_turtlebot.get_next_gimbal_angles()
        client_gimbal.goto_deg_d48ptu(azimuth_deg=az, elevation_deg=el)

    def _action_set_gimbal_az(self, target_objects, value, **kwargs):
        az = float(value)
        for client_gimbal in target_objects:
            client_gimbal.goto_deg_d48ptu(azimuth_deg=az)

    def _action_set_gimbal_el(self, target_objects, value, **kwargs):
        el = float(value)
        for client_gimbal in target_objects:
            client_gimbal.goto_deg_d48ptu(elevation_deg=el)

    def _action_set_mode_sivers(self, target_objects, value, **kwargs):
        mode = str(value)
        mode = "RXen1_TXen0" if mode == "rx" else "RXen0_TXen1"
        for client_rfsoc in target_objects:
            client_rfsoc.set_mode_sivers(mode)


@dataclass(kw_only=True)
class AnimationPlotConfig(PlotUtilsConfig):
    animate_plot_mode: tuple = None
    plot_configs: dict = None


class AnimatePlot(PlotUtils):
    def __init__(
        self, config: AnimationPlotConfig, signals_obj: SignalUtilsRfsoc, **overrides: Any
    ):
        super().__init__(config, **overrides)

        self.signals_obj = signals_obj
        self.signals_config = signals_obj.config

        self.config.n_plots_row = len(self.config.animate_plot_mode)
        self.config.n_plots_col = len(self.signals_config.freq_hop_list)

        self.config.plt_n_samples_rx = self.signals_config.n_samples_trx
        self.config.n_samp_ch_sp = self.signals_config.n_samples_ch // 2

        self.plot_colors = [
            "#57068C",
            "orange",
            "green",
            "red",
            "blue",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        # set matplotlib axes color cycle so subsequent ax.plot calls use our colors by default
        with contextlib.suppress(Exception):
            mpl.rcParams["axes.prop_cycle"] = cycler("color", self.plot_colors)
        self.mag_filter_list = {"process_list": ["fft"], "signal_name": ["h", "H"]}
        self.untouched_plot_list = {"process_list": ["IQ"], "signal_name": ["aoa_gauge", "nf_loc"]}

        self.anim_paused = False
        self.read_id = -1
        self.plots_initialized = False

    def process_signals_for_plot(self, rx_signal: RxSignal):
        """
        Instructions to build signals for plots:

        template:   ["signal_name|rx_id|tx_id|process_list"]

        h :         ["h|0|0|circshift|mag|dbmag"]
        h01 :       ["h|0|0|circshift|mag|dbmag", "h|1|0|circshift|mag|dbmag"]
        h_sparse :  ["h_sparse|0|0"]
        H :         ["H|0|0|fftshift|mag|dbmag"]
        H_phase :   ["H|0|0|fftshift|phase"]
        rxtd :      ["rxtd|0|0|real", "rxtd|0|0|imag"]
        rxtd01 :    ["rxtd|0|0|mag", "rxtd|1|0|mag"]
        rxtd_phase :["rxtd|0|0|phase"]
        rxfd :      ["rxtd|0|0|fft|fftshift|mag|dbmag"]
        rxfd01 :    ["rxtd|0|0|fft|fftshift|mag|dbmag", "rxtd|1|0|fft|fftshift|mag|dbmag"]
        txtd :      ["txtd|0|0"]
        txfd :      ["txtd|0|0|fft|fftshift|mag|dbmag"]
        IQ :        ["rxtd|0|0|fft|fftshift|IQ"]
        aoa_gauge : ["aoa_gauge|0|0"]
        nf_loc :    ["nf_loc|0|0"]
        """

        self.print("Processing signals for plot", thr=5)

        rxtd_base = rx_signal.rxtd_base
        h_est = rx_signal.h_est
        sparse_est_params = rx_signal.sparse_est_params

        supported_operations = ["+", "-", "*", "/"]
        signals = []
        for plot in self.config.animate_plot_mode:
            plot_signals = []
            rx_ids = []
            tx_ids = []
            title = ""

            sig_final = None
            label_final = None

            for index, signal_str in enumerate(plot):
                if signal_str in supported_operations:
                    continue

                x = None
                sig = None
                if index != 0:
                    title += ", "

                signal_desc = signal_str.strip().split("|")

                signal_name = signal_desc[0]
                rx_id = int(signal_desc[1])
                tx_id = int(signal_desc[2])
                rx_ids.append(rx_id)
                tx_ids.append(tx_id)

                signal_process_list = signal_desc[3:] if len(signal_desc) > 3 else []

                xlabel_mode = "time"
                if "mag" in signal_process_list:
                    ylabel_mode = "mag"
                    if "dbmag" in signal_process_list:
                        ylabel_mode += "_db"
                elif "phase" in signal_process_list:
                    ylabel_mode = "phase"
                elif "phase/2pi" in signal_process_list:
                    ylabel_mode = "phase/2pi"
                else:
                    ylabel_mode = "mag"
                if "IQ" in signal_process_list:
                    xlabel_mode = "IQ"
                    ylabel_mode = "IQ"

                if signal_name == "txtd":
                    x = self.signals_config.t_tx[: self.signals_config.n_samples_tx]
                    sig = self.signals_obj.tx_signal.txtd_base[0, tx_id]
                    title += "TX"
                    if "fft" in signal_process_list:
                        x = self.signals_config.freq_tx
                        xlabel_mode = "freq"
                        title += "-FD"
                    else:
                        x = self.signals_config.t_tx * 1e9
                        xlabel_mode = "time"
                        title += "-TD"
                elif signal_name == "rxtd":
                    sig = rxtd_base[0, rx_id]
                    title += "RX"
                    if "fft" in signal_process_list:
                        x = self.signals_config.freq_trx
                        xlabel_mode = "freq"
                        title += "-FD"
                    else:
                        x = self.signals_config.t_rx[: self.config.plt_n_samples_rx] * 1e9
                        xlabel_mode = "time"
                        title += "-TD"
                elif signal_name == "h":
                    x = self.signals_config.t_trx[: self.signals_config.n_samples_ch] * 1e9
                    sig = h_est[rx_id, tx_id]
                    title += "Channel"
                    if "fft" in signal_process_list:
                        xlabel_mode = "freq"
                        title += "-FD"
                    else:
                        xlabel_mode = "time"
                        title += "-TD"
                elif signal_name == "H":
                    x = self.signals_config.freq_trx[
                        (
                            self.signals_config.sc_range_ch[0]
                            + self.signals_config.n_samples_trx // 2
                        ) : (
                            self.signals_config.sc_range_ch[1]
                            + self.signals_config.n_samples_trx // 2
                            + 1
                        )
                    ]
                    h_est_freq = fftshift(fft(h_est))
                    sig = h_est_freq[rx_id, tx_id]
                    title += "Channel-FD"
                    if "ifft" in signal_process_list:
                        xlabel_mode = "time"
                        title += "-TD"
                    else:
                        xlabel_mode = "freq"
                        title += "-FD"
                elif signal_name == "h_sparse":
                    sig = sparse_est_params
                    title += "Multipath Channel PDP"
                    xlabel_mode = "time_h_sparse"
                    ylabel_mode = "snr"
                elif signal_name == "rx_ph_diff":
                    sig = self.signals_obj.rx_phase_list[-100:]
                    title += "RX-Phase Diff-TD"
                    xlabel_mode = "id"
                    ylabel_mode = "phase"
                elif signal_name == "aoa_gauge":
                    # Return the last AOA gauge value in radians
                    sig = self.signals_obj.aoa_list[-1]
                    title += "AOA Gauge"
                    xlabel_mode = "aoa_gauge"
                    ylabel_mode = "aoa_gauge"
                elif signal_name == "nf_loc":
                    sig = None
                    title += "Heatmap of TX Location probability in the room"
                    xlabel_mode = "nf_loc"
                    ylabel_mode = "nf_loc"
                else:
                    raise ValueError(f"Unsupported signal name: {signal_name}")

                if sig is not None and x is not None:
                    n_samples_plot = min(len(x), len(sig_final))
                    sig = sig[:n_samples_plot]
                    x = x[:n_samples_plot]
                sig, title_post = self.signals_obj.process_sig(
                    sig, process_list=signal_process_list
                )
                title += title_post
                label = f"RX {rx_id}/TX {tx_id}"
                if "real" in signal_process_list:
                    label += "-Real"
                if "imag" in signal_process_list:
                    label += "-Imag"

                if sig_final is None:
                    sig_final = sig.copy()
                    label_final = label

                if index > 0 and plot[index - 1] in supported_operations:
                    operation = plot[index - 1]
                    if operation == "+":
                        sig_final += sig
                    elif operation == "-":
                        sig_final -= sig
                    elif operation == "*":
                        sig_final *= sig
                    elif operation == "/":
                        sig_final /= sig

                    label_final += operation + label

                if not (len(plot) > index + 1 and plot[index + 1] in supported_operations):
                    sig_final = sig_final[0] if sig_final.ndim != 1 else sig_final
                    plot_signals.append(
                        PlotSignal(
                            signal_name=signal_name,
                            trx_id=[rx_id, tx_id],
                            process_list=signal_process_list,
                            x=x,
                            data=sig_final,
                            label=label_final,
                        )
                    )
                    sig_final = None
                    label_final = None

            title += ", RX/TX: "
            for rx_id, tx_id in zip(rx_ids, tx_ids, strict=False):
                title += f"{rx_id}/{tx_id}-"
            title = title[:-1]

            if xlabel_mode == "time":
                xlabel = "Time (ns)"
            elif xlabel_mode == "freq":
                xlabel = "Frequency (MHz)"
            elif xlabel_mode == "time_h_sparse":
                xlabel = "Time (ns)"
            elif xlabel_mode == "IQ":
                xlabel = "In-phase (I)"
            elif xlabel_mode == "id":
                xlabel = "Experiment ID"
            elif xlabel_mode == "aoa_gauge":
                xlabel = "Angle of Arrival (Deg)"
            elif xlabel_mode == "nf_loc":
                xlabel = "X (m)"

            if ylabel_mode == "mag":
                ylabel = "Magnitude"
            elif ylabel_mode == "mag_db":
                ylabel = "Magnitude (dB)"
            elif ylabel_mode == "phase":
                ylabel = "Phase (rad)"
            elif ylabel_mode == "phase/2pi":
                ylabel = "Phase (2π)"
            elif ylabel_mode == "IQ":
                ylabel = "Quadrature (Q)"
            elif ylabel_mode == "snr":
                ylabel = "SNR (dB)"
            elif ylabel_mode == "aoa_gauge":
                ylabel = "Angle of Arrival (Deg)"
            elif ylabel_mode == "nf_loc":
                ylabel = "Y (m)"

            signals.append(
                PlotChart(plot_signals=plot_signals, title=title, x_label=xlabel, y_label=ylabel)
            )

        return signals

    def toggle_pause(self, event):
        if event.key == "p":  # Press 'p' to pause/resume
            self.anim_paused = not self.anim_paused

    def update(self, frame, rx_signal: RxSignal = None):
        self.print("Updating plot", thr=5)

        if self.anim_paused:
            return self.line

        if rx_signal is None:
            raise ValueError("rx_signal cannot be None for plot_level >= 0")
        signals = self.process_signals_for_plot(rx_signal)

        for j in range(self.config.n_plots_col):
            line_id = 0
            for i in range(self.config.n_plots_row):
                # j = self.signals_config.fc_id - 1

                for signal in signals[i].plot_signals:
                    signal_name = signal.signal_name
                    rx_id = signal.trx_id[0]
                    tx_id = signal.trx_id[1]
                    signal_data = signal.data
                    signal_process_list = signal.process_list

                    if "IQ" in signal_process_list:
                        self.line[line_id][j].set_offsets(
                            np.column_stack((signal_data.real, signal_data.imag))
                        )
                        line_id += 1
                        margin = max(np.abs(signal_data)) * 0.1
                        self.ax[i][j].set_xlim(
                            min(signal_data.real) - margin, max(signal_data.real) + margin
                        )
                        self.ax[i][j].set_ylim(
                            min(signal_data.imag) - margin, max(signal_data.imag) + margin
                        )
                    elif signal_name == "rx_ph_diff":
                        self.line[line_id][j].set_data(np.arange(len(signal_data)), signal_data)
                        line_id += 1
                    elif signal_name == "aoa_gauge":
                        self.gauge_update_needle(self.ax[i][j], np.rad2deg(signal_data))
                        self.ax[i][j].set_xlim(0, 1)
                        self.ax[i][j].set_ylim(0.5, 1)
                        self.ax[i][j].axis("off")
                    elif signal_name == "h_sparse":
                        h_tr = signal_data.h_tr_mat
                        dly_est = signal_data.dly_est_mat
                        peaks = signal_data.peaks_mat

                        h_tr = h_tr[rx_id, tx_id]
                        dly_est = dly_est[rx_id, tx_id]
                        peaks = peaks[rx_id, tx_id]

                        # Plot the raw response
                        dly = np.arange(self.signals_config.n_samples_ch)
                        dly = dly - self.signals_config.n_samples_ch * (
                            dly > self.signals_config.n_samples_ch / 2
                        )
                        dly = dly / self.signals_config.fs_trx * 1e9
                        chan_pow = self.signals_obj.lin_to_db(np.abs(h_tr), mode="mag")

                        # Roll the response and shift the response
                        rots = self.signals_config.n_samp_ch_sp // 4
                        yshift = np.percentile(chan_pow, 25)
                        chan_powr = np.roll(chan_pow, rots) - yshift
                        dlyr = np.roll(dly, rots)
                        self.line[line_id][j].set_data(
                            dlyr[: self.signals_config.n_samp_ch_sp],
                            chan_powr[: self.signals_config.n_samp_ch_sp],
                        )
                        line_id += 1

                        # Compute the axes
                        ymax = np.max(chan_powr) + 5
                        ymin = -10

                        # Plot the locations of the detected peaks
                        peaks_ = np.abs(peaks) ** 2
                        peaks_ = self.signals_obj.lin_to_db(peaks_, mode="pow") - yshift
                        dly_est = dly_est * 1e9
                        dly_est = dly_est[
                            dly_est <= np.max(dlyr[: self.signals_config.n_samp_ch_sp])
                        ]
                        self.line[line_id][j].set_data(dly_est, peaks_)
                        line_id += 1
                        self.line[line_id][j].set_segments(
                            [[[i, ymin], [i, j]] for i, j in zip(dly_est, peaks_, strict=False)]
                        )
                        line_id += 1
                        self.ax[i][j].set_ylim([ymin, ymax])
                    elif signal_name == "nf_loc":
                        self.signals_obj.nf_model.plot_results(
                            self.ax[i][j],
                            RoomModel=self.signals_obj.RoomModel,
                            plot_type="init_est",
                        )
                    else:
                        self.line[line_id][j].set_ydata(signal_data)
                        line_id += 1

                if signal_name in self.mag_filter_list["signal_name"] or any(
                    item in signal_process_list for item in self.mag_filter_list["process_list"]
                ):
                    sig = (
                        signal_data[0]
                        if len(np.array(signal_data).shape) > 1
                        else signal_data.copy()
                    )
                    y_min = np.percentile(sig, 10)
                    y_max = np.max(sig) + 0.1 * (np.max(sig) - y_min)
                    self.ax[i][j].set_ylim(y_min, y_max)

                elif not (
                    signal_name in self.untouched_plot_list["signal_name"]
                    or any(
                        item in signal_process_list
                        for item in self.untouched_plot_list["process_list"]
                    )
                ):
                    try:
                        self.ax[i][j].relim()
                        self.ax[i][j].autoscale_view()
                    except Exception as e:
                        raise RuntimeError(
                            f"Error in autoscaling axes for signal {signal_name} with process list {signal_process_list}"
                        ) from e

        return self.line

    def update_once(self, rx_signal: RxSignal = None):
        if rx_signal is None:
            raise ValueError("rx_signal cannot be None for plot update")

        if not self.plots_initialized:
            self.init_plots(rx_signal)
        else:
            self.update(frame=0, rx_signal=rx_signal)

        if getattr(self, "fig", None) is not None:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.01)

    def init_plots(self, rx_signal: RxSignal = None):
        self.print("Initializing plots", thr=5)

        if self.config.plot_level < 0:
            return

        if rx_signal is None:
            raise ValueError("rx_signal cannot be None for plot_level >= 0")
        signals = self.process_signals_for_plot(rx_signal)

        # Set up the figure and plot
        self.line = [
            [None for j in range(self.config.n_plots_col)]
            for i in range(3 * self.config.n_plots_row)
        ]
        self.fig, self.ax = plt.subplots(self.config.n_plots_row, self.config.n_plots_col)
        if type(self.ax) is not np.ndarray:
            self.ax = np.array([self.ax])
        if len(self.ax.shape) < 2:
            self.ax = self.ax.reshape(-1, 1)
        self.fig.canvas.mpl_connect("key_press_event", self.toggle_pause)

        for j in range(self.config.n_plots_col):
            line_id = 0
            for i in range(self.config.n_plots_row):
                for signal in signals[i].plot_signals:
                    signal_name = signal.signal_name
                    label = signal.label
                    signal_process_list = signal.process_list
                    signal_data = signal.data
                    x_data = signal.x

                    if "IQ" in signal_process_list:
                        self.line[line_id][j] = self.ax[i][j].scatter(
                            signal_data.real,
                            signal_data.imag,
                            facecolors="none",
                            edgecolors="b",
                            s=10,
                        )
                        line_id += 1
                        self.ax[i][j].axhline(0, color="black", linewidth=0.5)
                        self.ax[i][j].axvline(0, color="black", linewidth=0.5)
                        self.ax[i][j].set_aspect("equal")
                        margin = max(np.abs(signal_data)) * 0.1
                        self.ax[i][j].set_xlim(
                            min(signal_data.real) - margin, max(signal_data.real + margin)
                        )
                        self.ax[i][j].set_ylim(
                            min(signal_data.imag) - margin, max(signal_data.imag + margin)
                        )

                    elif signal_name == "h_sparse":
                        # (h_tr, dly_est, peaks) = signal_data
                        (self.line[line_id][j],) = self.ax[i][j].plot([], [])
                        line_id += 1
                        # (markerline, stemlines, baseline)
                        self.line[line_id][j], self.line[line_id + 1][j], _ = self.ax[i][j].stem(
                            [0], [1], "r-", basefmt="", bottom=-10
                        )
                        line_id += 2

                    elif signal_name == "aoa_gauge":
                        self.draw_half_gauge(self.ax[i][j], min_val=-90, max_val=90)
                        self.gauge_update_needle(self.ax[i][j], 0, min_val=-90, max_val=90)
                        self.ax[i][j].set_xlim(0, 1)
                        self.ax[i][j].set_ylim(0.5, 1)
                        self.ax[i][j].axis("off")

                    elif signal_name == "nf_loc":
                        self.ax[i][j] = self.signals_obj.nf_model.plot_results(
                            self.ax[i][j],
                            RoomModel=self.signals_obj.RoomModel,
                            plot_type="init_est",
                        )
                        self.ax[i][j].set_yticks([])

                        self.ax[i][j].set_xlim(self.signals_config.nf_region[0])
                        self.ax[i][j].set_ylim(self.signals_config.nf_region[1])
                        self.ax[i][j].set_xticks(
                            np.arange(
                                self.signals_config.nf_region[0, 0],
                                self.signals_config.nf_region[0, 1],
                                1.0,
                            )
                        )
                        self.ax[i][j].set_yticks(
                            np.arange(
                                self.signals_config.nf_region[1, 0],
                                self.signals_config.nf_region[1, 1],
                                2.0,
                            )
                        )

                    else:
                        (self.line[line_id][j],) = self.ax[i][j].plot(
                            x_data, signal_data, label=label
                        )
                        line_id += 1

                # Truncate the title to a maximum of 30 characters
                title = (
                    (signals[i].title[: self.config.plot_configs["title_max_chars"]] + "...")
                    if len(signals[i].title) > self.config.plot_configs["title_max_chars"]
                    else signals[i].title
                )
                title = (
                    title
                    + f"\n Carrier Frequency: {self.signals_config.freq_hop_list[j] / 1e9} GHz"
                )
                x_label = signals[i].x_label
                y_label = signals[i].y_label
                self.ax[i][j].set_title(title)
                self.ax[i][j].set_xlabel(x_label)
                self.ax[i][j].set_ylabel(y_label)

                self.ax[i][j].title.set_fontsize(self.config.plot_configs["title_size"])
                self.ax[i][j].xaxis.label.set_fontsize(self.config.plot_configs["xaxis_size"])
                self.ax[i][j].yaxis.label.set_fontsize(self.config.plot_configs["yaxis_size"])
                self.ax[i][j].tick_params(
                    axis="both", which="major", labelsize=self.config.plot_configs["ticks_size"]
                )  # For major ticks
                self.ax[i][j].legend(fontsize=self.config.plot_configs["legend_size"])

                self.ax[i][j].grid(True)
                if not (
                    signal_name in self.untouched_plot_list["signal_name"]
                    or any(
                        item in signal_process_list
                        for item in self.untouched_plot_list["process_list"]
                    )
                ):
                    self.ax[i][j].relim()
                    self.ax[i][j].autoscale_view()
                self.ax[i][j].minorticks_on()

        for j in range(self.config.n_plots_col):
            for i in range(len(self.line)):
                if self.line[i][j] is not None:
                    # self.line[i][j].set_linewidth(3.0-0.5*self.n_plots_row-0.3*self.n_plots_col)
                    self.line[i][j].set_linewidth(self.config.plot_configs["line_width"])

        # Render once and keep the figure open for manual updates
        plt.tight_layout()
        plt.subplots_adjust(
            hspace=self.config.plot_configs["hspace"], wspace=self.config.plot_configs["wspace"]
        )
        self.plots_initialized = True
        # anim = animation.FuncAnimation(self.fig, self.update, frames=int(1e9), interval=self.config.plot_configs['anim_interval'], blit=False)
        plt.ion()
        plt.show(block=False)
        # self.fig.savefig(self.config.plot_configs['figs_save_path'], dpi=300)


if __name__ == "__main__":
    from python.sounder_configs import Configs_Class

    config = Configs_Class()

    signals_inst = SignalUtilsRfsoc(config)
    # signals_inst.collect_signals()
    signals_inst.compute_sys_response()
