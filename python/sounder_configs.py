from dataclasses import dataclass, field

from sounder import SounderConfig


@dataclass(kw_only=True)
class PlotSymbols:
    h00: str = "h|0|0|circshift|mag|dbmag"
    h01: str = "h|0|1|circshift|mag|dbmag"
    h10: str = "h|1|0|circshift|mag|dbmag"
    h11: str = "h|1|1|circshift|mag|dbmag"
    rxfd00: str = "rxtd|0|0|fft|fftshift|mag|dbmag"
    rxfd10: str = "rxtd|1|0|fft|fftshift|mag|dbmag"
    rxfd00_ph: str = "rxtd|0|0|fft|fftshift|phase/2pi"
    rxfd10_ph: str = "rxtd|1|0|fft|fftshift|phase/2pi"
    rxfd_ph_diff: tuple = ("rxfd00_ph", "-", "rxfd10_ph")
    rxtd00: str = "rxtd|0|0|mag"
    rxtd10: str = "rxtd|1|0|mag"
    rxtd00_ph: str = "rxtd|0|0|phase/2pi"
    rxtd10_ph: str = "rxtd|1|0|phase/2pi"
    rxtd00_r: str = "rxtd|0|0|real"
    rxtd00_i: str = "rxtd|0|0|imag"
    rxtd10_r: str = "rxtd|1|0|real"
    rxtd10_i: str = "rxtd|1|0|imag"
    rxtd_ph_diff: tuple = ("rxtd00_ph", "-", "rxtd10_ph")
    IQ00: tuple = ("rxtd|0|0|fft|fftshift|IQ",)
    aoa_gauge: tuple = ("aoa_gauge|0|0",)


@dataclass(kw_only=True)
class BaseConfig(SounderConfig):
    ant_d_m: tuple = (0.026,)
    wb_sc_range: tuple = (-260, 260)
    rx_same_delay: bool = False
    n_frame_rd: int = 32
    n_rd_rep: int = 1
    rx_chain: tuple = ("sync_time", "channel_est")

    plot_configs: dict = field(default_factory=lambda: {
            "title_size": 11,
            "title_max_chars": 35,
            "xaxis_size": 10,
            "yaxis_size": 10,
            "ticks_size": 10,
            "legend_size": 10,
            "line_width": 1.0,
            "marker_size": 8,
            "hspace": 0.5,
            "wspace": 0.5,
        })

    update_rfsoc_files: bool = False
    host_files_base_addr: str = ""
    host_ip: str = "192.168.2.1"
    host_username: str = ""
    host_password: str = ""

    overwrite_level: bool = True
    plot_level: int = 0
    verbose_level: int = 0

    host_role: str = "client"

    animate_plot_mode: tuple = (
        [PlotSymbols.h00],
        [PlotSymbols.rxtd00_r, PlotSymbols.rxtd00_i],
        [PlotSymbols.rxfd00],
        PlotSymbols.aoa_gauge,
    )

    n_save: int = 32
    save_format: str = "npz"


@dataclass(kw_only=True)
class PlotSaveConfig(BaseConfig):
    # freq_hop_config["list"] = [6.5e9, 10e9, 15.0e9, 20.0e9]
    tx_sig_sim: str = "shifted"
    sig_gen_mode: str = "ZadoffChu"

@dataclass(kw_only=True)
class MmwDemoConfig(BaseConfig):
    RFFE: str = "sivers"
    # freq_hop_config["list"] = [60.0e9]
    tx_sig_sim: str = "orthogonal"
    sig_gen_mode: str = "ZadoffChu"

@dataclass(kw_only=True)
class RfsocDemoConfig(BaseConfig):
    mix_freq_adc: float = 0.0e6
    do_rfsoc_mixer_settings: bool = False
    tx_sig_sim: str = "same"
    sig_gen_mode: str = "fft"
    sig_modulation: str = "4qam"

@dataclass(kw_only=True)
class FR3SpectrumSweepConfig(BaseConfig):
    tx_sig_sim: str = "same"
    sig_gen_mode: str = "fft"
    sig_mode: str = "wideband"
    sig_modulation: str = "4qam"
    measurement_configs: tuple = None
    network_topology: dict = field(default_factory=lambda: {
        "rfsoc_trx": {"type": "rfsoc", "role": "tx", "ip": "192.168.185.4", "protocol": "tcp"},
        "gimbal": {"type": "D48PTU", "port": "/dev/ttyUSB0"},
        "piradio_trx": {
            "type": "piradio",
            "role": "tx",
            "ip": "192.168.137.51",
            "protocol": "http",
            "username": "ubuntu",
            "password": "temppwd",
        },
    })
    action_loop: tuple = (
        {"targets": ["gimbal"],         "actions": ["set_gimbal_el"], "values": "-20:20:100"},
        {"targets": ["gimbal"],         "actions": ["set_gimbal_az"], "values": "1:10:10"},
        {"targets": ["piradio_trx"],    "actions": ["set_gain_db_rx"], "values": "-3:20:20"},
        {"targets": ["self"],           "actions": ["switch_sig_size"], "values": [8, 16, 32, 128]},
        # {"targets": ["piradio_trx"],    "actions": ["set_gain_db_rx"], "values": [3,7,10,17]},
        # {"targets": ["self"],           "actions": ["switch_sig_size"], "values": "1:256:20:log"},
        {"targets": ["rfsoc_trx"],      "actions": ["switch_sig_ss"], "values": "1:10:10"},
        {"targets": ["self"],           "actions": ["wait"], "values": [0.1]},
        {"targets": ["rfsoc_trx"],      "actions": ["capture"], "values": [256]},
        {"targets": ["self"],           "actions": ["save"], "values": [1],
                                        "params": {"save_list": ["signal"]}},
    )

@dataclass(kw_only=True)
class FR3DemoConfig(BaseConfig):
    # freq_hop_config["list"] = [6.5e9]
    tx_sig_sim: str = "orthogonal"

@dataclass(kw_only=True)
class FR3DemoMultiFreqConfig(BaseConfig):
    # freq_hop_config["list"] = [6.5e9, 8.75e9, 10.0e9]
    tx_sig_sim: str = "orthogonal"

@dataclass(kw_only=True)
class FR3AntCalibConfig(BaseConfig):
    # rotation_range_deg = [-90, 90]
    # rotation_step_deg = 1
    # rotation_delay = 0.5
    # freq_hop_config["mode"] = "sweep"
    # freq_hop_config["range"] = [6.0e9, 22.5e9]
    # freq_hop_config["step"] = 0.5e9
    tx_sig_sim: str = "shifted"
    sig_gen_mode: str = "ZadoffChu"
    measurement_configs: tuple = ("tx1_rx1_rx_rotate", "tx2_rx2_rx_rotate")

@dataclass(kw_only=True)
class FR3BeamFormConfig(BaseConfig):
    # rotation_range_deg = [-90, 90]
    # rotation_step_deg = 2
    # rotation_delay = 0.5
    # freq_hop_config["list"] = [10e9]
    sig_gen_mode: str = "fft"
    tx_sig_sim: str = "same"
    beamforming: bool = True
    steer_rad: tuple = (0, 0)
    def __post_init__(self):
        return super().__post_init__()
        self.measurement_configs: tuple = (6.5, f"bf_phi_{self.steer_rad[0]}")

@dataclass(kw_only=True)
class FR3NYU3StateConfig(BaseConfig):
    # rotation_range_deg = [-45, 45]
    # rotation_step_deg = 45
    # rotation_delay = 0.5
    # freq_hop_config["list"] = [6.5e9, 8.75e9, 10.0e9, 15.0e9, 21.7e9]
    tx_sig_sim: str = "shifted"
    sig_gen_mode: str = "ZadoffChu"

    # Naming: _Position_TX-Orient_RX-Orient_Reflect/NoReflect(r/n)-Blockage/NoBlockage(b/n)
    # Orientations: alpha: 0, beta: 45, gamma: -45
    # Good Pi-radio gains for OTA: 20dB for TX channels and 21dB for RX channels
    # Good Pi-radio gains for cabled calibration: 10dB for TX channels and 15dB for RX channels
    measurement_configs: tuple = (
        "calib_1-1_2-2",
        "calib_1-2_2-1",
        "A_beta_<rxorient>_n",
        "A_alpha_<rxorient>_n",
        "A_gamma_<rxorient>_n",
        "B_alpha_<rxorient>_n",
        "B_gamma_<rxorient>_n",
        "B_beta_<rxorient>_n",
        "C_beta_<rxorient>_n",
        "C_alpha_<rxorient>_n",
        "C_gamma_<rxorient>_n",
        "D_gamma_<rxorient>_n",
        "D_alpha_<rxorient>_n",
        "D_beta_<rxorient>_n",
        "E_beta_<rxorient>_n",
        "E_alpha_<rxorient>_n",
        "E_gamma_<rxorient>_n",
    )

@dataclass(kw_only=True)
class FR3NYU13StateConfig(BaseConfig):
    # rotation_range_deg = [-60, 60]
    # rotation_step_deg = 10
    # rotation_delay = 0.5
    # freq_hop_config["list"] = [6.5e9, 8.75e9, 10.0e9, 15.0e9, 21.7e9]
    tx_sig_sim: str = "shifted"
    sig_gen_mode: str = "ZadoffChu"

    # Naming: _Position_TX-Orient_RX-Orient_Reflect/NoReflect(r/n)-Blockage/NoBlockage(b/n)
    measurement_configs: tuple = (
        "calib_1-1_2-2",
        "calib_1-2_2-1",
        "C_alpha_<rxorient>_n",
        "C_alpha_<rxorient>_r",
        "C_alpha_<rxorient>_b",
    )

@dataclass(kw_only=True)
class FR3CFOConfig(BaseConfig):
    # freq_hop_config["list"] = [10.0e9]
    cfo_ppm: int = -100
    sig_gen_mode: str = "fft"
    tx_sig_sim: str = "orthogonal"
    sig_modulation: str = "4qam"

    def __post_init__(self):
        super().__post_init__()
        if self.host_role == "client_master":
            self.cfo = self.cfo_ppm * self.freq_hop_config["list"][0] / 1e6
            self.mix_freq_adc += self.cfo
            self.do_rfsoc_mixer_settings = True

        self.measurement_configs: tuple = (
            "{}GHz_{}ppm".format(self.freq_hop_config["list"][0] / 1e9, self.cfo_ppm)
        )

@dataclass(kw_only=True)
class TurtlebotDemoConfig(BaseConfig):
    # freq_hop_config["list"] = [10.0e9]
    tx_sig_sim: str = "same"
    sig_gen_mode: str = "ZadoffChu"
