from dataclasses import dataclass

from sounder import SounderConfig


@dataclass
class PlotSymbols:
    h00: str = "h|0|0|circshift|mag|dbmag"
    h01: str = "h|0|1|circshift|mag|dbmag"
    h10: str = "h|1|0|circshift|mag|dbmag"
    h11: str = "h|1|1|circshift|mag|dbmag"
    rxfd00: str = "rxtd|0|0|fft|fftshift|mag|dbmag"
    rxfd10: str = "rxtd|1|0|fft|fftshift|mag|dbmag"
    rxfd00_ph: str = "rxtd|0|0|fft|fftshift|phase/2pi"
    rxfd10_ph: str = "rxtd|1|0|fft|fftshift|phase/2pi"
    rxfd_ph_diff: list = ["rxfd00_ph", "-", "rxfd10_ph"]
    rxtd00: str = "rxtd|0|0|mag"
    rxtd10: str = "rxtd|1|0|mag"
    rxtd00_ph: str = "rxtd|0|0|phase/2pi"
    rxtd10_ph: str = "rxtd|1|0|phase/2pi"
    rxtd00_r: str = "rxtd|0|0|real"
    rxtd00_i: str = "rxtd|0|0|imag"
    rxtd10_r: str = "rxtd|1|0|real"
    rxtd10_i: str = "rxtd|1|0|imag"
    rxtd_ph_diff: list = ["rxtd00_ph", "-", "rxtd10_ph"]
    IQ00: list = ["rxtd|0|0|fft|fftshift|IQ"]
    aoa_gauge: list = ["aoa_gauge|0|0"]


@dataclass
class BaseConfig(SounderConfig):
    ant_d_m: list = ([0.026],)
    n_rx_ch_eq: int = (1,)
    wb_sc_range: list = ([-260, 260],)
    rx_same_delay: bool = (False,)
    n_frame_rd: int = (32,)
    n_rd_rep: int = (1,)
    save_parameters: bool = (True,)
    rx_chain = ["sync_time", "channel_est"]

    plot_configs: dict = (
        {
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
        },
    )

    update_rfsoc_files: bool = (True,)
    host_files_base_addr: str = ("",)
    host_ip: str = ("192.168.2.1",)
    host_username: str = ("",)
    host_password: str = ("",)

    overwrite_level: bool = (False,)
    plot_level: int = (0,)
    verbose_level: int = (0,)

    host_role: str = ("client",)

    animate_plot_mode = []
    animate_plot_mode.append([PlotSymbols.h00])
    animate_plot_mode.append([PlotSymbols.rxtd00_r, PlotSymbols.rxtd00_i])
    animate_plot_mode.append([PlotSymbols.rxfd00])
    animate_plot_mode.append(PlotSymbols.aoa_gauge)

    n_save = 32
    save_format = "npz"


@dataclass
class PlotSaveConfig(BaseConfig):
    freq_hop_config["list"] = [6.5e9, 10e9, 15.0e9, 20.0e9]
    tx_sig_sim = "shifted"
    sig_gen_mode = "ZadoffChu"


class MmwDemoConfig(BaseConfig):
    RFFE = "sivers"
    freq_hop_config["list"] = [60.0e9]
    tx_sig_sim = "orthogonal"
    sig_gen_mode = "ZadoffChu"


class RfsocDemoConfig(BaseConfig):
    mix_freq_adc = 0.0e6
    do_rfsoc_mixer_settings = False
    tx_sig_sim = "same"
    sig_gen_mode = "fft"
    sig_modulation = "4qam"


class FR3SpectrumSweepConfig(BaseConfig):
    tx_sig_sim = "same"
    sig_gen_mode = "fft"
    sig_mode = "wideband"
    sig_modulation = "4qam"
    measurement_configs = []
    network_topology = {
        "rfsoc_tx": {"type": "rfsoc", "role": "tx", "ip": "192.168.3.1", "protocol": "tcp"},
        "rfsoc_rx": {"type": "rfsoc", "role": "rx", "ip": "192.168.3.1", "protocol": "tcp"},
        "lintrack": {"type": "lintrack", "role": "rx", "ip": "192.168.137.100", "protocol": "tcp"},
        "turntable": {
            "type": "turntable",
            "role": "rx",
            "port": "/dev/ttyACM0",
            "baudrate": 115200,
            "protocol": "tcp",
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
    action_loop = [
        "piradio_rx/set_gain_db_rx/-3:20:20/",
        "self/switch_sig_size/[8,16,32,128]/",
        # 'piradio_rx/set_gain_db_rx/[3,7,10,17]/',
        # 'self/switch_sig_size/1:256:20:log/',
        "self/switch_sig_ss/1:10:10/",
        # 'self/wait/[1]/',
        "rfsoc_rx/capture/[10]/",
        'self/save/[1]/["signal"]/m',
    ]


class FR3DemoConfig(BaseConfig):
    freq_hop_config["list"] = [6.5e9]
    tx_sig_sim = "orthogonal"


class FR3DemoMultiFreqConfig(BaseConfig):
    freq_hop_config["list"] = [6.5e9, 8.75e9, 10.0e9]
    tx_sig_sim = "orthogonal"


class FR3AntCalibConfig(BaseConfig):
    rotation_range_deg = [-90, 90]
    rotation_step_deg = 1
    rotation_delay = 0.5
    freq_hop_config["mode"] = "sweep"
    freq_hop_config["range"] = [6.0e9, 22.5e9]
    freq_hop_config["step"] = 0.5e9
    tx_sig_sim = "shifted"
    sig_gen_mode = "ZadoffChu"
    measurement_configs = []
    measurement_configs.append("tx1_rx1_rx_rotate")
    measurement_configs.append("tx2_rx2_rx_rotate")


class FR3BeamFormConfig(BaseConfig):
    rotation_range_deg = [-90, 90]
    rotation_step_deg = 2
    rotation_delay = 0.5
    freq_hop_config["list"] = [10e9]
    sig_gen_mode = "fft"
    tx_sig_sim = "same"
    beamforming = True
    steer_rad = [0, 0]
    measurement_configs = [6.5]
    measurement_configs.append(f"bf_phi_{steer_rad[0]}")


class FR3NYU3StateConfig(BaseConfig):
    rotation_range_deg = [-45, 45]
    rotation_step_deg = 45
    rotation_delay = 0.5
    freq_hop_config["list"] = [6.5e9, 8.75e9, 10.0e9, 15.0e9, 21.7e9]
    tx_sig_sim = "shifted"
    sig_gen_mode = "ZadoffChu"

    # Naming: _Position_TX-Orient_RX-Orient_Reflect/NoReflect(r/n)-Blockage/NoBlockage(b/n)
    # Orientations: alpha: 0, beta: 45, gamma: -45
    # Good Pi-radio gains for OTA: 20dB for TX channels and 21dB for RX channels
    # Good Pi-radio gains for cabled calibration: 10dB for TX channels and 15dB for RX channels
    measurement_configs = [
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
    ]


class FR3NYU13StateConfig(BaseConfig):
    rotation_range_deg = [-60, 60]
    rotation_step_deg = 10
    rotation_delay = 0.5
    freq_hop_config["list"] = [6.5e9, 8.75e9, 10.0e9, 15.0e9, 21.7e9]
    tx_sig_sim = "shifted"
    sig_gen_mode = "ZadoffChu"

    # Naming: _Position_TX-Orient_RX-Orient_Reflect/NoReflect(r/n)-Blockage/NoBlockage(b/n)
    measurement_configs = [
        "calib_1-1_2-2",
        "calib_1-2_2-1",
        "C_alpha_<rxorient>_n",
        "C_alpha_<rxorient>_r",
        "C_alpha_<rxorient>_b",
    ]


class FR3CFOConfig(BaseConfig):
    freq_hop_config["list"] = [10.0e9]
    cfo_ppm = -100
    sig_gen_mode = "fft"
    tx_sig_sim = "orthogonal"
    sig_modulation = "4qam"

    def __post_init__(self):
        super().__post_init__()
        if self.host_role == "client_master":
            self.cfo = self.cfo_ppm * self.freq_hop_config["list"][0] / 1e6
            self.mix_freq_adc += self.cfo
            self.do_rfsoc_mixer_settings = True

        self.measurement_configs = [
            "{}GHz_{}ppm".format(self.freq_hop_config["list"][0] / 1e9, self.cfo_ppm)
        ]


class TurtlebotDemoConfig(BaseConfig):
    freq_hop_config["list"] = [10.0e9]
    tx_sig_sim = "same"
    sig_gen_mode = "ZadoffChu"
