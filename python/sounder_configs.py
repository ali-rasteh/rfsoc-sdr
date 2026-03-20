from dataclasses import dataclass, field

from sounder import SounderConfig


@dataclass(kw_only=True)
class PlotSymbols:
    h00: str = "h|0|0|circshift|mag|dbmag"
    h01: str = "h|0|1|circshift|mag|dbmag"
    h10: str = "h|1|0|circshift|mag|dbmag"
    h11: str = "h|1|1|circshift|mag|dbmag"
    txfd00: str = "txtd|0|0|fft|fftshift|mag|dbmag"
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
    save_format: str = "npz"

    animate_plot_mode: tuple = (
        [PlotSymbols.rxtd00_r, PlotSymbols.rxtd00_i],
        [PlotSymbols.rxfd00],
        # [PlotSymbols.txfd00],
        # [PlotSymbols.h00],
        # PlotSymbols.aoa_gauge,
    )

@dataclass(kw_only=True)
class FR3SpectrumSweepConfig(BaseConfig):
    n_frame_rd: int = 1
    rx_chain: tuple = ("sync_time",)
    tx_sig_sim: str = "same"
    sig_gen_mode: str = "fft"
    sig_mode: str = "wideband"
    sig_modulation: str = "4qam"
    network_topology: dict = field(default_factory=lambda: {
        "rfsoc_trx": {"type": "rfsoc", "role": "tx", "ip": "192.168.185.4"},
        "gimbal_tx": {"type": "D48PTU", "port": "/dev/ttyUSB0"},
        "piradio_rx": {
            "type": "piradio",
            "role": "rx",
            "ip": "192.168.185.51",
        },
        "piradio_tx": {
            "type": "piradio",
            "role": "tx",
            "ip": "192.168.137.51",
        },
    })
        # {"targets": ["piradio_rx"],    "actions": ["set_gain_db_rx"], "values": [3,7,10,17]},
    action_loop: tuple = (
        {"targets": ["piradio_rx", "piradio_tx"],    "actions": ["hop_freq"], "values": [10.0e9]},
        {"targets": ["piradio_tx"],     "actions": ["set_gain_db_tx"], "values": [35.0]},
        {"targets": ["rfsoc_trx"],      "actions": ["transmit_signal"]},
        {"targets": ["self"],           "actions": ["switch_sig_size"], "values": [8, 16, 64, 256]},
        {"targets": ["gimbal_tx"],      "actions": ["set_gimbal_el"], "values": "-20:0.0:3"},
        {"targets": ["gimbal_tx"],      "actions": ["set_gimbal_az"], "values": "-45:45:2"},
        {"targets": ["piradio_rx"],     "actions": ["set_gain_db_rx"], "values": "15:38:10"},
        # {"targets": ["self"],           "actions": ["switch_sig_size"], "values": "2:256:10:log"},
        {"targets": ["rfsoc_trx"],      "actions": ["switch_sig_ss"], "values": "1:100:5"},
        # {"targets": ["self"],           "actions": ["wait"], "values": [0.01]},
        {"targets": ["rfsoc_trx"],      "actions": ["capture"], "values": [2],
                                        "params": {"process_signal": False}},
        {"targets": ["self"],           "actions": ["update_plot"], "values": [1]},
        # {"targets": ["self"],           "actions": ["print"], "values": [1],
        #                                   "params": {"print_list": ["snr"]}},
        {"targets": ["self"],           "actions": ["save", "store"], "values": [1],
                                        "params": {"save_list": ["signal", "sig_interval"]}},
    )

@dataclass(kw_only=True)
class FR3RoboticLocalizationConfig(BaseConfig):
    role = "master"
    ant_d_m: tuple = (0.026,)
    wb_sc_range: tuple = (-260, 260)
    n_frame_rd: int = 1
    rx_chain: tuple = ()
    tx_sig_sim: str = "same"
    sig_gen_mode: str = "ZadoffChu"
    sig_mode: str = "wideband"

    def __post_init__(self):
        super().__post_init__()
        if self.role == "master":
            self.network_topology: dict = {
                "rfsoc_rx": {"type": "rfsoc", "role": "rx", "ip": "192.168.185.4"},
                "piradio_rx": {
                    "type": "piradio",
                    "role": "rx",
                    "ip": "192.168.185.51",
                },
                "controller_tx": {"type": "controller_client", "ip": "10.20.47.103"},
                "turtlebot_rx": {"type": "turtlebot"},
            }
        else:
            self.network_topology: dict = {
                "rfsoc_tx": {"type": "rfsoc", "role": "tx", "ip": "192.168.3.1"},
                "piradio_tx": {
                    "type": "piradio",
                    "role": "tx",
                    "ip": "192.168.137.51",
                },
                "lintrack_tx": {"type": "lintrack"},
                "gimbal_tx": {"type": "D48PTU", "port": "/dev/ttyUSB0"},
                "controller_tx": {"type": "controller_server"},
            }
        self.action_loop: tuple = (
            # {"targets": ["rfsoc_rx"],           "actions": ["calibrate_rfsoc"], "values": [1]},
            {"targets": ["piradio_rx", "controller_tx"],    "actions": ["hop_freq"], "values": [10.0e9]},
            {"targets": ["controller_tx"],      "actions": ["set_gain_db_tx"], "values": [25.0]},
            {"targets": ["piradio_rx"],         "actions": ["set_gain_db_rx"], "values": [25.0]},
            {"targets": ["controller_tx"],      "actions": ["transmit_signal"]},
            # {"targets": ["turtlebot_rx"],       "actions": ["move_turtlebot"], "values": "1:1000:1000"},
            # {"targets": ["turtlebot_rx", "controller_tx"],      "actions": ["move_lintrack_trurtlebot"], "values": "1:20:20"},
            # {"targets": ["turtlebot_rx", "controller_tx"],      "actions": ["move_gimbal_trurtlebot"], "values": [1]},
            {"targets": ["self"],               "actions": ["loop"], "values": "1:100:100"},
            {"targets": ["rfsoc_rx"],           "actions": ["capture"], "values": [2],
                                                "params": {"process_signal": False}},
            {"targets": ["self"],               "actions": ["update_plot"], "values": [1]},
            # {"targets": ["self"],               "actions": ["save", "store"], "values": [1],
            #                                 "params": {"save_list": ["signal", "snr_db", "aoa", "turtlebot_info"]}},
        )
