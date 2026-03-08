import os
import platform
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from file_utils import FileUtils, FileUtilsConfig
from sigcom_toolkit.general import General
from signal_utils_rfsoc import ExperimentOperator, ExperimentOperatorConfig


@dataclass(kw_only=True)
class SounderConfig(ExperimentOperatorConfig):
    # Board and RFSoC FPGA project parameters
    transmit_signal: bool = False  # If True, sends TX signal

    # Mixer parameters
    rfsoc_mixer_mode: str = "analog"  # Mixer mode, analog or digital

    # RFFE and antennas parameters
    RFFE: str = "piradio"  # RF front end to use, piradio or sivers
    n_tx_ant: int = 2  # Number of transmitter antennas
    n_rx_ant: int = 2  # Number of receiver antennas

    # Connections parameters
    network_topology: dict = field(default_factory=lambda: {
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
    })

    # File transfer parameters
    update_rfsoc_files: bool = False  # If True, updates the RFSoC files
    modify_rfsoc_files: bool = True  # If True, modifies the RFSoC files to be true for the server mode
    files_dwnld_target: str = "rfsoc"  # Target for file download, rfsoc or raspi
    host_files_base_addr: str = "~/RFSoC_SDR/python/"  # Base address for the host files
    host_ip: str = "192.168.3.100"  # Host IP address
    host_username: str = "root"  # Host username
    host_password: str = "root"  # Host password
    local_base_addr: str = "./"  # Local base address for the files

    # Signals information
    fs: float = 245.76e6 * 4  # Sampling frequency in RFSoC
    n_samples: int = 1024  # Number of samples

    # Save parameters
    save_parameters: bool = True  # If True, saves current parameters
    load_parameters: bool = False  # If True, loads parameters from the file

    # Action parameters
    config_dir: str = os.path.join(os.getcwd(), "config/")  # Configuration directory

    overwrite_level: bool = True

    def detect_running_platform(self):
        system_info = platform.uname()
        if "pynq" in system_info.node.lower():
            self.running_platform = "rfsoc"
        else:
            self.running_platform = "host"

    def __post_init__(self):
        super().__post_init__()

        self.detect_running_platform()

        self.config_path = os.path.join(self.config_dir, "config.json")  # Configuration load path
        self.config_save_path = os.path.join(
            self.config_dir, "config.json"
        )  # Configuration save path

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


class Sounder(General):
    def __init__(self, config: SounderConfig, **overrides: Any):
        super().__init__(config, **overrides)

        self.create_dirs([self.config.config_dir])

        operator_config = ExperimentOperatorConfig().update_from_config(self.config)
        self.operator = ExperimentOperator(operator_config)

        if self.config.save_parameters:
            self.save_class_attributes_to_json(self.config, self.config.config_save_path)
            self.config.save_parameters = False
        if self.config.load_parameters:
            self.load_class_attributes_from_json(self.config, self.config.config_path)

        self.print(f"Running the code as {self.config.host_role}", thr=1)

        if self.config.running_platform == "rfsoc" and (
            self.config.update_rfsoc_files or self.config.modify_rfsoc_files
        ):
            self.update_rfsoc_files()

        self.operator.gen_tx_signal()

    def update_rfsoc_files(self):
        file_utils_config = FileUtilsConfig(
            scp_connect=self.config.update_rfsoc_files
        ).update_from_config(self.config)
        file_utils = FileUtils(file_utils_config)
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
        from rfsoc import RFSoC, RFSoCConfig

        rfsoc_config = RFSoCConfig().update_from_config(self.config)
        rfsoc_config.update_from_config(self.operator.config)
        rfsoc_inst = RFSoC(rfsoc_config)
        rfsoc_inst.txtd = self.operator.tx_signal.txtd
        if self.config.transmit_signal:
            rfsoc_inst.send_frame(rfsoc_inst.txtd)

        # Receiving a test frame to verify connection
        rfsoc_inst.recv_frame_once(n_frame=self.config.n_frame_rd)
        rfsoc_inst.run_tcp()

    def run(self):

        if self.config.running_platform == "rfsoc":
            self.run_rfsoc()

        elif "client" in self.config.host_role:
            self.operator.init_objects()
            self.operator.run_operator()
