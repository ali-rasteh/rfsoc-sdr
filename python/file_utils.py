"""
This script is used to copy and modify files from a remote host to a local directory.
It supports two targets: 'rfsoc' and 'raspi', each with its own set of files and parameters to modify.
Functions:
    main(config):
        Main function to handle the file copying and modification process.
        Args:
            config (Configs_Class): An instance of Configs_Class containing configuration parameters.
Usage:
    Run this script directly to copy and modify files based on the specified target.
    You can modify parameters at the beginning of the main function to customize the behavior.
    Example:
        python copy_files.py
"""

import os
from dataclasses import dataclass

from configs import Configs_Class
from tcp_comm import Scp_Com, ScpComConfig
from sigcom_toolkit.general import General, GeneralConfig



@dataclass
class FileUtilsConfig(GeneralConfig):
    scp_connect: bool = False
    host_ip: str = '192.168.3.100'
    username: str = 'root'
    password: str = 'root'
    host_files_base_addr: str = '~/RFSoC_SDR/python/'
    local_base_addr: str = './'
    files_to_download: list = None
    configs_to_modify: dict = None
    files_to_convert: dict = None


class File_Utils(General):

    def __init__(self, config: FileUtilsConfig, **overrides):
        super().__init__(config, **overrides)

        if self.config.scp_connect:
            self.scp_client = Scp_Com(config=ScpComConfig(
                host_ip=self.config.host_ip,
                username=self.config.username,
                password=self.config.password
            ))
        else:
            self.scp_client = None



    def download_files(self):

        # Ensure the local directory exists
        if not os.path.exists(self.local_base_addr):
            self.print(f"Local directory {self.local_base_addr} does not exist. Creating it.", thr=0)
            os.makedirs(self.local_base_addr, exist_ok=True)

        # self.files_to_download_ = [os.path.join(host_files_base_addr, file) for file in files_to_download]
        self.files_to_download_ = self.files_to_download.copy()

        # self.download_files(files_to_download_, local_base_addr)
        temp_dir = "/tmp/rfsoc/"
        os.makedirs(temp_dir, exist_ok=True)
        self.scp_client.download_files_with_pattern(self.host_files_base_addr, self.files_to_download_, temp_dir)
        self.modify_files(base_dir=temp_dir)
        self.changed_files = self.sync_directories(temp_dir, self.local_base_addr)
        for file in self.configs_to_modify:
            if file in self.changed_files:
                self.changed_files.remove(file)
        changed = (len(self.changed_files) > 0)
        
        return changed


    def modify_files(self, base_dir=None):
        if base_dir is None:
            base_dir = self.local_base_addr
        changed = False
        for file in self.configs_to_modify:
            local_script_path = os.path.join(base_dir, file)
            for param in self.configs_to_modify[file]:
                result = self.modify_text_file(local_script_path, param, self.configs_to_modify[file][param])
                if result:
                    changed = True
        return changed


    def convert_files(self):
        changed = False
        for file in self.files_to_convert:
            file_1 = os.path.join(self.local_base_addr, file)
            file_2 = os.path.join(self.local_base_addr, self.files_to_convert[file])
            if file_1 in self.changed_files:
                self.convert_file_format(file_1, file_2)
                changed = True
        return changed



if __name__ == "__main__":

    config = Configs_Class()
    file_utils = File_Utils(config=config)
    file_utils.download_files()
    file_utils.modify_files()


