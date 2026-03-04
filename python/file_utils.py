import os
import shutil
from dataclasses import dataclass

import nbformat
from sigcom_toolkit.general import General, GeneralConfig
from tcp_comm import ScpCom, ScpComConfig


@dataclass
class FileUtilsConfig(GeneralConfig):
    scp_connect: bool = False
    host_ip: str = "192.168.3.100"
    username: str = "root"
    password: str = "root"
    host_files_base_addr: str = "~/RFSoC_SDR/python/"
    local_base_addr: str = "./"
    files_to_download: tuple = None
    configs_to_modify: dict = None
    files_to_convert: dict = None


class FileUtils(General):
    def __init__(self, config: FileUtilsConfig, **overrides):
        super().__init__(config, **overrides)

        if self.config.scp_connect:
            self.scp_client = ScpCom(
                config=ScpComConfig(
                    host_ip=self.config.host_ip,
                    username=self.config.username,
                    password=self.config.password,
                )
            )
        else:
            self.scp_client = None

    # Modify a parameter in the Python script
    def modify_text_file(self, file_path, param_name, new_value):
        """
        Modifies the value of a specified parameter in a text file.
        This method reads the content of the given file, searches for the specified
        parameter, and updates its value to the new value provided. The parameter
        is expected to be in the format 'param_name = value'.
        Args:
            file_path (str): The path to the text file to be modified.
            param_name (str): The name of the parameter to be updated.
            new_value (Any): The new value to set for the parameter.
        Returns:
            None
        Raises:
            IOError: If the file cannot be read or written.
        """

        changed = False

        with open(file_path) as file:
            lines = file.readlines()
        with open(file_path, "w") as file:
            for line in lines:
                if param_name in line and "=" in line:
                    old_value = line.split("=")[1].strip()
                    if old_value != repr(new_value):
                        line = f"{param_name} = {repr(new_value)}\n"
                        changed = True
                        self.print(
                            f"Parameter '{param_name}' updated to '{new_value}' in {file_path}.",
                            thr=3,
                        )
                    else:
                        changed = False
                        self.print(
                            f"Parameter '{param_name}' already set to '{new_value}' in {file_path}.",
                            thr=5,
                        )

                file.write(line)

        return changed

    # Convert .py to .ipynb
    def convert_file_format(self, file_1_path, file_2_path):
        """
        Converts a Python script file to a Jupyter notebook file.
        This method reads the content of a Python script file specified by `file_1_path`,
        creates a new Jupyter notebook with the script content as a code cell, and writes
        the notebook to a file specified by `file_2_path`.
        Args:
            file_1_path (str): The path to the input Python script file.
            file_2_path (str): The path to the output Jupyter notebook file.
        Returns:
            None
        """

        with open(file_1_path) as file:
            code = file.read()
        notebook = nbformat.v4.new_notebook()
        notebook.cells.append(nbformat.v4.new_code_cell(code))
        with open(file_2_path, "w") as file:
            nbformat.write(notebook, file)
        self.print(f"Converted {file_1_path} to {file_2_path}.", thr=3)

    def sync_directories(self, base_dir_1, base_dir_2):
        """Compare and sync files from base_dir_1 to base_dir_2."""
        changed_files = []
        for root, _, files in os.walk(base_dir_1):
            for file in files:
                path1 = os.path.join(root, file)
                rel_path = os.path.relpath(path1, base_dir_1)
                path2 = os.path.join(base_dir_2, rel_path)

                # Ensure target directory exists
                os.makedirs(os.path.dirname(path2), exist_ok=True)

                if os.path.exists(path2):
                    hash1 = self.compute_hash(path1)
                    hash2 = self.compute_hash(path2)
                    if hash1 != hash2:
                        shutil.copy2(path1, path2)
                        changed_files.append(path2)
                        self.print(f"Overwritten: {path2}", thr=0)
                else:
                    shutil.copy2(path1, path2)
                    changed_files.append(path2)
                    self.print(f"Copied new file: {path2}", thr=0)

                os.remove(path1)
                self.print(f"Deleted (same content): {path1}", thr=5)

        return changed_files

    def download_files(self):

        if self.scp_client is None:
            raise RuntimeError("SCP client is not initialized. Set scp_connect=True in config.")

        if not self.config.files_to_download:
            raise ValueError("files_to_download is empty; nothing to download.")

        # Ensure the local directory exists
        if not os.path.exists(self.config.local_base_addr):
            self.print(
                f"Local directory {self.config.local_base_addr} does not exist. Creating it.", thr=0
            )
            os.makedirs(self.config.local_base_addr, exist_ok=True)

        # self.files_to_download_ = [os.path.join(host_files_base_addr, file) for file in files_to_download]
        self.files_to_download_ = self.config.files_to_download.copy()

        # self.download_files(files_to_download_, local_base_addr)
        temp_dir = "/tmp/rfsoc/"
        os.makedirs(temp_dir, exist_ok=True)
        self.scp_client.download_files_with_pattern(
            self.config.host_files_base_addr, self.files_to_download_, temp_dir
        )
        self.modify_files(base_dir=temp_dir)
        self.changed_files = self.sync_directories(temp_dir, self.config.local_base_addr)
        for file in self.config.configs_to_modify:
            if file in self.changed_files:
                self.changed_files.remove(file)
        changed = len(self.changed_files) > 0

        return changed

    def modify_files(self, base_dir=None):
        if base_dir is None:
            base_dir = self.config.local_base_addr
        changed = False
        for file in self.config.configs_to_modify:
            local_script_path = os.path.join(base_dir, file)
            for param in self.config.configs_to_modify[file]:
                result = self.modify_text_file(
                    local_script_path, param, self.config.configs_to_modify[file][param]
                )
                if result:
                    changed = True
        return changed

    def convert_files(self):
        changed = False
        for file in self.config.files_to_convert:
            file_1 = os.path.join(self.config.local_base_addr, file)
            file_2 = os.path.join(self.config.local_base_addr, self.config.files_to_convert[file])
            if file_1 in self.changed_files:
                self.convert_file_format(file_1, file_2)
                changed = True
        return changed

