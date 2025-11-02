"""
This script is used to copy and modify files from a remote host to a local directory.
It supports two targets: 'rfsoc' and 'raspi', each with its own set of files and parameters to modify.
Functions:
    main(params):
        Main function to handle the file copying and modification process.
        Args:
            params (Params_Class): An instance of Params_Class containing configuration parameters.
Usage:
    Run this script directly to copy and modify files based on the specified target.
    You can modify parameters at the beginning of the main function to customize the behavior.
    Example:
        python copy_files.py
"""

from backend import *
from backend import be_np as np, be_scp as scipy
from params import Params_Class
from tcp_comm import Scp_Com
from sigcom-toolkit.general import General





class File_Utils(General):

    def __init__(self, params, scp_connect=False):

        super().__init__(params)


        if scp_connect:
            params = params.copy()
            params.username = getattr(params, 'host_username', 'root')
            params.password = getattr(params, 'host_password', 'root')
            # params.host_ip = getattr(params, 'host_ip', '192.168.3.100')
            self.scp_client = Scp_Com(params)
        else:
            self.scp_client = None


        self.host_files_base_addr = getattr(params, 'host_files_base_addr', '~/RFSoC_SDR/python/')
        self.local_base_addr = getattr(params, 'local_base_addr', './')

        self.files_to_download = getattr(params, 'files_to_download', None)
        self.params_to_modify = getattr(params, 'params_to_modify', None)
        self.files_to_convert = getattr(params, 'files_to_convert', None)



    def download_files(self):

        # Ensure the local directory exists
        if not os.path.exists(self.local_base_addr):
            self.print(f"Local directory {self.local_base_addr} does not exist. Creating it.", thr=0)
            os.makedirs(self.local_base_addr, exist_ok=True)

        # self.files_to_download_ = [os.path.join(host_files_base_addr, file) for file in files_to_download]
        self.files_to_download_ = self.files_to_download.copy()


        # for pattern in self.files_to_download:
        #     local_files = glob.glob(os.path.join(self.local_base_addr, pattern))
        #     # file = file.split('/')[-1]
        #     for file in local_files:
        #         if os.path.exists(file):
        #             os.remove(file)
        #             self.print(f"Deleted: {file}", thr=1)

        # for item in os.listdir(self.local_base_addr):
        #     item_path = os.path.join(self.local_base_addr, item)
        #     try:
        #         # Remove directories
        #         if os.path.isdir(item_path):
        #             shutil.rmtree(item_path)
        #         # Remove files
        #         elif os.path.isfile(item_path):
        #             continue
        #             # os.remove(item_path)
        #         elif os.path.islink(item_path):
        #             os.unlink(item_path)
        #         self.print(f"Deleted: {item_path}", thr=1)
        #     except Exception as e:
        #         self.print(f"Error deleting {item_path}: {e}", thr=0)


        # self.download_files(files_to_download_, local_base_addr)
        temp_dir = "/tmp/rfsoc/"
        os.makedirs(temp_dir, exist_ok=True)
        self.scp_client.download_files_with_pattern(self.host_files_base_addr, self.files_to_download_, temp_dir)
        self.modify_files(base_dir=temp_dir)
        self.changed_files = self.sync_directories(temp_dir, self.local_base_addr)
        for file in self.params_to_modify:
            if file in self.changed_files:
                self.changed_files.remove(file)
        changed = (len(self.changed_files) > 0)
        
        return changed



    def modify_files(self, base_dir=None):
        if base_dir is None:
            base_dir = self.local_base_addr
        changed = False
        for file in self.params_to_modify:
            local_script_path = os.path.join(base_dir, file)
            for param in self.params_to_modify[file]:
                result = self.modify_text_file(local_script_path, param, self.params_to_modify[file][param])
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

    params = Params_Class()
    file_utils = File_Utils(params)
    file_utils.download_files()
    file_utils.modify_files()




