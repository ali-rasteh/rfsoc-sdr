from backend import *
from backend import be_np as np, be_scp as scipy
try:
    from rfsoc import RFSoC
except Exception as e:
    print("Error importing RFSoC class: ", e)
from params import Params_Class
from signal_utilsrfsoc import Signal_Utils_Rfsoc, Animate_Plot
from tcp_comm import Tcp_Comm_RFSoC, Tcp_Comm_LinTrack, REST_Com_Piradio, Tcp_Comm_Controller
from serial_comm import Serial_Comm_TurnTable
from file_utils import File_Utils



def rfsoc_run(params):
    
    if params.mode=='server' and (params.update_rfsoc_files or params.modify_rfsoc_files):
        file_utils = File_Utils(params, scp_connect=params.update_rfsoc_files)
        changed_1 = False
        changed_2 = False
        changed_3 = False

        if params.update_rfsoc_files:
            changed_1 = file_utils.download_files()
        if params.update_rfsoc_files or params.modify_rfsoc_files:
            changed_2 = file_utils.modify_files()
        if params.update_rfsoc_files:
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




    signals_inst = Signal_Utils_Rfsoc(params)
    if params.save_parameters:
        params.save_parameters = False
        signals_inst.save_class_attributes_to_json(params, params.params_save_path)
    if params.load_parameters:
        signals_inst.load_class_attributes_from_json(params, params.params_path)
        params.calc_params()

    signals_inst.print("Running the code in mode {}".format(params.mode), thr=1)
    (txtd_base, txtd) = signals_inst.gen_tx_signal()
    # TODO
    signals_inst.txtd_base = txtd_base
    signals_inst.txtd = txtd


    if params.mode=='server':
        rfsoc_inst = RFSoC(params)
        rfsoc_inst.txtd = txtd
        if params.send_signal:
            rfsoc_inst.send_frame(txtd)

        rfsoc_inst.recv_frame_one(n_frame=params.n_frame_rd)
        signals_inst.rx_operations(txtd_base, rfsoc_inst.rxtd)
        if params.run_tcp_server:
            rfsoc_inst.run_tcp()


    params.show_saved_sigs=len(params.saved_sig_plot)>0
    if 'client' in params.mode and not params.show_saved_sigs:

        if 'channel' in params.save_list or 'signal' in params.save_list:
            signals_inst.save_signal_channel(txtd_base, save_list=params.save_list)

        signals_inst.operator()



    if 'client' in params.mode and not 'slave' in params.mode:
        # signals_inst.animate_plot(txtd_base, plot_mode=params.animate_plot_mode, plot_level=0)
        animate_plot_inst = Animate_Plot(params, signals_inst, txtd_base)
        animate_plot_inst.init_objects(txtd_base=txtd_base)
        animate_plot_inst.init_plots()


def create_objects(params):
    signals_inst = Signal_Utils_Rfsoc(params)
    return signals_inst


if __name__ == '__main__':
    
    params = Params_Class()
    rfsoc_run(params)

