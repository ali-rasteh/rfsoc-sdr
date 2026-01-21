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


    if params.mode=='server':
        rfsoc_inst = RFSoC(params)
        rfsoc_inst.txtd = txtd
        if params.send_signal:
            rfsoc_inst.send_frame(txtd)
        if params.recv_signal:
            rfsoc_inst.recv_frame_one(n_frame=params.n_frame_rd)
            signals_inst.rx_operations(txtd_base, rfsoc_inst.rxtd)
        if params.run_tcp_server:
            rfsoc_inst.run_tcp()



    client_rfsoc = None
    client_lintrack = None
    client_turntable = None
    client_piradio = None
    client_controller = None
    
    params.show_saved_sigs=len(params.saved_sig_plot)>0
    if 'client' in params.mode and not params.show_saved_sigs:

        if params.use_linear_track:
            client_lintrack = Tcp_Comm_LinTrack(params)
            client_lintrack.init_tcp_client()
            # client_lintrack.return2home()
            # client_lintrack.go2end()

        if params.use_turntable:
            client_turntable = Serial_Comm_TurnTable(params)
            try:
                client_turntable.connect()
                client_turntable.move_to_position(0)
                if params.calibrate_turntable:
                    client_turntable.calibrate()
                client_turntable.interactive_move()
            except:
                client_turntable.list_ports()
                raise Exception("Turntable not connected or wrong port, please check the port list")

        if params.control_piradio:
            # client_piradio = ssh_Com_Piradio(params)
            # client_piradio.init_ssh_client()
            # client_piradio.initialize()
            client_piradio = REST_Com_Piradio(params)
            client_piradio.set_frequency(fc=params.fc)

        if 'master' in params.mode:
            client_controller = Tcp_Comm_Controller(params)
            client_controller.init_tcp_client()
            client_controller.set_frequency_piradio(params.fc)
        elif 'slave' in params.mode:
            controller = Tcp_Comm_Controller(params)
            controller.init_tcp_server()
            controller.obj_piradio = client_piradio
            controller.obj_rfsoc = client_rfsoc
            controller.run_tcp_server(controller.parse_and_execute)


        if params.control_rfsoc:
            client_rfsoc=Tcp_Comm_RFSoC(params)
            client_rfsoc.init_tcp_client()

            if params.send_signal:
                # client_rfsoc.transmit_data_default()
                client_rfsoc.transmit_data(txtd)
                pass


            client_rfsoc.set_frequency_mixer(params.mix_freq_dac, params.mix_freq_adc)
            if params.RFFE=='sivers':
                client_rfsoc.set_frequency_sivers(params.fc)
                if params.send_signal:
                    client_rfsoc.set_mode('RXen0_TXen1')
                    client_rfsoc.set_tx_gain()
                elif params.recv_signal:
                    client_rfsoc.set_mode('RXen1_TXen0')
                    client_rfsoc.set_rx_gain()

            signals_inst.init_objects(client_rfsoc=client_rfsoc, client_lintrack=client_lintrack, 
                              client_turntable=client_turntable, client_piradio=client_piradio, 
                              client_controller=client_controller, txtd_base=txtd_base)

            signals_inst.calibrate_rx_phase_offset(client_rfsoc)
            if params.control_piradio:
                if params.set_piradio_opt_gains:
                    signals_inst.find_optimal_gain_piradio(client_rfsoc, client_piradio, client_controller)
                    signals_inst.set_optimal_gain_piradio(client_piradio, client_controller)
                if params.set_piradio_opt_losupp:
                    signals_inst.set_optimal_losupp_piradio(client_piradio, client_controller)
            if params.nf_param_estimate:
                signals_inst.create_near_field_model()

            if 'channel' in params.save_list or 'signal' in params.save_list:
                signals_inst.save_signal_channel(client_rfsoc, client_turntable, client_piradio, client_controller, txtd_base, save_list=params.save_list)
            
            # signals_inst.operator()
        
        

    if 'client' in params.mode and not 'slave' in params.mode:
        # signals_inst.animate_plot(txtd_base, plot_mode=params.animate_plot_mode, plot_level=0)
        animate_plot_inst = Animate_Plot(params, signals_inst, txtd_base)

        animate_plot_inst.init_objects(client_rfsoc=client_rfsoc, client_lintrack=client_lintrack, 
                        client_turntable=client_turntable, client_piradio=client_piradio, 
                        client_controller=client_controller, txtd_base=txtd_base)
        
        animate_plot_inst.init_plots()




if __name__ == '__main__':
    
    params = Params_Class()
    rfsoc_run(params)

