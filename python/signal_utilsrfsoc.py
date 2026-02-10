import os
import time
import itertools
import ast
from dataclasses import dataclass
from typing import Any
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import scipy.io
from scipy import constants


from sigcom_toolkit.signal_utils import Signal_Utils, SignalUtilsConfig
from sigcom_toolkit.general import GeneralConfig, General
from tcp_comm import (
    PiradioRestComConfig,
    TCPComRFSoCConfig,
    TCPComLinTrackConfig,
    TCPComControllerConfig,
    Tcp_Comm_RFSoC,
    Tcp_Comm_LinTrack,
    REST_Com_Piradio,
    Tcp_Comm_Controller,
)
from serial_comm import SerialComTurnTableConfig, Serial_Comm_TurnTable




@dataclass
class RxSignal:
    rxtd: np.ndarray
    rxtd_base: np.ndarray
    h_est_full: np.ndarray
    H_est: np.ndarray
    H_est_max: np.ndarray
    sparse_est_params: dict


@dataclass
class PlotChart:
    plot_signals: dict
    title: str
    x_label: str
    y_label: str

@dataclass
class PlotSignal:
    signal_name: str
    trx_id: list
    process_list: list
    x: np.ndarray
    data: np.ndarray
    label: str

@dataclass
class TxSignal:
    txtd: np.ndarray
    txtd_base: np.ndarray


@dataclass
class ClientRFSoCConfig(TCPComRFSoCConfig):
    calib_config_path: str = './calib_config.npz'

class ClientRFSoC(Tcp_Comm_RFSoC):
    def __init__(self, config: ClientRFSoCConfig, **overrides: Any):
        super().__init__(config, **overrides)

        self.rx_phase_offset = 0
        self.rx_delay_offset = 0


    def calibrate_rx_phase_offset(self):
        '''
        This function calibrates the phase offset between the receivers ports in RFSoCs
        '''
        input_ = input("Press Y for phase offset calibration (and position the TX/RX at AoA = 0) or any key to use the saved phase offset: ")

        if input_.lower()!='y':
            if os.path.exists(self.config.calib_config_path):
                self.rx_phase_offset = np.load(self.config.calib_config_path)['rx_phase_offset']
                self.rx_delay_offset = np.load(self.config.calib_config_path)['rx_delay_offset']
                self.print("Using saved phase offset between RX ports: {:0.3f} Rad".format(self.rx_phase_offset), thr=1)
                # self.print("Using saved delay offset between RX ports: {:0.3f} s".format(self.rx_delay_offset), thr=1)
            else:
                self.print("No saved calibration found, please calibrate the phase offset", thr=0)
                self.rx_phase_offset = 0
                self.rx_delay_offset = 0
            return
        else:
            phase_diff_list = []
            delay_list = []
            for i in range(self.calib_iter):
                rxtd = self.receive_data_rfsoc(mode='once')
                rxtd = rxtd[0]
                phase_diff = Signal_Utils.calc_phase_offset(rxtd[0,:], rxtd[1,:])
                delay = phase_diff / (2*np.pi*self.fc)
                phase_diff_list.append(phase_diff)
                delay_list.append(delay)

            self.rx_phase_offset = np.mean(phase_diff_list)
            self.rx_delay_offset = np.mean(delay_list)
            np.savez(self.config.calib_config_path, rx_phase_offset=self.rx_phase_offset, rx_delay_offset=self.rx_delay_offset, fc=self.fc)
            self.print("Calibrated and saved phase offset between RX ports: {:0.3f} Rad".format(self.rx_phase_offset), thr=1)
            # self.print("Calibrated and saved delay offset between RX ports: {:0.3f} s".format(self.rx_delay_offset), thr=1)



@dataclass
class PiRadioConfig(PiradioRestComConfig):
    freq_hop_list: list = [10.0e9]
    stable_fc_piradio: float = 10.0e9
    optimal_gains_path: str = './optimal_gains.json'
    piradio_gain_sw_dly_default: float = 0.1
    freq_range: list = [6.0, 22.5]

class PiRadioFR3Trx(REST_Com_Piradio):
    def __init__(self, config: PiRadioConfig, **overrides: Any):        
        super().__init__(config, **overrides)

        self.fc_id = 0

    def hop_freq(self, fc_id=None, freq=None, set_opt_losupp=False):
        if fc_id is None:
            fc_id = (self.fc_id + 1) % len(self.config.freq_hop_list)
        if freq is not None:
            fc = freq
        else:
            fc = self.config.freq_hop_list[int(fc_id)]
        if self.fc != fc:
            self.set_frequency_piradio(fc=fc)

            if set_opt_losupp:
                self.set_optimal_losupp_piradio(fc=fc)

            self.fc_id = fc_id
            self.fc = fc
            self.wl = constants.c / self.fc


    def set_optimal_losupp_piradio(self, fc=None):
        if fc is None:
            fc = self.fc

        self.print("Setting optimal LO suppression for TX and RX in Pi-Radio", thr=1)
        
        lo_supp_lut = { 6.5: [-0.026, -0.021],  7.5: [-0.025, -0.016],  8.5: [-0.001, -0.036],  9.5: [0.078, -0.045],
                        10.5: [0.192, -0.146],  11.5: [0.113, -0.08],   12.5: [0.055, -0.03],   13.5: [0.04, 0.008],
                        14.5: [0.016, -0.002],  15.5: [-0.002, -0.022], 16.5: [0.004, -0.065],  17.5: [0.034, -0.065],
                        18.5: [0.049, -0.005],  19.5: [0.075, 0.003],   20.5: [0.116, 0.049],   21.5: [0.07, 0.027],    22.5: [-0.025, -0.027]}

        nearest_fc = min(lo_supp_lut.keys(), key=lambda x: abs(x - fc / 1e9))
        optimal_lo_supp = lo_supp_lut[nearest_fc]

        self.print("Nearest frequency: {} GHz, Optimal LO suppression: {}".format(nearest_fc, optimal_lo_supp), thr=1)
        self.set_bias_piradio(chan=0, iq='I', bias_voltage=optimal_lo_supp[0])
        self.set_bias_piradio(chan=0, iq='Q', bias_voltage=optimal_lo_supp[1])
        self.set_bias_piradio(chan=1, iq='I', bias_voltage=optimal_lo_supp[0])
        self.set_bias_piradio(chan=1, iq='Q', bias_voltage=optimal_lo_supp[1])
    

    def set_optimal_gain_piradio(self, side='both', tx_rx_distance=3.0):
        self.print("Setting optimal TX/RX gains in Pi-Radio", thr=0)

        freq_list = list(self.optimal_gains[tx_rx_distance].keys())
        nearest_fc = min(freq_list, key=lambda x: abs(x - self.config.stable_fc_piradio/1e9))

        if side=='rx' or side=='both':
            rx_gain_optimal = self.optimal_gains[tx_rx_distance][nearest_fc]['rx_gain']
            self.set_gain_piradio(trx='rx', chan=0, gain_db=rx_gain_optimal)
            self.set_gain_piradio(trx='rx', chan=1, gain_db=rx_gain_optimal)
        if side=='tx' or side=='both':
            tx_gain_optimal = self.optimal_gains[tx_rx_distance][nearest_fc]['tx_gain']
            self.set_gain_piradio(trx='tx', chan=0, gain_db=tx_gain_optimal)
            self.set_gain_piradio(trx='tx', chan=1, gain_db=tx_gain_optimal)




@dataclass
class SignalUtilsRFSoCConfig(SignalUtilsConfig):
    pass


class Signal_Utils_Rfsoc(Signal_Utils):
    def __init__(self, config: SignalUtilsRFSoCConfig, **overrides):
        super().__init__(config, **overrides)

        # self.import_attributes(config)
        self._network_topology = self.config.network_topology
        self._network_objects = self.config.network_objects

        self.rx_phase_list = []
        self.rot_angle_id = 0
        self.aoa_list = []
        self.lin_track_dir = 'forward'
        self.tx_rx_distance = 3.0
        self.tx_signal = None
        self.rx_signal = None
        self.animate_plotter = None

        self.print("signals object initialization done", thr=1)


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
            if item['type'] in ['rfsoc', 'controller'] and item['role']=='tx':
                rfsoc_tx_list.append(name)
        return rfsoc_tx_list

    @property
    def piradio_tx_list(self):
        piradio_tx_list = []
        for name in self.network_topology:
            item = self.network_topology[name]
            if item['type'] in ['piradio', 'controller'] and item['role']=='tx':
                piradio_tx_list.append(name)
        return piradio_tx_list

    @property
    def rfsoc_rx_list(self):
        rfsoc_rx_list = []
        for name in self.network_topology:
            item = self.network_topology[name]
            if item['type'] in ['rfsoc', 'controller'] and item['role']=='rx':
                rfsoc_rx_list.append(name)
        return rfsoc_rx_list

    @property
    def piradio_rx_list(self):
        piradio_rx_list = []
        for name in self.network_topology:
            item = self.network_topology[name]
            if item['type'] in ['piradio', 'controller'] and item['role']=='rx':
                piradio_rx_list.append(name)
        return piradio_rx_list


    def init_objects(self):

        # TODO Check all these
        # client_rfsoc
        # client_lintrack
        # client_turntable
        # client_piradio
        # client_controller

        for name in self.network_topology:
            item = self.network_topology[name]
            if item['type']=='rfsoc':
                ip_address = item['ip']
                rfsoc_config = TCPComRFSoCConfig().update_from_config(self.config)
                self._network_objects[name] = Tcp_Comm_RFSoC(rfsoc_config, server_ip=ip_address)
                self._network_objects[name].init_tcp_client()

            elif item['type']=='lintrack':
                ip_address = item['ip']
                lintrack_config = TCPComLinTrackConfig().update_from_config(self.config)
                self._network_objects[name] = Tcp_Comm_LinTrack(lintrack_config, server_ip=ip_address)
                self._network_objects[name].init_tcp_client()
                # self._network_objects[name].return2home()
                # self._network_objects[name].go2end()
            
            elif item['type']=='turntable':
                port = item.get('port', 'COM6')
                baudrate = item.get('baudrate', 115200)
                rotation_delay = item.get('rotation_delay', 0.0)
                turntable_config = SerialComTurnTableConfig(port=port, baudrate=baudrate, rotation_delay=rotation_delay)
                self._network_objects[name] = Serial_Comm_TurnTable(turntable_config)
                try:
                    self._network_objects[name].connect()
                    self._network_objects[name].move_to_position(0)
                    if 'calibrate' in item and item['calibrate']:
                        self._network_objects[name].calibrate()
                    self._network_objects[name].interactive_move()
                except:
                    self._network_objects[name].list_ports()
                    raise Exception("Turntable not connected or wrong port, please check the port list")

            elif item['type']=='piradio':
                ip_address = item['ip']
                piradio_config = PiradioRestComConfig().update_from_config(self.config)
                self._network_objects[name] = REST_Com_Piradio(piradio_config, ip_address=ip_address)
                self._network_objects[name].set_frequency_piradio(fc=self.config.fc)
            
            
            elif item['type']=='controller':
                ip_address = item['ip']
                controller_config = TCPComControllerConfig().update_from_config(self.config)
                self._network_objects[name] = Tcp_Comm_Controller(controller_config, server_ip=ip_address)
                self._network_objects[name].init_tcp_client()
                self._network_objects[name].set_frequency_piradio(self.config.fc)

            if 'slave' in self.config.host_role:
                controller_config = TCPComControllerConfig().update_from_config(self.config)
                self._network_objects['self'] = Tcp_Comm_Controller(controller_config)
                self._network_objects['self'].init_tcp_server()
                piradio_key = next((k for k, v in self.network_topology.items() if v['type'] == 'piradio'), None)
                rfsoc_key = next((k for k, v in self.network_topology.items() if v['type'] == 'rfsoc'), None)
                if not piradio_key or not rfsoc_key:
                    raise ValueError("Slave mode requires at least one piradio and one rfsoc in network_topology")
                self._network_objects['self'].obj_piradio = self._network_objects[piradio_key]
                self._network_objects['self'].obj_rfsoc = self._network_objects[rfsoc_key]
                self._network_objects['self'].run_tcp_server(self._network_objects['self'].parse_and_execute)


        for item in self.rfsoc_tx_list:
            client_rfsoc = self.network_objects[item]
            client_rfsoc.transmit_data_rfsoc(self.tx_signal.txtd)


        for item in self.rfsoc_rx_list:
            client_rfsoc = self.network_objects[item]
            client_rfsoc.set_frequency_mixer_rfsoc(self.config.mix_freq_dac, self.config.mix_freq_adc)
            if self.config.RFFE=='sivers':
                client_rfsoc.set_frequency_sivers(self.config.fc)
                client_rfsoc.set_mode_sivers('RXen1_TXen0')
                client_rfsoc.set_rx_gain_sivers()
            client_rfsoc.calibrate_rx_phase_offset()


        for item in self.rfsoc_tx_list:
            client_rfsoc = self._network_objects[item]
            client_rfsoc.set_frequency_mixer_rfsoc(self.config.mix_freq_dac, self.config.mix_freq_adc)
            if self.config.RFFE=='sivers':
                client_rfsoc.set_frequency_sivers(self.config.fc)
                client_rfsoc.set_mode_sivers('RXen0_TXen1')
                client_rfsoc.set_tx_gain_sivers()

        try:
            from tb4_aoa_viz.aoa_bridge import get_publish_aoa_fn
            from tb4_aoa_viz.snr_bridge import get_publish_snr_fn
            self.publish_aoa_turtlebot = get_publish_aoa_fn("/aoa_angle")
            self.publish_snr_turtlebot = get_publish_snr_fn("/snr_db")
        except ImportError:
            self.print("tb4_aoa_viz package not found, turtlebot publishing disabled", thr=0)
            self.publish_aoa_turtlebot = lambda x: None
            self.publish_snr_turtlebot = lambda x: None

        self.print("signals object init done", thr=1)


    def gen_tx_signal(self):
        txtd_base = []
        txtd = []
        for ant_id in range(self.config.n_tx_ant):
            if 'tone' in self.config.sig_mode:
                if self.config.sig_mode=='tone_1':
                    nsc = 1
                elif self.config.sig_mode=='tone_2':
                    nsc = 2
                txtd_base_s = self.generate_tone(freq_mode=self.config.tone_f_mode, sc=self.config.sc_tone, f=self.config.f_tone, sig_mode=self.config.sig_mode, gen_mode=self.config.sig_gen_mode)
            elif 'wideband' in self.sig_mode:
                nsc = self.config.wb_sc_range[1] - self.config.wb_sc_range[0] + 1
                txtd_base_s = self.generate_wideband(bw_mode=self.config.wb_bw_mode, sc_range=self.config.wb_sc_range, bw_range=self.config.wb_bw_range, modulation=self.config.sig_modulation, sig_mode=self.config.sig_mode, gen_mode=self.config.sig_gen_mode, seed=self.config.seed_list[ant_id])
            elif self.config.sig_mode == 'load':
                txtd_base_s = np.load(self.config.sig_path)
            else:
                raise ValueError('Unsupported signal mode: ' + self.config.sig_mode)
            txtd_base_s /= np.max([np.abs(txtd_base_s.real), np.abs(txtd_base_s.imag)])
            txtd_base_s *= self.db_to_lin(self.config.sig_gain_db, mode='mag')
            txtd_base.append(txtd_base_s)

            self.config.sig_pow_dbm = self.lin_to_db(0.5 * 1000, mode='pow') + self.config.sig_gain_db
            bw = (nsc/self.config.nfft_tx) * self.config.fs_tx
            self.config.sig_psd_dbm = self.config.sig_pow_dbm - self.lin_to_db(bw, mode='pow')
            self.config.sig_psd_dbm_sc = self.config.sig_pow_dbm - self.lin_to_db(nsc, mode='pow')
            self.print('TX Signal power for antenna {}: {:0.3f} dbm'.format(ant_id, self.config.sig_pow_dbm), thr=4)
            self.print('TX Signal PSD for antenna {}: {:0.3f} dBm/Hz = {:0.3f} dBm/MHz = {:0.3f} dBm/sc'.format(ant_id, self.config.sig_psd_dbm, self.config.sig_psd_dbm+self.lin_to_db(1e6, mode='pow'), self.config.sig_psd_dbm_sc), thr=4)

            title = 'TX signal spectrum in base-band for antenna {}'.format(ant_id)
            xlabel = 'Frequency (MHz)'
            ylabel = 'Magnitude (dB)'
            self.plot_signal(x=self.config.freq_tx, sigs=txtd_base[ant_id], mode='fft', scale='dB20', title=title, xlabel=xlabel, ylabel=ylabel, plot_level=4)
            title = 'Base-band TX signal in time domain at \n the time transition for antenna {}'.format(ant_id)
            xlabel = 'Time (s)'
            ylabel = 'Magnitude'
            n=int(np.round(self.config.fs_tx/self.config.f_max))
            t=self.config.t_tx[:2*n]
            sig_real=np.concatenate((txtd_base[ant_id].real[-n:], txtd_base[ant_id].real[:n]))
            sig_imag=np.concatenate((txtd_base[ant_id].imag[-n:], txtd_base[ant_id].imag[:n]))
            self.plot_signal(x=t, sigs={'real':sig_real, 'imag':sig_imag}, mode='time', scale='linear', title=title, xlabel=xlabel, ylabel=ylabel, plot_level=4, legend=True)

        txtd_base = np.array(txtd_base)

        if self.config.tx_sig_sim == 'shifted':
            if self.config.n_tx_ant < 2:
                raise ValueError("tx_sig_sim='shifted' requires at least two TX antennas")
            txtd_base[1,:] = np.roll(txtd_base[0,:], shift=(384), axis=-1)

        if self.config.rfsoc_mixer_mode=='digital' and self.config.mix_freq_dac!=0:
            for ant_id in range(self.config.n_tx_ant):
                txtd_s = self.freq_shift(txtd_base[ant_id], shift=self.config.mix_freq_dac, fs=self.config.fs_tx)
                txtd.append(txtd_s)        
        else:
            txtd = txtd_base.copy()

        txtd = np.array(txtd)

        if self.config.beamforming:
            txtd_base = self.beam_form(txtd_base)
            txtd = self.beam_form(txtd)

        if self.config.n_tx_ant > 1:
            self.print(f"Dot product of transmitted signals: {np.abs(np.vdot(txtd_base[1], txtd_base[0]))}", thr=4)
        # self.plot_signal(sigs = np.abs(np.correlate(txtd_base[1,:], txtd_base[0,:], mode='full')))

        self.tx_signal = TxSignal(txtd=txtd, txtd_base=txtd_base)
        return self.tx_signal


    def validate_saved_signals(self, rxtd, txtd=None, thr = 1e-8):
        self.print("Sanity check for saved signals", thr=2)

        mses = []
        for i in range(self.config.n_save):
            mse = self.mse(rxtd[i,0], rxtd[i,1])
            mses.append(mse)
            mse = self.mse(rxtd[i-self.config.n_save//self.config.n_frame_rd,0], rxtd[i,0])
            mses.append(mse)
            mse = self.mse(rxtd[i-self.config.n_save//self.config.n_frame_rd,1], rxtd[i,1])
            mses.append(mse)
            mse = self.mse(rxtd[i-1,0], rxtd[i,0])
            mses.append(mse)
            mse = self.mse(rxtd[i-1,1], rxtd[i,1])
            mses.append(mse)
            
        if np.min(mses) < thr:
            self.print("RX signals are not saved correctly", thr=0)
            raise ValueError('RX signals are not saved correctly')
        
        if txtd is not None:
            offset = np.argmax(np.abs(txtd[0,0]))-np.argmax(np.abs(txtd[0,1]))
            self.print("Offset between TX signals: {}".format(offset), thr=0)

        self.print("Sanity check passed", thr=3)



    def process_sys_response(self):
        self.sys_response = np.load(self.config.sys_response_path)['h_est_full_avg']
        self.sys_response /= np.max(np.abs(self.sys_response))


    def compute_sys_response(self):
        n_rx = 1

        if n_rx == 1:
            sys_response_folder = os.path.join(os.getcwd(), 'sigs_tx1_rx1_rx_rotate/')
        elif n_rx == 2:
            sys_response_folder = os.path.join(os.getcwd(), 'sigs_tx2_rx2_rx_rotate/')

        n_tx = n_rx
        postfix = "{}x{}".format(n_rx, n_tx)
        sys_response_path = os.path.join(self.config.channel_dir, 'sys_response_{}.npz'.format(postfix))
        if not os.path.exists(sys_response_folder):
            os.makedirs(sys_response_folder)

        if not os.path.exists(sys_response_path):
            sys_response = {}

            for file_name in os.listdir(sys_response_folder):
                if file_name.endswith('.npz') or file_name.endswith('.mat'):
                    self.print("Processing file: {}".format(file_name), thr=0)
                    file_path = os.path.join(sys_response_folder, file_name)
                    if file_name.endswith('.npz'):
                        data = np.load(file_path)
                        data_dict = {key: data[key] for key in data.files}
                    elif file_name.endswith('.mat'):
                        data = scipy.io.loadmat(file_path)
                        data_dict = {key: value for key, value in data.items() if not key.startswith('__')}

                    spec_list = file_name.split('_')
                    angle = float(spec_list[0])
                    sys_response[angle] = {}

                    txtd_base = data_dict['txtd']
                    txtd_base = txtd_base[0]

                    rxtd_dict = {}
                    for key, value in data_dict.items():
                        if not 'rxtd_' in key:
                            continue
                        frequency = float(key.split('_')[-1])
                        rxtd_dict[frequency] = value
                        
                    for frequency, rxtd in rxtd_dict.items():
                        rxtd = np.mean(rxtd, axis=0)
                        rx_signal = self.rx_operations(txtd_base, rxtd)
                        max_gain = np.max(np.abs(rx_signal.h_est_full), axis=-1)
                        sys_response[angle][frequency] = max_gain


            angles = [float(angle) for angle in sys_response.keys()]
            angles = np.array(angles)
            angles = np.sort(angles)
            frequencies = np.array(list(sys_response[angles[0]].keys()))
            frequencies = np.sort(frequencies)

            n_rx_ = np.shape(sys_response[angles[0]][frequencies[0]])[0]
            n_tx_ = np.shape(sys_response[angles[0]][frequencies[0]])[1]
            sys_response_matrix = np.zeros((len(angles), len(frequencies), n_rx_, n_tx_))
            for i, angle in enumerate(angles):
                for j, frequency in enumerate(frequencies):
                    sys_response_matrix[i,j] = sys_response[angle][frequency]
            
            
            np.savez(sys_response_path, sys_response_matrix=sys_response_matrix, angles=angles, frequencies=frequencies)
        
        else:
            data = np.load(sys_response_path)
            sys_response_matrix = data['sys_response_matrix']
            angles = data['angles']
            frequencies = data['frequencies']
        

        sys_response_matrix /= np.max(sys_response_matrix)
        sys_response_matrix = self.lin_to_db(sys_response_matrix, mode='mag')


        plot_params_dict = {'title_size': 18, 'title_weight': 'bold', 'title_max_chars': 45, 'xaxis_size': 16, 'yaxis_size': 16, 'ticks_size': 14, 'legend_size': 16, 'line_width': 2.0, 'marker_size': 8, 'hspace': 0.5, 'wspace': 0.5}


        fixed_angles = [-90, -30, 0, 30, 90]
        if n_rx > 1:
            fixed_angles = [-90, 0, 30, 90]
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        lines = []
        for fixed_angle in fixed_angles:
            angle_id = np.where(angles==fixed_angle)[0][0]
            for rx_id in range(n_rx):
                line, = ax.plot(frequencies, sys_response_matrix[angle_id,:,rx_id,0], label='Angle {}, RX {}'.format(fixed_angle, rx_id))
                lines.append(line)
        plot_params_dict['title'] = 'System Response vs Frequency at Different Angles'
        plot_params_dict['xlabel'] = 'Frequency (GHz)'
        plot_params_dict['ylabel'] = 'Normalized Response (dB)'
        self.set_plot_params(ax, lines, plot_params_dict)
        plt.savefig(os.path.join(self.config.figs_dir, 'sys_response_vs_freq_{}.pdf'.format(postfix)))


        fixed_freqs = [6.0, 8.0, 10.0, 15.0, 20.0]
        if n_rx > 1:
            fixed_freqs = [6.0, 10.0, 15.0, 20.0]
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        lines = []
        for fixed_freq in fixed_freqs:
            freq_id = np.where(frequencies==fixed_freq)[0][0]
            for rx_id in range(n_rx):
                line, = ax.plot(angles, sys_response_matrix[:,freq_id,rx_id,0], label='Fc {}GHz, RX {}'.format(fixed_freq, rx_id))
                lines.append(line)
        plot_params_dict['title'] = 'System Response vs Angle at Different Frequencies'
        plot_params_dict['xlabel'] = 'Angle (Deg)'
        plot_params_dict['ylabel'] = 'Normalized Response (dB)'
        self.set_plot_params(ax, lines, plot_params_dict)
        plt.show()
        plt.savefig(os.path.join(self.config.figs_dir, 'sys_response_vs_angle_{}.pdf'.format(postfix)))


        rx_id = 0
        tx_id = 0
        for rx_id in range(n_rx):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            lines = []
            cax = ax.imshow(sys_response_matrix[:,:,rx_id,tx_id], extent=[frequencies[0], frequencies[-1], angles[0], angles[-1]], aspect='auto', origin='lower', cmap='viridis')
            cbar = fig.colorbar(cax, ax=ax, label='Normalized Response (dB)')
            cbar.ax.tick_params(labelsize=plot_params_dict['ticks_size'])
            cbar.ax.yaxis.label.set_size(plot_params_dict['yaxis_size'])
            plot_params_dict['title'] = '2D Heat Diagram of System Response for RX {}'.format(rx_id)
            plot_params_dict['xlabel'] = 'Frequency (GHz)'
            plot_params_dict['ylabel'] = 'Angle (Deg)'
            self.set_plot_params(ax, lines, plot_params_dict)
            plt.savefig(os.path.join(self.config.figs_dir, 'sys_response_2D_{}_RX{}.pdf'.format(postfix, rx_id)))
            plt.show()





    def collect_signals(self):
        collect_count = 512
        ignore_less_count = False
        # input_folder = self.config.channel_dir
        input_folder = self.config.sig_dir
        # input_folder = "./sigs_tx1_rx1_rx_rotate"
        output_folder = os.path.join(input_folder, 'collected')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file_name in os.listdir(input_folder):
            if file_name.endswith('.npz') or file_name.endswith('.mat'):
                self.print("Processing file: {}".format(file_name), thr=0)
                file_path = os.path.join(input_folder, file_name)
                if file_name.endswith('.npz'):
                    data = np.load(file_path)
                    data_dict = {key: data[key] for key in data.files}
                elif file_name.endswith('.mat'):
                    data = scipy.io.loadmat(file_path)
                    data_dict = {key: value for key, value in data.items() if not key.startswith('__')}

                collected_data = {}
                for key, value in data_dict.items():
                    # print(key, value.shape)
                    if not any (x in key for x in ['rxtd', 'h_est_full']):
                        continue
                    elif ignore_less_count and value.shape[0] < collect_count:
                        continue
                    else:
                        if key == 'txtd':
                            collect_count_ = 1
                        else:
                            collect_count_ = collect_count
                        collect_count_ = min(value.shape[0], collect_count_)
                        collected_data[key] = value[:collect_count_]

                for key, value in collected_data.items():
                    if 'rxtd' in key:
                        rxtd = value
                        self.validate_saved_signals(rxtd=rxtd, txtd=collected_data['txtd'])
                output_file_path = os.path.join(output_folder, file_name)
                print([(key, value.shape) for (key, value) in collected_data.items()])
                # np.savez(output_file_path, **collected_data)


    def find_optimal_gain_piradio(self, client_rfsoc_rx, client_piradio_rx, client_piradio_tx):

        if os.path.exists(self.config.optimal_gains_path):
            self.optimal_gains = self.load_dict_from_json(self.config.optimal_gains_path, convert_values=True)
        else:
            self.optimal_gains = {}

        input_ = input("Press Y for TX/RX optimal gains calibration or any key to use the saved data: ")
        if input_.lower()!='y':
            self.print("Using saved TX/RX optimal gains...", thr=0)
            return

        self.print("Finding optimal gain for TX/RX in Pi-Radio", thr=1)
        tx_rx_distance = input("Enter the distance between the TX and RX in meters: ")
        if tx_rx_distance != '':
            try:
                self.tx_rx_distance = float(tx_rx_distance)
            except:
                raise ValueError('Invalid distance value: {}'.format(self.tx_rx_distance))
        else:
            pass
        self.optimal_gains[self.tx_rx_distance] = {}

        max_total_gain_dB = 60
        min_tx_gain_dB = 10
        max_tx_gain_dB = 30
        min_rx_gain_dB = 10
        max_rx_gain_dB = 40
        gain_step_dB = 1
        
        tx_gain_dB_list = np.arange(min_tx_gain_dB, max_tx_gain_dB+gain_step_dB, gain_step_dB)
        rx_gain_dB_list = np.arange(min_rx_gain_dB, max_rx_gain_dB+gain_step_dB, gain_step_dB)

        freq_list = [client_piradio_rx.config.stable_fc_piradio]
        for frequency in freq_list:
            self.print("Finding gains for frequency: {} GHz".format(frequency), thr=1)
            for client in [client_piradio_rx, client_piradio_tx]:
                client.hop_freq(freq=frequency)

            self.optimal_gains[self.tx_rx_distance][frequency] = {}
        
            snr_dB_optimal = 0
            tx_gain_dB_optimal = 0
            rx_gain_dB_optimal = 0

            for tx_gain_dB in tx_gain_dB_list:
                if tx_gain_dB < min_tx_gain_dB or tx_gain_dB > max_tx_gain_dB:
                    continue
                self.print("Setting TX gain to {} dB".format(tx_gain_dB), thr=1)
                client_piradio_tx.set_gain_piradio(trx='tx', chan=0, gain_db=tx_gain_dB)
                client_piradio_tx.set_gain_piradio(trx='tx', chan=1, gain_db=tx_gain_dB)

                for rx_gain_dB in rx_gain_dB_list:
                    if rx_gain_dB < min_rx_gain_dB or rx_gain_dB > max_rx_gain_dB:
                        continue
                    if tx_gain_dB + rx_gain_dB > max_total_gain_dB:
                        continue

                    self.print("Setting RX gain to {} dB".format(rx_gain_dB), thr=1)
                    client_piradio_rx.set_gain_piradio(trx='rx', chan=0, gain_db=rx_gain_dB)
                    client_piradio_rx.set_gain_piradio(trx='rx', chan=1, gain_db=rx_gain_dB)
                    if client_piradio_rx.config.gain_sw_dly == 0:
                        time.sleep(2*client_piradio_rx.config.piradio_gain_sw_dly_default)

                    rxtd = client_rfsoc_rx.receive_data_rfsoc(mode='once')
                    snr = self.calculate_snr(sig_td=rxtd[0,:,:self.config.n_samples_trx], sig_sc_range=self.config.sc_range)
                    snr_dB = self.lin_to_db(snr, mode='pow')
                    self.print("SNR for TX gain {} dB and RX gain {} dB: {:.3f} dB".format(tx_gain_dB, rx_gain_dB, snr_dB), thr=1)
                    if snr_dB > snr_dB_optimal:
                        snr_dB_optimal = snr_dB
                        tx_gain_dB_optimal = tx_gain_dB
                        rx_gain_dB_optimal = rx_gain_dB

            self.print("Optimal TX gain for frequency {}: {} dB".format(frequency,tx_gain_dB_optimal), thr=1)
            self.print("Optimal RX gain for frequency {}: {} dB".format(frequency,rx_gain_dB_optimal), thr=1)
            self.print("Optimal SNR for frequency {}: {} dB".format(frequency,snr_dB_optimal), thr=1)

            self.optimal_gains[self.tx_rx_distance][frequency]['tx_gain'] = int(tx_gain_dB_optimal)
            self.optimal_gains[self.tx_rx_distance][frequency]['rx_gain'] = int(rx_gain_dB_optimal)


        self.save_dict_to_json(self.optimal_gains, self.config.optimal_gains_path)
        self.print("Calculated and saved optimal TX/RX gains...", thr=1)

        return self.optimal_gains
    

    def rx_operations(self, txtd_base, rxtd):
        # Expand the dimension for 1 frame received signals
        if len(rxtd.shape)<3:
            rxtd = np.expand_dims(rxtd, axis=0)
        sparse_est_params = None
        plt_frm_id = 0
        n_rd_rep = rxtd.shape[0]

        for ant_id in range(self.config.n_rx_ant):
            title = 'RX signal spectrum for antenna {}'.format(ant_id)
            xlabel = 'Frequency (MHz)'
            ylabel = 'Magnitude (dB)'
            self.plot_signal(x=self.config.freq_rx, sigs=rxtd[plt_frm_id, ant_id], mode='fft', scale='dB20', title=title, xlabel=xlabel, ylabel=ylabel, plot_level=4)

            title = 'RX signal in time domain (zoomed) for antenna {}'.format(ant_id)
            xlabel = 'Time (s)'
            ylabel = 'Magnitude'
            n = 4*int(np.round(self.config.fs_rx/self.config.f_max))
            self.plot_signal(x=self.config.t_rx[:n], sigs=rxtd[plt_frm_id, ant_id,:n], mode='time_IQ', scale='linear', title=title, xlabel=xlabel, ylabel=ylabel, legend=True, plot_level=4)

        if self.config.rfsoc_mixer_mode == 'digital' and self.config.mix_freq_adc!=0:
            rxtd_base = np.zeros_like(rxtd)
            for ant_id in range(self.config.n_rx_ant):
                for frm_id in range(n_rd_rep):
                    rxtd_base[frm_id, ant_id,:] = self.freq_shift(rxtd[frm_id, ant_id], shift=-1*self.config.mix_freq_adc, fs=self.config.fs_rx)
        else:
            rxtd_base = rxtd.copy()

        if 'filter' in self.config.rx_chain:
            for ant_id in range(self.config.n_rx_ant):
                for frm_id in range(n_rd_rep):
                    cf = (self.config.filter_bw_range[0]+self.config.filter_bw_range[1])/2
                    cutoff = self.config.filter_bw_range[1] - self.config.filter_bw_range[0]
                    rxtd_base[frm_id, ant_id,:] = self.filter(rxtd_base[frm_id, ant_id,:], center_freq=cf, cutoff=cutoff, fil_order=64, plot=False)

                title = 'RX signal spectrum after filtering in base-band for antenna {}'.format(ant_id)
                xlabel = 'Frequency (MHz)'
                ylabel = 'Magnitude (dB)'
                self.plot_signal(x=self.config.freq_rx, sigs=rxtd_base[0, ant_id], mode='fft', scale='dB20', title=title, xlabel=xlabel, ylabel=ylabel, plot_level=4)

        for ant_id in range(self.config.n_rx_ant):
            # n_samples = min(len(txtd_base), len(rxtd_base))
            txfd_base_ = np.abs(fftshift(fft(txtd_base[ant_id,:self.config.n_samples])))
            rxfd_base_ = np.abs(fftshift(fft(rxtd_base[plt_frm_id, ant_id,:self.config.n_samples])))

            title = 'TX and RX signals spectrum in base-band for antenna {}'.format(ant_id)
            xlabel = 'Frequency (MHz)'
            ylabel = 'Magnitude (dB)'
            scale = np.max(txfd_base_)/np.max(rxfd_base_)
            self.print("TX to RX spectrum scale for antenna {}: {:0.3f}".format(ant_id, scale), thr=4)
            xlim=(-2*self.config.f_max/1e6, 2*self.config.f_max/1e6)
            f1=np.abs(self.config.freq - xlim[0]).argmin()
            f2=np.abs(self.config.freq - xlim[1]).argmin()
            ylim=(np.min(rxfd_base_[f1:f2]*scale), 1.1*np.max(rxfd_base_[f1:f2]*scale))
            self.plot_signal(x=self.config.freq, sigs={"txfd_base":txfd_base_, "Scaled rxfd_base":rxfd_base_*scale}, scale='dB20', title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, legend=True, plot_level=5)
            self.print("txfd_base max freq for antenna {}: {} MHz".format(ant_id, self.config.freq[(self.config.nfft>>1)+np.argmax(txfd_base_[self.config.nfft>>1:])]), thr=4)
            self.print("rxfd_base max freq for antenna {}: {} MHz".format(ant_id, self.config.freq[(self.config.nfft>>1)+np.argmax(rxfd_base_[self.config.nfft>>1:])]), thr=4)

        if 'pilot_separate' in self.config.rx_chain:
            n_samples_rx = self.config.n_samples_trx * 2
        else:
            n_samples_rx = self.config.n_samples_trx

        txtd_base = txtd_base[:,:self.config.n_samples_trx]
        if 'integrate' in self.config.rx_chain:
            rxtd_base = self.integrate_signal(rxtd_base, n_samples=n_samples_rx)

        if 'sync_time' in self.config.rx_chain:
            rxtd_base_s = []
            for frm_id in range(n_rd_rep):
                if 'sync_time_frac' in self.config.rx_chain:
                    sync_frac = True
                else:
                    sync_frac = False
                rxtd_base_s_ = self.sync_time(rxtd_base[frm_id], txtd_base, sc_range=self.config.sc_range, rx_same_delay=self.config.rx_same_delay, sync_frac=sync_frac)
                rxtd_base_s.append(rxtd_base_s_)
            rxtd_base_s = np.array(rxtd_base_s)
        else:
            rxtd_base_s = rxtd_base.copy()
            rxtd_base_s = np.stack((rxtd_base_s, rxtd_base_s), axis=2)
        
        if 'sync_freq' in self.config.rx_chain:
            cfo_coarse = self.estimate_cfo(txtd_base, rxtd_base_s, mode='coarse', sc_range=self.config.sc_range)
            rxtd_base_t = self.sync_frequency(rxtd_base_s, cfo_coarse, mode='time')
            cfo_fine = self.estimate_cfo(txtd_base, rxtd_base_t, mode='fine', sc_range=self.config.sc_range)
            cfo = cfo_coarse + cfo_fine
            rxtd_base_s = self.sync_frequency(rxtd_base_s, cfo, mode='time')

        if 'pilot_separate' in self.config.rx_chain:
            rxtd_pilot_s = rxtd_base_s[:,:,:,:n_samples_rx//2]
            rxtd_base_s = rxtd_base_s[:,:,:,n_samples_rx//2:]
        else:
            rxtd_pilot_s = rxtd_base_s.copy()
        

        rxtd_base = np.stack((rxtd_base_s[:,0,0,:self.config.n_samples_trx], rxtd_base_s[:,1,0,:self.config.n_samples_trx]), axis=1)
        rxtd_pilot = np.stack((rxtd_pilot_s[:,0,0,:self.config.n_samples_trx], rxtd_pilot_s[:,1,0,:self.config.n_samples_trx]), axis=1)
        
        if 'channel_est' in self.config.rx_chain:
            if 'sys_res_deconv' in self.config.rx_chain:
                self.process_sys_response()
            else:
                self.sys_response = None
            snr_est = self.db_to_lin(self.config.snr_est_db, mode='pow')

            if 'sparse_est' in self.config.rx_chain:
                h = []
                for frm_id in range(n_rd_rep):
                    h_est_full, H_est, H_est_max = self.channel_estimate(txtd_base, rxtd_pilot_s[frm_id], sys_response=self.sys_response, sc_range_ch=self.config.sc_range_ch, snr_est=snr_est)
                    h.append(h_est_full)
                h = np.array(h)
                h = h.transpose(3,1,2,0)
                g = self.sys_response.copy() if self.sys_response is not None else None
                if g is not None:
                    g = g.transpose(2,0,1)
                ndly = 5000
                sparse_est_params = self.sparse_est(h=h, g=g, sc_range_ch=self.config.sc_range_ch, npaths=self.config.npath_max, nframe_avg=1, ndly=ndly, drange=self.config.sparse_ch_samp_range, cv=True, n_ignore=self.config.sparse_ch_n_ignore)
            else:
                h_est_full, H_est, H_est_max = self.channel_estimate(txtd_base, rxtd_pilot_s, sys_response=self.sys_response, sc_range_ch=self.config.sc_range_ch, snr_est=snr_est)
            
            self.rx_phase_list, self.aoa_list = self.estimate_mimo_params(txtd_base, rxtd_pilot, self.config.fc, h_est_full, H_est_max, self.rx_phase_list, self.aoa_list)
            if len(self.rx_phase_list)>self.config.nfft_trx//10:
                self.rx_phase_list.pop(0)
            if len(self.aoa_list)>self.config.nfft_trx//10:
                self.aoa_list.pop(0)
        else:
            h_est_full = np.ones((self.config.n_rx_ant, self.config.n_tx_ant, self.config.n_samples_ch), dtype=complex)
            H_est = np.ones((self.config.n_rx_ant, self.config.n_tx_ant), dtype=complex)
            H_est_max = H_est.copy()
        if 'channel_eq' in self.config.rx_chain and 'channel_est' in self.config.rx_chain:
            rxtd_base = self.channel_equalize(txtd_base, rxtd_base[plt_frm_id], h_est_full, H_est, sc_range=self.config.sc_range, sc_range_ch=self.config.sc_range_ch, null_sc_range=self.config.null_sc_range, n_rx_ch_eq=self.config.n_rx_ch_eq)

        if len(rxtd_base.shape)==3:
            rxtd_base = rxtd_base[plt_frm_id]
        self.rx_signal = RxSignal(
            rxtd=rxtd,
            rxtd_base=rxtd_base,
            h_est_full=h_est_full,
            H_est=H_est,
            H_est_max=H_est_max,
            sparse_est_params=sparse_est_params,
        )

        return self.rx_signal
    


    def process_sig(self, sig=None, process_list=[]):
        if sig is None:
            return None
        
        sig = sig.copy()
        title = ""
        for item in process_list:
            if item in ['tx', 'rx', 'h', 'H']:
                continue
            elif item == 'fft':
                sig = fft(sig, axis=-1)
                # title += "-FFT"
                title += "-FD"
            elif item == 'psd':
                nfft = 2**int(np.ceil(np.log2(len(sig))))
                sig = self.psd(sig, fs=self.config.fs_rx, nfft=nfft)
            elif item == 'ifft':
                sig = ifft(sig, axis=-1)
                title += "-IFFT"
            elif item == 'fftshift':
                sig = fftshift(sig, axes=-1)
            elif item == 'ifftshift':
                sig = ifftshift(sig, axes=-1)
            elif item == 'mag':
                sig = np.abs(sig)
                title += "-Mag"
            elif item == 'phase':
                sig = np.angle(sig)
                title += "-Phase"
            elif item == 'phase/2pi':
                sig = np.angle(sig) / (2*np.pi)
                title += "-Phase/2pi"
            elif item == 'phase_unwrap':
                sig = np.unwrap(np.angle(sig))
                title += "-PhaseUnwrap"
            elif item == 'real':
                sig = np.real(sig)
                title += "-Real"
            elif item == 'imag':
                sig = np.imag(sig)
                title += "-Imag"
            elif item == 'IQ':
                n_samples = sig.shape[-1]
                sig = sig[self.config.sc_range[0]+n_samples//2:self.config.sc_range[1]+n_samples//2+1]
                title += "-IQ"
            elif item == 'conj':
                sig = np.conj(sig)
                title += "-Conj"
            elif item == 'dbmag':
                sig = self.lin_to_db(sig, mode='mag')
                title += "-dBMag"
            elif item == 'dbpow':
                sig = self.lin_to_db(sig, mode='pow')
                title += "-dBPow"
            elif item == 'circshift':
                im = np.argmax(np.abs(sig), axis=-1)
                sig = np.roll(sig, -im + len(sig)//4, axis=-1)
            elif item == 'normalize':
                sig = sig / np.max(np.abs(sig))
                title += "-Norm"
            else:
                raise ValueError("Invalid operation: {}".format(item))

        return sig, title


    def operator(self):
        measurement = {}
        save_id = 1
        phys_config = None
        phys_config_id = 0
        angle_id = 0
        freq_id = 0
        read_id = 0

        def parse_action(spec):
            spec_list = spec.split("/")
            target = spec_list[0]
            action = spec_list[1]
            rng = spec_list[2]

            if len(spec_list) <= 3:
                param = None
            else:
                param = [p for p in spec_list[3:] if p != ""]

            try:
                rng = eval(rng)
                if not isinstance(rng, (int, float, np.integer, np.floating)):
                    rng = np.array(rng) if hasattr(rng, '__len__') else np.array([float(rng)])
                else:
                    rng = np.array([rng])
            except:
                if rng.count(":")<2:
                    rng = None
                else:
                    start, stop, count = map(float, rng.split(":")[:3])
                    if 'log' in rng:
                        rng = np.logspace(np.log10(start), np.log10(stop), int(count))
                    else:
                        rng = np.linspace(start, stop, int(count))

            print(target, action, rng, param)
            return (target, action, rng, param)

        loop_list = [parse_action(item) for item in self.config.action_loop]
        targets = [item[0] for item in loop_list]
        actions = [item[1] for item in loop_list]
        ranges = [item[2] for item in loop_list]
        params = [item[3] for item in loop_list]


        prev = None
        default_actions = ['capture', 'save', 'wait']
        for values in itertools.product(*ranges):
            print(values)
            if prev is None:
                changed_idxs = range(len(values))  # first iteration: everything is "changed"
            else:
                # indices where the value differs from previous step
                changed_idxs = [i for i, (a, b) in enumerate(zip(prev, values)) if a != b or actions[i] in default_actions]
            prev = values

            # process only the actions whose value changed
            for i in changed_idxs:
                target_spec = targets[i]
                action = actions[i]
                value = values[i]
                param = params[i]

                if isinstance(target_spec, str) and target_spec.strip().startswith(("[", "(")):
                    targets_list = ast.literal_eval(target_spec)
                else:
                    targets_list = [target_spec]
                if not isinstance(targets_list, list):
                    targets_list = [targets_list]
                target_objects = []
                for target in targets_list:
                    target_object = self.network_objects[target] if target in self.network_objects else None
                    if target_object is None and not target in ['self']:
                        raise ValueError("Invalid target object: {}".format(target))
                    target_objects.append(target_object)


                if action == 'change_phys_config':
                    if not self.config.measurement_configs:
                        raise ValueError("measurement_configs is empty; cannot change physical configuration")
                    phys_config = self.config.measurement_configs[phys_config_id]
                    self.print(f'Please change the physical configuration to: {phys_config}', thr=0)
                    phys_config_id = (phys_config_id + 1) % len(self.config.measurement_configs)


                if action == 'change_tx_rx_distance':
                    tx_rx_distance = input('Please enter the TX to RX distance in meters (empty for default): ')
                    if tx_rx_distance != '':
                        try:
                            tx_rx_distance = float(tx_rx_distance)
                        except:
                            raise ValueError('Invalid distance value: {}'.format(tx_rx_distance))
                        self.tx_rx_distance = tx_rx_distance


                if action == 'capture':
                    client_rfsoc = target_objects[0]
                    process_signal = (param and param[0] == 'process')

                    rxtd_save=[]

                    if process_signal:
                        # n_rd_rep = self.n_save
                        n_rd_rep = int(value)
                    else:
                        # n_rd_rep = self.n_save//self.n_frame_rd
                        n_rd_rep = int(value)//self.config.n_frame_rd
                    rxtd = client_rfsoc.receive_data_rfsoc(n_rd_rep=n_rd_rep, mode='once', verbose=False)
                    self.rx_signal = RxSignal(
                        rxtd=rxtd,
                    )

                    if process_signal:
                        for i in range(self.config.n_save):
                            self.print("Channel Save Iteration: {}".format(i+1), thr=0)
                            rxtd = client_rfsoc.receive_data_rfsoc(n_rd_rep=n_rd_rep, mode='once')

                            # to handle the dimenstion needed for read repeat
                            rxtd_frame = rxtd[0] if rxtd.ndim == 3 else rxtd
                            rx_signal = self.rx_operations(self.tx_signal.txtd_base, rxtd_frame)
                            self.rx_signal = rx_signal

                            rxtd_save.append(rx_signal.rxtd_base)
                    else:
                        rxtd_save = np.empty((self.config.n_save, self.config.n_rx_ant, self.config.n_samples_tx), dtype=rxtd.dtype)
                        for i in range(self.config.n_frame_rd):
                            rxtd_save[i::self.config.n_frame_rd] = rxtd[:,:,i*self.config.n_samples_tx:(i+1)*self.config.n_samples_tx]

                    txtd_save = np.expand_dims(self.tx_signal.txtd_base, axis=0)
                    rxtd_save = np.array(rxtd_save)

                    self.validate_saved_signals(rxtd=rxtd_save)


                if action == 'capture_from_file':
                    sigs_save = np.load(self.config.sig_save_path)

                    rxtd = sigs_save['rxtd_{:.1f}'.format(self.config.fc/1e9)][read_id*self.config.n_rd_rep:(read_id+1)*self.config.n_rd_rep]
                    txtd_base = sigs_save['txtd'][0]

                    rx_signal = self.rx_operations(txtd_base, rxtd)
                    self.rx_signal = rx_signal
                    tx_signal = TxSignal(
                        txtd_base=txtd_base,
                    )
                    read_id+=1


                if action == 'update_plot':
                    if self.animate_plotter is None:
                        self.animate_plotter = Animate_Plot(self.config, self)

                    rx_signal = self.rx_signal
                    if rx_signal is None:
                        raise ValueError("update_plot requires a valid rx_signal; run capture first")

                    self.animate_plotter.update_once(rx_signal)


                if action == 'save':
                    # self.print("Starting to save signals for configuration: {}".format(phys_config), thr=0)
                    if not param or len(param) < 2:
                        raise ValueError("save action requires at least two params: save_list and save_prefix")
                    save_list = eval(param[0])  # e.g., ['signal', 'channel']

                    save_prefix = param[1]
                    save_postfix = phys_config if phys_config is not None else ''
                    save_name = f'{save_prefix}_{save_postfix}_{save_id}.{self.config.save_format}'

                    measurement['id'] = save_id
                    if 'signal' in save_list:
                        measurement['txtd'] = txtd_save.copy()
                        measurement['rxtd_{}'.format(self.config.fc/1e9)] = rxtd_save.copy()

                    measurement['sig_interval'] = [self.config.wb_sc_range[0]+(self.config.nfft_tx >> 1), self.config.wb_sc_range[1]+(self.config.nfft_tx >> 1)]
                    # measurement['tx_gain_db'] = tx_gain_db
                    # measurement['rx_gain_db'] = rx_gain_db

                    if 'signal' in save_list:
                        sig_save_path=os.path.join(self.config.sig_dir, save_name)
                        if self.config.save_format == 'npz':
                            np.savez(sig_save_path, **measurement)
                        elif self.config.save_format == 'mat':
                            scipy.io.savemat(sig_save_path, measurement)

                    save_id += 1


                if action == 'wait':
                    wait_time = float(value)
                    self.print("Waiting for {} seconds...".format(wait_time), thr=2)
                    time.sleep(wait_time)


                # TODO update this part
                if action == 'report_time':
                    freq_switch_time = 0.052 + self.config.piradio_freq_sw_dly
                    remaining_time = (len(rotation_angles) - angle_id) * (rotation_time + len(self.config.freq_hop_list)*(freq_switch_time))
                    self.print("Remaining time to save signals: {:0.0f} s".format(remaining_time), thr=0)
                    angle_id += 1


                if action == 'rotate_table':
                    client_turntable = target_objects[0]
                    angle = float(value)
                    self.print("Rotating to angle: {}".format(angle), thr=0)

                    start_time = time.time()
                    client_turntable.move_to_position(angle)
                    rotation_time = time.time()-start_time
                    self.print("Time taken to rotate: {:0.3f} s".format(rotation_time), thr=2)


                if action == 'move_lin_track':
                    client_lintrack = target_objects[0]
                    distance = float(value)
                    client_lintrack.move(lin_track_id=0, distance=distance)


                if action == 'return_lin_track_home':
                    client_lintrack.return2home(lin_track_id=0)

                if action == 'publish_aoa_ros2':
                    aoa = self.aoa_list[-1] if len(self.aoa_list)>0 else 0
                    self.publish_aoa_turtlebot(aoa)


                if action == 'publish_snr_ros2':
                    snr = self.calculate_snr(sig_td=self.rx_signal.rxtd[0,:,:self.config.n_samples_trx], sig_sc_range=self.config.sc_range)
                    snr_dB = self.lin_to_db(snr, mode='pow')
                    self.publish_snr_turtlebot(snr_dB)


                if action == 'hop_freq':
                    clients = []
                    client_piradio_rx = target_objects[0]
                    clients.append(client_piradio_rx)
                    if len(target_objects)>1:
                        client_piradio_tx = target_objects[1]
                        clients.append(client_piradio_tx)
                    frequency = None
                    try:
                        frequency = float(param[0])
                        for client in clients:
                            client.hop_freq(freq=frequency)
                    except Exception:
                        for client in clients:
                            client.hop_freq()
                        if clients:
                            frequency = clients[0].fc
                    if frequency is None:
                        self.print("Saving signals after frequency hop", thr=0)
                    else:
                        self.print("Saving signals for Freq: {} GHz".format(frequency/1e9), thr=0)


                if action == 'set_gain_db_tx':
                    client_piradio = target_objects[0]
                    gain_db = int(value)
                    tx_gain_db = gain_db
                    client_piradio.set_gain_piradio(trx='tx', chan=0, gain_db=gain_db)
                    client_piradio.set_gain_piradio(trx='tx', chan=1, gain_db=gain_db)


                if action == 'set_gain_db_rx':
                    client_piradio = target_objects[0]
                    gain_db = int(value)
                    rx_gain_db = gain_db
                    client_piradio.set_gain_piradio(trx='rx', chan=0, gain_db=gain_db)
                    client_piradio.set_gain_piradio(trx='rx', chan=1, gain_db=gain_db)


                if action == 'find_optimal_gain_piradio':
                    client_rfsoc_rx, client_piradio_rx, client_piradio_tx = target_objects
                    self.find_optimal_gain_piradio(client_rfsoc_rx, client_piradio_rx, client_piradio_tx)


                if action == 'set_optimal_gain_piradio':
                    client_piradio_rx, client_piradio_tx = target_objects
                    client_piradio_rx.set_optimal_gain_piradio(tx_rx_distance=self.tx_rx_distance)
                    client_piradio_tx.set_optimal_gain_piradio(tx_rx_distance=self.tx_rx_distance)


                if action == 'set_optimal_losupp_piradio':
                    client_piradio_rx, client_piradio_tx = target_objects
                    client_piradio_rx.set_optimal_losupp_piradio()
                    client_piradio_tx.set_optimal_losupp_piradio()


                if action == 'switch_sig_size':
                    sig_size = int(value)


                if action == 'switch_sig_ss':
                    client_rfsoc = target_objects[0]
                    region = self.generate_random_regions(shape=(1024,), n_regions=1, min_size=[sig_size], max_size=[sig_size])
                    self.config.wb_sc_range = [region[0][0].start-(self.config.nfft_tx >> 1), region[0][0].stop-1-(self.config.nfft_tx >> 1)]
                    tx_signal = self.gen_tx_signal()
                    client_rfsoc.transmit_data_rfsoc(tx_signal.txtd)
    



    def stream_rx_td_to_matlab(self, rxtd, freq):
        import io
        import socket
        import scipy.io
        
        matlab_stream_ip = '10.20.38.213'     # IP address for the MATLAB data transfer
        matlab_stream_port = 50007             # Port for the MATLAB data transfer

        try:
            buf = io.BytesIO()
            scipy.io.savemat(buf, {'rxtd': rxtd}, do_compression=True)
            scipy.io.savemat(buf, {'freq': freq}, do_compression=True)
            data = buf.getvalue()
            print(len(data), "bytes of rxtd data to be sent to MATLAB stream")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(.1)
                sock.connect((matlab_stream_ip, matlab_stream_port))
                sock.sendall(len(data).to_bytes(8, byteorder='big'))  # Send the length of the data first
                sock.sendall(data)
                self.print("rxtd data sent to MATLAB stream at {}:{}".format(matlab_stream_ip, matlab_stream_port), thr=1)
        
        except Exception as e:
            self.print("Error in streaming rxtd to MATLAB: {}".format(e), thr=0)






@dataclass
class AnimationPlotConfig(GeneralConfig):
    animate_plot_mode: list = []
    plot_configs: dict = None

class Animate_Plot(General):
    def __init__(self, config: AnimationPlotConfig, signals_obj: Signal_Utils_Rfsoc, **overrides: Any):
        super().__init__(config, **overrides)

        self.signals_obj = signals_obj
        self.signals_config = signals_obj.config

        self.config.n_plots_row = len(self.config.animate_plot_mode)
        self.config.n_plots_col = len(self.signals_config.freq_hop_list)

        self.config.plt_n_samples_rx = self.signals_config.n_samples_trx
        self.config.n_samp_ch_sp = self.signals_config.n_samples_ch // 2

        self.plot_colors = ['#57068C', 'orange', 'green', 'red', 'blue', 'brown', 'pink', 'gray', 'olive', 'cyan']
        # set matplotlib axes color cycle so subsequent ax.plot calls use our colors by default
        try:
            mpl.rcParams['axes.prop_cycle'] = cycler('color', self.plot_colors)
        except Exception:
            # fail silently if matplotlib/cycler are not available at init time
            pass
        self.mag_filter_list = {"process_list": ['fft'], "signal_name": ['h', 'H']}
        self.untouched_plot_list = {"process_list": ['IQ'], "signal_name": ['aoa_gauge', 'nf_loc']}

        self.anim_paused = False
        self.read_id = -1
        self.plots_initialized = False

        self.start_time = time.time()


    def process_signals_for_plot(self, rx_signal: RxSignal):

        '''
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
        '''

        rxtd_base = rx_signal.rxtd_base
        h_est_full = rx_signal.h_est_full
        H_est_full = rx_signal.H_est_full
        sparse_est_params = rx_signal.sparse_est_params

        supported_operations = ['+', '-', '*', '/']
        signals=[]
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

                signal_desc = signal_str.strip().split('|')

                signal_name = signal_desc[0]
                rx_id = int(signal_desc[1])
                tx_id = int(signal_desc[2])
                rx_ids.append(rx_id)
                tx_ids.append(tx_id)

                if len(signal_desc)>3:
                    signal_process_list = signal_desc[3:]
                else:
                    signal_process_list = []

                xlabel_mode = 'time'
                if 'mag' in signal_process_list:
                    ylabel_mode = 'mag'
                    if 'dbmag' in signal_process_list:
                        ylabel_mode += '_db'
                elif 'phase' in signal_process_list:
                    ylabel_mode = 'phase'
                elif 'phase/2pi' in signal_process_list:
                    ylabel_mode = 'phase/2pi'
                else:
                    ylabel_mode = 'mag'
                if 'IQ' in signal_process_list:
                    xlabel_mode = 'IQ'
                    ylabel_mode = 'IQ'


                if signal_name == 'txtd':
                    x = self.signals_config.t_tx[:self.signals_config.n_samples_tx]
                    sig = self.signals_obj.tx_signal.txtd_base[tx_id]
                    title += "TX"
                    if 'fft' in signal_process_list:
                        x = self.signals_config.freq_tx
                        xlabel_mode = 'freq'
                        title += "-FD"
                    else:
                        x = self.signals_config.t_tx*1e9
                        xlabel_mode = 'time'
                        title += "-TD"
                elif signal_name == 'rxtd':
                    sig = rxtd_base[rx_id]
                    title += "RX"
                    if 'fft' in signal_process_list:
                        x = self.signals_config.freq_trx
                        xlabel_mode = 'freq'
                        title += "-FD"
                    else:
                        x = self.signals_config.t_rx[:self.config.plt_n_samples_rx]*1e9
                        xlabel_mode = 'time'
                        title += "-TD"
                elif signal_name == 'h':
                    x = self.signals_config.t_trx[:self.signals_config.n_samples_ch]*1e9
                    sig = h_est_full[rx_id, tx_id]
                    title += "Channel"
                    if 'fft' in signal_process_list:
                        xlabel_mode = 'freq'
                        title += "-FD"
                    else:
                        xlabel_mode = 'time'
                        title += "-TD"
                elif signal_name == 'H':
                    x = self.signals_config.freq_trx[(self.signals_config.sc_range_ch[0]+self.signals_config.n_samples_trx//2):(self.signals_config.sc_range_ch[1]+self.signals_config.n_samples_trx//2+1)]
                    sig = H_est_full[rx_id, tx_id]
                    title += "Channel-FD"
                    if 'ifft' in signal_process_list:
                        xlabel_mode = 'time'
                        title += "-TD"
                    else:
                        xlabel_mode = 'freq'
                        title += "-FD"
                elif signal_name == 'h_sparse':
                    sig = sparse_est_params
                    title += "Multipath Channel PDP"
                    xlabel_mode = 'time_h_sparse'
                    ylabel_mode = 'snr'
                elif signal_name == 'rx_ph_diff':
                    sig = self.signals_obj.rx_phase_list
                    title += "RX-Phase Diff-TD"
                    xlabel_mode = 'id'
                    ylabel_mode = 'phase'
                elif signal_name == 'aoa_gauge':
                    # Return the last AOA gauge value in radians
                    sig = self.signals_obj.aoa_list[-1]
                    title += "AOA Gauge"
                    xlabel_mode = 'aoa_gauge'
                    ylabel_mode = 'aoa_gauge'
                elif signal_name == 'nf_loc':
                    sig = None
                    title += 'Heatmap of TX Location probability in the room'
                    xlabel_mode = 'nf_loc'
                    ylabel_mode = 'nf_loc'
                else:
                    raise ValueError('Unsupported signal name: {}'.format(signal_name))

                sig, title_post = self.signals_obj.process_sig(sig, process_list=signal_process_list)
                title += title_post
                label = "RX {}/TX {}".format(rx_id, tx_id)
                if 'real' in signal_process_list:
                    label += "-Real"
                if 'imag' in signal_process_list:
                    label += "-Imag"

                if sig_final is None:
                    sig_final = sig.copy()
                    label_final = label

                if index>0 and plot[index-1] in supported_operations:
                    operation = plot[index-1]
                    if operation == '+':
                        sig_final += sig
                    elif operation == '-':
                        sig_final -= sig
                    elif operation == '*':
                        sig_final *= sig
                    elif operation == '/':
                        sig_final /= sig

                    label_final += operation + label

                if not (len(plot) > index+1 and plot[index+1] in supported_operations):
                    plot_signals.append(PlotSignal(signal_name=signal_name, trx_id=[rx_id, tx_id], process_list=signal_process_list, x=x, data=sig_final, label=label_final))
                    sig_final = None
                    label_final = None

            title += ", RX/TX: "
            for rx_id, tx_id in zip(rx_ids, tx_ids):
                title += "{}/{}-".format(rx_id, tx_id)
            title = title[:-1]


            if xlabel_mode == 'time':
                xlabel = "Time (ns)"
            elif xlabel_mode == 'freq':
                xlabel = "Frequency (MHz)"
            elif xlabel_mode == 'time_h_sparse':
                xlabel = "Time (ns)"
            elif xlabel_mode == 'IQ':
                xlabel = "In-phase (I)"
            elif xlabel_mode == 'id':
                xlabel = "Experiment ID"
            elif xlabel_mode == 'aoa_gauge':
                xlabel = "Angle of Arrival (Deg)"
            elif xlabel_mode == 'nf_loc':
                xlabel = "X (m)"


            if ylabel_mode == 'mag':
                ylabel = "Magnitude"
            elif ylabel_mode == 'mag_db':
                ylabel = "Magnitude (dB)"
            elif ylabel_mode == 'phase':
                ylabel = "Phase (rad)"
            elif ylabel_mode == 'phase/2pi':
                ylabel = "Phase (2π)"
            elif ylabel_mode == 'IQ':
                ylabel = "Quadrature (Q)"
            elif ylabel_mode == 'snr':
                ylabel = "SNR (dB)"
            elif ylabel_mode == 'aoa_gauge':
                ylabel = "Angle of Arrival (Deg)"
            elif ylabel_mode == 'nf_loc':
                ylabel = "Y (m)"


            signals.append(PlotChart(plot_signals=plot_signals, title=title, x_label=xlabel, y_label=ylabel))

        return signals


    def toggle_pause(self, event):
        if event.key == 'p':  # Press 'p' to pause/resume
            self.anim_paused = not self.anim_paused


    def update(self, frame, rx_signal: RxSignal=None):
        if self.anim_paused:
            return self.line

        if rx_signal is None:
            raise ValueError("rx_signal cannot be None for plot_level >= 0")
        signals = self.process_signals_for_plot(rx_signal)

        line_id = 0
        for i in range(self.config.n_plots_row):
            j = self.signals_config.fc_id - 1

            for signal in signals[i].plot_signals:

                signal_name = signal.signal_name
                rx_id = signal.trx_id[0]
                tx_id = signal.trx_id[1]
                signal_data = signal.data
                signal_process_list = signal.process_list

                if 'IQ' in signal_process_list:
                    self.line[line_id][j].set_offsets(np.column_stack((signal_data.real, signal_data.imag)))
                    line_id+=1
                    margin = max(np.abs(signal_data)) * 0.1
                    self.ax[i][j].set_xlim(min(signal_data.real) - margin, max(signal_data.real) + margin)
                    self.ax[i][j].set_ylim(min(signal_data.imag) - margin, max(signal_data.imag) + margin)
                elif signal_name == 'rx_ph_diff':
                    self.line[line_id][j].set_data(np.arange(len(signal_data)), signal_data)
                    line_id+=1
                elif signal_name == 'aoa_gauge':
                    self.gauge_update_needle(self.ax[i][j], np.rad2deg(signal_data))
                    self.ax[i][j].set_xlim(0, 1)
                    self.ax[i][j].set_ylim(0.5, 1)
                    self.ax[i][j].axis('off')
                elif signal_name == 'h_sparse':
                    (h_tr, dly_est, peaks, npath_est) = signal_data
                    h_tr = h_tr[rx_id, tx_id]
                    dly_est = dly_est[rx_id, tx_id]
                    peaks = peaks[rx_id, tx_id]
                    
                    # Plot the raw response
                    dly = np.arange(self.signals_config.n_samples_ch)
                    dly = dly - self.signals_config.n_samples_ch*(dly > self.signals_config.n_samples_ch/2)
                    dly = dly / self.signals_config.fs_trx *1e9
                    chan_pow = self.signals_obj.lin_to_db(np.abs(h_tr), mode='mag')

                    # Roll the response and shift the response
                    rots = self.signals_config.n_samp_ch_sp//4
                    yshift = np.percentile(chan_pow, 25)
                    chan_powr = np.roll(chan_pow, rots) - yshift
                    dlyr = np.roll(dly, rots)
                    self.line[line_id][j].set_data(dlyr[:self.signals_config.n_samp_ch_sp], chan_powr[:self.signals_config.n_samp_ch_sp])
                    line_id+=1

                    # Compute the axes
                    ymax = np.max(chan_powr)+5
                    ymin = -10

                    # Plot the locations of the detected peaks
                    peaks_ = np.abs(peaks)**2
                    peaks_  = self.signals_obj.lin_to_db(peaks_, mode='pow')-yshift
                    dly_est = dly_est*1e9
                    dly_est = dly_est[dly_est<=np.max(dlyr[:self.signals_config.n_samp_ch_sp])]
                    self.line[line_id][j].set_data(dly_est, peaks_)
                    line_id+=1
                    self.line[line_id][j].set_segments([[[i,ymin], [i,j]] for i,j in zip(dly_est, peaks_)])
                    line_id+=1
                    self.ax[i][j].set_ylim([ymin, ymax])
                elif signal_name == 'nf_loc':
                    self.signals_obj.nf_model.plot_results(self.ax[i][j], RoomModel=self.signals_obj.RoomModel, plot_type='init_est')
                else:
                    self.line[line_id][j].set_ydata(signal_data)
                    line_id+=1


            if signal_name in self.mag_filter_list['signal_name'] or any(item in signal_process_list for item in self.mag_filter_list['process_list']):
                if len(np.array(signal_data).shape)>1:
                    sig = signal_data[0]
                else:
                    sig = signal_data.copy()
                y_min = np.percentile(sig, 10)
                y_max = np.max(sig) + 0.1*(np.max(sig)-y_min)
                self.ax[i][j].set_ylim(y_min, y_max)

            elif not (signal_name in self.untouched_plot_list['signal_name'] or any(item in signal_process_list for item in self.untouched_plot_list['process_list'])):
                try:
                    self.ax[i][j].relim()
                    self.ax[i][j].autoscale_view()
                except Exception as e:
                    print("Error in autoscale {}".format(e))

        return self.line


    def update_once(self, rx_signal: RxSignal=None):
        if rx_signal is None:
            raise ValueError("rx_signal cannot be None for plot update")

        if not self.plots_initialized:
            self.init_plots(rx_signal)
        else:
            self.update(frame=0, rx_signal=rx_signal)
            if self.fig is not None:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                plt.pause(0.001)


    def init_plots(self, rx_signal: RxSignal=None):
        if self.config.plot_level<0:
            return
        
        if rx_signal is None:
            raise ValueError("rx_signal cannot be None for plot_level >= 0")
        signals = self.process_signals_for_plot(rx_signal)
        
        # Set up the figure and plot
        self.line = [[None for j in range(self.config.n_plots_col)] for i in range(3*self.config.n_plots_row)]
        self.fig, self.ax = plt.subplots(self.config.n_plots_row, self.config.n_plots_col)
        if type(self.ax) is not np.ndarray:
            self.ax = np.array([self.ax])
        if len(self.ax.shape)<2:
            self.ax = self.ax.reshape(-1, 1)
        self.fig.canvas.mpl_connect('key_press_event', self.toggle_pause)


        for j in range(self.config.n_plots_col):
            line_id = 0
            for i in range(self.config.n_plots_row):
                for signal in signals[i].plot_signals:

                    signal_name = signal.signal_name
                    label = signal.label
                    signal_process_list = signal.process_list
                    signal_data = signal.data
                    x_data = signal.x
                
                    if 'IQ' in signal_process_list:
                        self.line[line_id][j] = self.ax[i][j].scatter(signal_data.real, signal_data.imag, facecolors='none', edgecolors='b', s=10)
                        line_id+=1
                        self.ax[i][j].axhline(0, color='black',linewidth=0.5)
                        self.ax[i][j].axvline(0, color='black',linewidth=0.5)
                        self.ax[i][j].set_aspect('equal')
                        margin = max(np.abs(signal_data)) * 0.1
                        self.ax[i][j].set_xlim(min(signal_data.real)-margin, max(signal_data.real+margin))
                        self.ax[i][j].set_ylim(min(signal_data.imag)-margin, max(signal_data.imag+margin))

                    elif signal_name=='h_sparse':
                        # (h_tr, dly_est, peaks) = signal_data
                        self.line[line_id][j], = self.ax[i][j].plot([], [])
                        line_id+=1
                        # (markerline, stemlines, baseline)
                        self.line[line_id][j], self.line[line_id+1][j], _ = self.ax[i][j].stem([0], [1], 'r-', basefmt='', bottom=-10)
                        line_id+=2

                    elif signal_name=='aoa_gauge':
                        self.signals_obj.draw_half_gauge(self.ax[i][j], min_val=-90, max_val=90)
                        self.signals_obj.gauge_update_needle(self.ax[i][j], 0, min_val=-90, max_val=90)
                        self.ax[i][j].set_xlim(0, 1)
                        self.ax[i][j].set_ylim(0.5, 1)
                        self.ax[i][j].axis('off')
                        
                    elif signal_name=='nf_loc':
                        self.ax[i][j] = self.signals_obj.nf_model.plot_results(self.ax[i][j], RoomModel=self.signals_obj.RoomModel, plot_type='init_est')
                        self.ax[i][j].set_yticks([])

                        self.ax[i][j].set_xlim(self.signals_config.nf_region[0])
                        self.ax[i][j].set_ylim(self.signals_config.nf_region[1])
                        self.ax[i][j].set_xticks(np.arange(self.signals_config.nf_region[0,0], self.signals_config.nf_region[0,1], 1.0))
                        self.ax[i][j].set_yticks(np.arange(self.signals_config.nf_region[1,0], self.signals_config.nf_region[1,1], 2.0))

                    else:
                        self.line[line_id][j], = self.ax[i][j].plot(x_data, signal_data, label=label)
                        line_id+=1


                # Truncate the title to a maximum of 30 characters
                title = (signals[i].title[:self.config.plot_configs['title_max_chars']] + '...') if len(signals[i].title) > self.config.plot_configs['title_max_chars'] else signals[i].title
                title = title + "\n Carrier Frequency: {} GHz".format(self.signals_config.freq_hop_list[j]/1e9)
                x_label = signals[i].x_label
                y_label = signals[i].y_label
                self.ax[i][j].set_title(title)
                self.ax[i][j].set_xlabel(x_label)
                self.ax[i][j].set_ylabel(y_label)

                self.ax[i][j].title.set_fontsize(self.config.plot_configs['title_size'])
                self.ax[i][j].xaxis.label.set_fontsize(self.config.plot_configs['xaxis_size'])
                self.ax[i][j].yaxis.label.set_fontsize(self.config.plot_configs['yaxis_size'])
                self.ax[i][j].tick_params(axis='both', which='major', labelsize=self.config.plot_configs['ticks_size'])  # For major ticks
                self.ax[i][j].legend(fontsize=self.config.plot_configs['legend_size'])

                self.ax[i][j].grid(True)
                if not (signal_name in self.untouched_plot_list['signal_name'] or any(item in signal_process_list for item in self.untouched_plot_list['process_list'])):
                    self.ax[i][j].relim()
                    self.ax[i][j].autoscale_view()
                self.ax[i][j].minorticks_on()

        for j in range(self.config.n_plots_col):
            for i in range(len(self.line)):
                if self.line[i][j] is not None:
                    # self.line[i][j].set_linewidth(3.0-0.5*self.n_plots_row-0.3*self.n_plots_col)
                    self.line[i][j].set_linewidth(self.config.plot_configs['line_width'])

        # Render once and keep the figure open for manual updates
        plt.tight_layout()
        plt.subplots_adjust(hspace=self.config.plot_configs['hspace'], wspace=self.config.plot_configs['wspace'])
        self.plots_initialized = True
        # anim = animation.FuncAnimation(self.fig, self.update, frames=int(1e9), interval=self.config.plot_configs['anim_interval'], blit=False)
        plt.ion()
        plt.show(block=False)
        # self.fig.savefig(self.config.plot_configs['figs_save_path'], dpi=300)



if __name__ == "__main__":
    from configs import Configs_Class
    config = Configs_Class()

    signals_inst = Signal_Utils_Rfsoc(config)
    # signals_inst.collect_signals()
    signals_inst.compute_sys_response()




