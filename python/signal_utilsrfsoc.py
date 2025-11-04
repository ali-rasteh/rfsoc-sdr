from backend import *
from backend import be_np as np, be_scp as scipy
from sigcom_toolkit.signal_utils import Signal_Utils
from sigcom_toolkit.general import General
try:
    from near_field import Sim as Near_Field_Model, RoomModel
except:
    pass



class Signal_Utils_Rfsoc(Signal_Utils):
    def __init__(self, params):
        super().__init__(params)

        self.import_attributes(params)

        self.rx_phase_offset = 0
        self.rx_delay_offset = 0
        self.fc_id = 0
        self.rot_angle_id = 0
        self.nf_loc_idx = 0
        self.nf_sep_idx = 0
        self.rx_phase_list = []
        self.aoa_list = []
        self.lin_track_dir = 'forward'
        self.tx_rx_distance = 3.0

        self.print("signals object initialization done", thr=1)
        

    def gen_tx_signal(self):
        txtd_base = []
        txtd = []
        for ant_id in range(self.n_tx_ant):
            if 'tone' in self.sig_mode:
                if self.sig_mode=='tone_1':
                    nsc = 1
                elif self.sig_mode=='tone_2':
                    nsc = 2
                txtd_base_s = self.generate_tone(freq_mode=self.tone_f_mode, sc=self.sc_tone, f=self.f_tone, sig_mode=self.sig_mode, gen_mode=self.sig_gen_mode)
            elif 'wideband' in self.sig_mode:
                nsc = self.wb_sc_range[1] - self.wb_sc_range[0] + 1
                txtd_base_s = self.generate_wideband(bw_mode=self.wb_bw_mode, sc_range=self.wb_sc_range, bw_range=self.wb_bw_range, modulation=self.sig_modulation, sig_mode=self.sig_mode, gen_mode=self.sig_gen_mode, seed=self.seed_list[ant_id])
            elif self.sig_mode == 'load':
                txtd_base_s = np.load(self.sig_path)
            else:
                raise ValueError('Unsupported signal mode: ' + self.sig_mode)
            txtd_base_s /= np.max([np.abs(txtd_base_s.real), np.abs(txtd_base_s.imag)])
            txtd_base_s *= self.db_to_lin(self.sig_gain_db, mode='mag')
            txtd_base.append(txtd_base_s)

            self.sig_pow_dbm = self.lin_to_db(0.5 * 1000, mode='pow') + self.sig_gain_db
            bw = (nsc/self.nfft_tx) * self.fs_tx
            self.sig_psd_dbm = self.sig_pow_dbm - self.lin_to_db(bw, mode='pow')
            self.sig_psd_dbm_sc = self.sig_pow_dbm - self.lin_to_db(nsc, mode='pow')
            print('TX Signal power for antenna {}: {:0.3f} dbm'.format(ant_id, self.sig_pow_dbm))
            print('TX Signal PSD for antenna {}: {:0.3f} dBm/Hz = {:0.3f} dBm/MHz = {:0.3f} dBm/sc'.format(ant_id, self.sig_psd_dbm, self.sig_psd_dbm+self.lin_to_db(1e6, mode='pow'), self.sig_psd_dbm_sc))

            title = 'TX signal spectrum in base-band for antenna {}'.format(ant_id)
            xlabel = 'Frequency (MHz)'
            ylabel = 'Magnitude (dB)'
            self.plot_signal(x=self.freq_tx, sigs=txtd_base[ant_id], mode='fft', scale='dB20', title=title, xlabel=xlabel, ylabel=ylabel, plot_level=4)
            title = 'Base-band TX signal in time domain at \n the time transition for antenna {}'.format(ant_id)
            xlabel = 'Time (s)'
            ylabel = 'Magnitude'
            n=int(np.round(self.fs_tx/self.f_max))
            t=self.t_tx[:2*n]
            sig_real=np.concatenate((txtd_base[ant_id].real[-n:], txtd_base[ant_id].real[:n]))
            sig_imag=np.concatenate((txtd_base[ant_id].imag[-n:], txtd_base[ant_id].imag[:n]))
            self.plot_signal(x=t, sigs={'real':sig_real, 'imag':sig_imag}, mode='time', scale='linear', title=title, xlabel=xlabel, ylabel=ylabel, plot_level=4, legend=True)

        txtd_base = np.array(txtd_base)

        if self.tx_sig_sim == 'shifted':
            # txtd_base[1,:] = txtd_base[0,:].copy()
            # txtd_base[1,:] = np.roll(txtd_base[0,:], shift=(self.n_samples_tx//2), axis=-1)
            txtd_base[1,:] = np.roll(txtd_base[0,:], shift=(384), axis=-1)

            # dot_prod = []
            # for i in range(0, self.n_samples_tx, 1):
            #     txtd_base[1,:] = np.roll(txtd_base[0,:], shift=(i), axis=-1)
            #     dot_prod.append(np.abs(np.vdot(txtd_base[1], txtd_base[0])))
            # dot_prod = np.array(dot_prod)
            # print(np.argsort(dot_prod)[:20])
            # print(np.sort(dot_prod)[:20])


        if self.mixer_mode=='digital' and self.mix_freq!=0:
            for ant_id in range(self.n_tx_ant):
                txtd_s = self.freq_shift(txtd_base[ant_id], shift=self.mix_freq, fs=self.fs_tx)
                txtd.append(txtd_s)
            
                # txfd = np.abs(fftshift(fft(txtd)))
                title = 'TX signal spectrum after upconversion for antenna {}'.format(ant_id)
                xlabel = 'Frequency (MHz)'
                ylabel = 'Magnitude (dB)'
                self.plot_signal(x=self.freq_tx, sigs=txtd[ant_id], mode='fft', scale='dB20', title=title, xlabel=xlabel, ylabel=ylabel, plot_level=4)
        else:
            txtd = txtd_base.copy()
            # txfd = txfd_base.copy()

        txtd = np.array(txtd)
        
        if self.beamforming:
            txtd_base = self.beam_form(txtd_base)
            txtd = self.beam_form(txtd)

        print("Dot product of transmitted signals: ", np.abs(np.vdot(txtd_base[1], txtd_base[0])))
        # print("Correlation of transmitted signals: ", np.max(np.abs(np.correlate(txtd_base[0], txtd_base[1], mode='full'))))
        # self.plot_signal(sigs = np.abs(np.correlate(txtd_base[1,:], txtd_base[0,:], mode='full')))

        return (txtd_base, txtd)


    def create_near_field_model(self):
        self.RoomModel = RoomModel(xlim=self.nf_walls[0], ylim=self.nf_walls[1])
        # # Place a source
        # xsrc = np.array([2,4])
        # # Find the reflections
        # xref = self.RoomModel.find_reflection(xsrc)
        # # Create all the transmitters
        # xtx =  np.vstack((xsrc, xref))

        self.nf_region = self.nf_walls.copy()
        room_width = self.nf_walls[0,1] - self.nf_walls[0,0]
        room_length = self.nf_walls[1,1] - self.nf_walls[1,0]
        self.nf_region[0,0] -= room_width
        self.nf_region[0,1] += room_width
        # self.nf_region[1,0] -= room_length
        self.nf_region[1,1] += room_length
        self.nf_model = Near_Field_Model(fc=self.fc, fsamp=self.fs_rx, nfft=self.nfft_ch, nantrx=self.n_rx_ant,
                        rxlocsep=self.nf_rx_loc_sep, sepdir=self.nf_rx_sep_dir, antsep=self.nf_rx_ant_sep, npath_est=self.nf_npath_max[1], 
                        stop_thresh=self.nf_stop_thr, region=self.nf_region, tx=self.nf_tx_loc)
        
        self.nf_model.gen_tx_pos()
        self.nf_model.compute_rx_pos()
        self.nf_model.compute_freq_resp()
        self.nf_model.create_tx_test_points()
        self.nf_model.path_est_init()
        self.nf_model.locate_tx()
        # self.nf_model.plot_results(RoomModel=self.RoomModel, plot_type='init_est')

        self.nf_rx_loc = self.nf_model.rxloc
        self.nf_rx_ant_pos = self.nf_model.rxantpos

        self.print("Near field model created", thr=1)
    


    def handle_nf(self, h_est_full, sparse_est_params):
        if self.nf_param_estimate:
            # h_index = self.animate_plot_mode.index('h')
            if self.nf_loc_idx==0:
                self.nf_sep_idx = 0

                if self.use_linear_track:
                    self.client_lintrack.return2home(lin_track_id=0)
                    self.client_lintrack.return2home(lin_track_id=1)
                    time.sleep(0.5)
                    # distance = -1000*(len(self.nf_rx_loc)-1)
                    # distance = np.round(distance, 2)
                    # client_lintrack.move(lin_track_id=0, distance=distance)
                    # time.sleep(0.1)
                self.h_nf = []
                self.dly_est_nf = []
                self.peaks_nf = []
                self.npaths_nf = []
                self.nf_loc_idx+=1
                self.nf_sep_idx+=1

            elif self.nf_loc_idx==len(self.nf_rx_loc)+1:
                self.h_nf = np.array(self.h_nf)
                self.dly_est_nf = np.array(self.dly_est_nf)
                self.peaks_nf = np.array(self.peaks_nf)
                self.npaths_nf = np.array(self.npaths_nf)
                self.est_nf_param(self.h_nf, self.dly_est_nf, self.peaks_nf, self.npaths_nf)
                self.nf_loc_idx = 0
                self.nf_sep_idx = 0
            else:

                if self.nf_sep_idx==0:
                    if self.use_linear_track:
                        distance = 1000*(self.nf_rx_ant_sep[0]*self.wl - self.nf_rx_ant_sep[-1]*self.wl)
                        distance = np.round(distance, 2)
                        self.client_lintrack.move(lin_track_id=1, distance=distance)
                        time.sleep(0.5)
                        self.ant_dx = self.nf_rx_ant_sep[0]

                        if self.nf_loc_idx < len(self.nf_rx_loc):
                            distance = 1000*(self.nf_rx_loc[self.nf_loc_idx,0] - self.nf_rx_loc[self.nf_loc_idx-1,0])
                            distance = np.round(distance, 2)
                            self.client_lintrack.move(lin_track_id=1, distance=distance)
                            self.client_lintrack.move(lin_track_id=0, distance=distance)
                            time.sleep(0.5)
                            
                    self.nf_sep_idx+=1
                    self.nf_loc_idx+=1
                elif self.nf_sep_idx==len(self.nf_rx_ant_sep)+1:
                    self.nf_sep_idx = 0
                else:
                    self.h_nf.append(h_est_full)
                    (h_tr, dly_est, peaks, npath_est) = sparse_est_params
                    self.dly_est_nf.append(dly_est)
                    self.peaks_nf.append(peaks)
                    self.npaths_nf.append(npath_est)

                    if self.use_linear_track:
                        if self.nf_sep_idx < len(self.nf_rx_ant_sep):
                            distance = 1000*(self.nf_rx_ant_sep[self.nf_sep_idx]*self.wl - self.nf_rx_ant_sep[self.nf_sep_idx-1]*self.wl)
                            distance = np.round(distance, 2)
                            self.client_lintrack.move(lin_track_id=1, distance=distance)
                            time.sleep(0.5)
                            self.ant_dx = self.nf_rx_ant_sep[self.nf_sep_idx]
                    
                    self.nf_sep_idx+=1
            
                self.ant_dx_m = self.ant_dx * self.wl



    def est_nf_param(self, h, dly_est, peaks, npaths):
        """
        Parameters
        -------
        h : np.array of shape (nfft,n_rx,n_meas)
            The channel frequency response.
        """

        h = np.transpose(h.copy(), (3,1,2,0))
        dly_est = np.transpose(dly_est.copy(), (3,1,2,0))
        peaks = np.transpose(peaks.copy(), (3,1,2,0))
        npaths = np.transpose(npaths.copy(), (1,2,0))
        n_paths_min = np.min(npaths)

        # Sort delay and peaks of each measurement based on the paths delays
        dly_sort_idx = np.argsort(dly_est, axis=0)
        dly_est = np.take_along_axis(dly_est, dly_sort_idx, axis=0)
        peaks = np.take_along_axis(peaks, dly_sort_idx, axis=0)


        # self.plot_signal(self.t_trx[:100], np.abs(h[:100,1,1,0]), scale='dB20')

        txid = 0

        self.nf_model.chan_td = h[:,:,txid,:]
        self.nf_model.chan_fd = fft(h[:,:,txid,:], axis=0)
        self.nf_model.sparse_dly_est = dly_est[:,:,txid,:]
        self.nf_model.sparse_peaks_est = peaks[:,:,txid,:]
        # self.nf_model.npath_est = n_paths_min
        self.print("Number of paths estimated: {}".format(n_paths_min), thr=0)

        self.nf_model.path_est_init()
        self.nf_model.locate_tx(npath_est=n_paths_min)
        # self.nf_model.plot_results(RoomModel=self.RoomModel, plot_type='')

        n_epochs = 1000
        lr_init = 0.1
        H_gt = fft(h.copy(), axis=0)
        tx_ant_vec = self.nf_tx_ant_loc[:,:,:] - (self.nf_tx_ant_loc[0,0,:])[None,None,:] + 0.01
        rx_ant_vec = self.nf_rx_ant_loc[:,:,:] - (self.nf_rx_ant_loc[0,0,:])[None,None,:]
        phase_diff = np.angle(peaks[:n_paths_min,0,0,:] * np.conj(peaks[:n_paths_min,1,0,:]))
        # phase_diff = np.mean(phase_diff, axis=-1)
        aoa = np.zeros(phase_diff.shape)
        for m in range(phase_diff.shape[-1]):
            ant_dx_m = np.linalg.norm(self.nf_rx_ant_loc[1,m,:] - self.nf_rx_ant_loc[0,m,:], axis=-1)
            aoa[:,m] = self.phase_to_aoa(phase_diff[:,m], wl=self.wl, ant_dim=self.ant_dim, ant_dx_m=ant_dx_m, ant_dy_m=self.ant_dy_m)
            # aoa = self.phase_to_aoa(phase_diff, wl=self.wl, ant_dim=self.ant_dim, ant_dx_m=self.ant_dx_m, ant_dy_m=self.ant_dy_m)
        trx_unit_vec = np.stack((np.sin(aoa), np.cos(aoa)), axis=-1)
        # print("phase_diff: ", phase_diff[:,0])
        # print("aoa: ", aoa[:,0])
        # print("trx_unit_vec: ", trx_unit_vec[:,0,:])
        path_delay = self.nf_model.abs_delay.copy()[:n_paths_min,:,None,:] * np.ones(dly_est[:n_paths_min].shape)
        path_gain = peaks.copy()[:n_paths_min]
        # print("path_delay: ", path_delay[:,0,0,0])
        # path_delay = None
        # path_gain = None
        freq = self.freq_ch.copy()
        self.nf_model.nf_channel_param_est(n_paths=n_paths_min, n_epochs=n_epochs, lr_init=lr_init, H_gt=H_gt, tx_ant_vec=tx_ant_vec, rx_ant_vec=rx_ant_vec, trx_unit_vec=trx_unit_vec, path_delay=path_delay, path_gain=path_gain, freq=freq)



    def calibrate_rx_phase_offset(self, client_rfsoc):
        '''
        This function calibrates the phase offset between the receivers ports in RFSoCs
        '''
        input_ = input("Press Y for phase offset calibration (and position the TX/RX at AoA = 0) or any key to use the saved phase offset: ")

        if input_.lower()!='y':
            if os.path.exists(self.calib_params_path):
                self.rx_phase_offset = np.load(self.calib_params_path)['rx_phase_offset']
                self.rx_delay_offset = np.load(self.calib_params_path)['rx_delay_offset']
                # self.print("Using saved phase offset between RX ports: {:0.3f} Rad".format(self.rx_phase_offset), thr=1)
                self.print("Using saved delay offset between RX ports: {:0.3f} s".format(self.rx_delay_offset), thr=1)
            else:
                self.print("No saved calibration found, please calibrate the phase offset", thr=0)
                self.rx_phase_offset = 0
                self.rx_delay_offset = 0
            return
        else:
            phase_diff_list = []
            delay_list = []
            for i in range(self.calib_iter):
                rxtd = self.receive_data(client_rfsoc, mode='once')
                rxtd = rxtd[0]
                phase_diff = self.calc_phase_offset(rxtd[0,:], rxtd[1,:])
                delay = phase_diff / (2*np.pi*self.fc)
                phase_diff_list.append(phase_diff)
                delay_list.append(delay)

            self.rx_phase_offset = np.mean(phase_diff_list)
            self.rx_delay_offset = np.mean(delay_list)
            np.savez(self.calib_params_path, rx_phase_offset=self.rx_phase_offset, rx_delay_offset=self.rx_delay_offset, fc=self.fc)
            # self.print("Calibrated and saved phase offset between RX ports: {:0.3f} Rad".format(self.rx_phase_offset), thr=1)
            self.print("Calibrated and saved delay offset between RX ports: {:0.3f} s".format(self.rx_delay_offset), thr=1)



    def validate_saved_signals(self, rxtd, txtd=None, thr = 1e-8):
        self.print("Sanity check for saved signals", thr=2)

        mses = []
        for i in range(self.n_save):
            mse = self.mse(rxtd[i,0], rxtd[i,1])
            mses.append(mse)
            mse = self.mse(rxtd[i-self.n_save//self.n_frame_rd,0], rxtd[i,0])
            mses.append(mse)
            mse = self.mse(rxtd[i-self.n_save//self.n_frame_rd,1], rxtd[i,1])
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
        self.sys_response = np.load(self.sys_response_path)['h_est_full_avg']
        self.sys_response /= np.max(np.abs(self.sys_response))


    def compute_sys_response(self):
        n_rx = 1

        if n_rx == 1:
            sys_response_folder = os.path.join(os.getcwd(), 'sigs_tx1_rx1_rx_rotate/')
        elif n_rx == 2:
            sys_response_folder = os.path.join(os.getcwd(), 'sigs_tx2_rx2_rx_rotate/')

        n_tx = n_rx
        postfix = "{}x{}".format(n_rx, n_tx)
        sys_response_path = os.path.join(self.channel_dir, 'sys_response_{}.npz'.format(postfix))
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
                        (rxtd_base, h_est_full, H_est, H_est_max, sparse_est_params) = self.rx_operations(txtd_base, rxtd)
                        max_gain = np.max(np.abs(h_est_full), axis=-1)
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
        plt.savefig(os.path.join(self.figs_dir, 'sys_response_vs_freq_{}.pdf'.format(postfix)))


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
        plt.savefig(os.path.join(self.figs_dir, 'sys_response_vs_angle_{}.pdf'.format(postfix)))


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
            plt.savefig(os.path.join(self.figs_dir, 'sys_response_2D_{}_RX{}.pdf'.format(postfix, rx_id)))
            plt.show()





    def collect_signals(self):
        collect_count = 512
        ignore_less_count = False
        # input_folder = self.channel_dir
        input_folder = self.sig_dir
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


    def save_signal_channel(self, client_rfsoc, client_turntable, client_piradio, client_controller, txtd_base, save_list=[]):
        rx_chain_main = self.rx_chain.copy()
        if 'sys_res_deconv' in self.rx_chain:
            self.rx_chain.remove('sys_res_deconv')
        if 'sparse_est' in self.rx_chain:
            self.rx_chain.remove('sparse_est')


        for config in self.measurement_configs:
            tx_rx_distance = input('Please change the configuration to: {} and enter the TX to RX distance in meters (empty for default): '.format(config))
            if tx_rx_distance != '':
                try:
                    tx_rx_distance = float(tx_rx_distance)
                except:
                    raise ValueError('Invalid distance value: {}'.format(tx_rx_distance))
                self.tx_rx_distance = tx_rx_distance
            if self.set_piradio_opt_gains:
                # self.find_optimal_gain_piradio(client_rfsoc, client_piradio, client_controller)
                self.set_optimal_gain_piradio(client_piradio, client_controller)


            self.print("Starting to save signals for configuration: {}".format(config), thr=0)

            if 'calib' in config:
                rotation_angles = [0]
                use_turntable = False
                rotation_delay = 0
                mode = 'calib'
            else:
                rotation_angles = self.rotation_angles
                use_turntable = self.use_turntable
                rotation_delay = client_turntable.rotation_delay if use_turntable else 0
                mode = 'measurement'

            rotation_time = 1.514 + rotation_delay
            freq_switch_time = 0.052 + self.piradio_freq_sw_dly_default
            total_time = len(rotation_angles) * (rotation_time + len(self.freq_hop_list)*(freq_switch_time))
            self.print("Anticipated time to save signals: {:0.0f} s".format(total_time), thr=0)
            

            for angle_id in range(len(rotation_angles)):

                remaining_time = (len(rotation_angles) - angle_id) * (rotation_time + len(self.freq_hop_list)*(freq_switch_time))
                self.print("Remaining time to save signals: {:0.0f} s".format(remaining_time), thr=0)

                angle = rotation_angles[angle_id]
                self.print("Rotating to angle: {}".format(angle), thr=0)
                if use_turntable:
                    start_time = time.time()
                    client_turntable.move_to_position(angle)
                    rotation_time = time.time()-start_time
                    self.print("Time taken to rotate: {:0.3f} s".format(rotation_time), thr=2)


                measurements = {}
                for freq_id in range(len(self.freq_hop_list)):
                    frequency = self.freq_hop_list[freq_id]
                    self.print("Saving signals for Freq: {} GHz".format(frequency/1e9), thr=0)

                    start_time = time.time()
                    self.hop_freq(client_piradio, client_controller, fc_id=freq_id)

                    # test = np.load(self.sig_save_path)
                    rxtd_save=[]
                    h_est_full_save=[]
                    H_est_save=[]
                    H_est_max_save=[]
                        
                    if 'channel' in save_list:
                        n_rd_rep = self.n_save
                    else:
                        n_rd_rep = self.n_save//self.n_frame_rd
                    rxtd = self.receive_data(client_rfsoc, n_rd_rep=n_rd_rep, mode='once', verbose=False)
                    # raise ValueError('Stop')
                    
                    if 'channel' in save_list:
                        for i in range(self.n_save):
                            self.print("Channel Save Iteration: {}".format(i+1), thr=0)
                            rxtd = self.receive_data(client_rfsoc, n_rd_rep=n_rd_rep, mode='once')

                            # to handle the dimenstion needed for read repeat
                            (rxtd_base, h_est_full, H_est, H_est_max, sparse_est_params) = self.rx_operations(txtd_base, rxtd[i])

                            rxtd_save.append(rxtd_base)
                            
                            h_est_full_save.append(h_est_full)
                            H_est_save.append(H_est)
                            H_est_max_save.append(H_est_max)

                        h_est_full_save = np.array(h_est_full_save)
                        H_est_save = np.array(H_est_save)
                        H_est_max_save = np.array(H_est_max_save)

                        # h_est_full_avg = np.mean(h_est_full_save, axis=0)
                        # rxtd_avg = np.mean(rxtd_save, axis=0)
                        # self.rx_chain = ['channel_est']
                        # (rxtd_avg, h_est_full_avg, H_est_avg, H_est_max_avg, sparse_est_params) = self.rx_operations(txtd_base, rxtd_avg)
                    else:
                        rxtd_save = np.empty((self.n_save, self.n_rx_ant, self.n_samples_tx), dtype=rxtd.dtype)
                        for i in range(self.n_frame_rd):
                            rxtd_save[i::self.n_frame_rd] = rxtd[:,:,i*self.n_samples_tx:(i+1)*self.n_samples_tx]
                        # print(rxtd_save.shape)

                        # for i in range(self.n_frame_rd):
                        #     if rxtd_save is None:
                        #         rxtd_save = rxtd[:,:,i*self.n_samples_tx:(i+1)*self.n_samples_tx]
                        #     else:
                        #         rxtd_save = np.vstack((rxtd_save, rxtd[:,:,i*self.n_samples_tx:(i+1)*self.n_samples_tx]))
                        #     print(rxtd_save.shape)


                    txtd_save = np.expand_dims(txtd_base, axis=0)
                    rxtd_save = np.array(rxtd_save)

                    self.validate_saved_signals(rxtd=rxtd_save)

                    if 'signal' in save_list:
                        measurements['txtd'] = txtd_save
                        measurements['rxtd_{}'.format(frequency/1e9)] = rxtd_save.copy()
                    if 'channel' in save_list:
                        measurements['h_est_full_{}'.format(frequency/1e9)] = h_est_full_save.copy()

                    freq_switch_time = time.time()-start_time
                    self.print("Time taken to save signals: {:0.3f} s".format(freq_switch_time), thr=2)


                postfix = config
                if self.measurement_type == 'FR3_nyu_3state':
                    if mode != 'calib':
                        angles_dict = {0: 'alpha', 45: 'beta', -45: 'gamma'}
                        postfix = postfix.replace('<rxorient>', angles_dict[angle])
                        # save_name = f'{frequency/1e9}' + postfix + '.' + self.save_format
                    save_name = postfix + '.' + self.save_format
                elif self.measurement_type == 'FR3_nyu_13state':
                    if mode != 'calib':
                        postfix = postfix.replace('<rxorient>', str(angle))
                        # save_name = f'{frequency/1e9}' + postfix + '.' + self.save_format
                    save_name = postfix + '.' + self.save_format
                elif self.measurement_type == 'FR3_ant_calib' or self.measurement_type == 'FR3_beamforming':
                    if mode != 'calib':
                        save_name = '{}_'.format(angle) + postfix + '.' + self.save_format
                    else:
                        save_name = postfix + '.' + self.save_format
                else:
                    save_name = postfix + '.' + self.save_format
                        
                        
                if 'signal' in save_list:
                    sig_save_path=os.path.join(self.sig_dir, save_name)
                    if self.save_format == 'npz':
                        # np.savez(sig_save_path, txtd=txtd_save, rxtd=rxtd_save)
                        np.savez(sig_save_path, **measurements)
                    elif self.save_format == 'mat':
                        scipy.io.savemat(sig_save_path, measurements)
                if 'channel' in save_list:
                    channel_save_path=os.path.join(self.channel_dir, save_name)
                    if self.save_format == 'npz':
                        # np.savez(channel_save_path, h_est_full=h_est_full_save, h_est_full_avg=h_est_full_avg, H_est=H_est_save, H_est_max=H_est_max_save)
                        # np.savez(channel_save_path, h_est_full=h_est_full_save)
                        np.savez(channel_save_path, **measurements)
                    elif self.save_format == 'mat':
                        scipy.io.savemat(channel_save_path, measurements)


        self.rx_chain = rx_chain_main.copy()
    

    def receive_data(self, client_rfsoc, n_rd_rep=1, mode='once', verbose=False):
        rxtd=[]
        for i in range(n_rd_rep):
            if verbose:
                self.print("Reading iteration: {}".format(i+1), thr=0)
            rxtd_ = client_rfsoc.receive_data(mode=mode)
            rxtd_ = rxtd_.squeeze(axis=0)
            rxtd.append(rxtd_)
        rxtd = np.array(rxtd)
        return rxtd


    def hop_freq(self, client_piradio, client_controller, fc_id=None, freq=None):
            if fc_id is not None:
                fc_id = fc_id
            else:
                fc_id = (self.fc_id + 1) % len(self.freq_hop_list)
            if freq is not None:
                fc = freq
            else:
                fc = self.freq_hop_list[int(fc_id)]
            if self.fc != fc:
                if self.control_piradio:
                    client_piradio.set_frequency(fc=fc)
                    if 'master' in self.mode:
                        client_controller.set_frequency_piradio(fc=fc)

                    if self.set_piradio_opt_losupp:
                        self.set_optimal_losupp_piradio(client_piradio, client_controller, fc=fc)

                    # if client_piradio.freq_sw_dly == 0:
                    #     sleep_time = max(self.piradio_bias_sw_dly, self.piradio_freq_sw_dly)
                    #     time.sleep(self.piradio_freq_sw_dly)

                self.fc_id = fc_id
                self.fc = fc
                self.wl = self.c / self.fc



    def find_optimal_gain_piradio(self, client_rfsoc, client_piradio, client_controller):

        if os.path.exists(self.optimal_gains_path):
            self.optimal_gains = self.load_dict_from_json(self.optimal_gains_path, convert_values=True)
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
            # self.tx_rx_distance = 3.0
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

        # freq_step = 0.5
        # freq_list = np.arange(client_piradio.freq_range[0], client_piradio.freq_range[1]+freq_step, freq_step)
        freq_list = [self.stable_fc_piradio]
        for frequency in freq_list:
            self.print("Finding gains for frequency: {} GHz".format(frequency), thr=1)
            self.hop_freq(client_piradio, client_controller, freq=frequency)

            self.optimal_gains[self.tx_rx_distance][frequency] = {}
        
            snr_dB_optimal = 0
            tx_gain_dB_optimal = 0
            rx_gain_dB_optimal = 0

            for tx_gain_dB in tx_gain_dB_list:
                if tx_gain_dB < min_tx_gain_dB or tx_gain_dB > max_tx_gain_dB:
                    continue
                self.print("Setting TX gain to {} dB".format(tx_gain_dB), thr=1)
                if 'master' in self.mode:
                    client_controller.set_gain_piradio(trx='tx', chan=0, gain_db=tx_gain_dB)
                    client_controller.set_gain_piradio(trx='tx', chan=1, gain_db=tx_gain_dB)

                for rx_gain_dB in rx_gain_dB_list:
                    if rx_gain_dB < min_rx_gain_dB or rx_gain_dB > max_rx_gain_dB:
                        continue
                    if tx_gain_dB + rx_gain_dB > max_total_gain_dB:
                        continue

                    self.print("Setting RX gain to {} dB".format(rx_gain_dB), thr=1)
                    client_piradio.set_gain(trx='rx', chan=0, gain_db=rx_gain_dB)
                    client_piradio.set_gain(trx='rx', chan=1, gain_db=rx_gain_dB)
                    if client_piradio.gain_sw_dly == 0:
                        time.sleep(2*self.piradio_gain_sw_dly_default)

                    rxtd = self.receive_data(client_rfsoc, mode='once')
                    snr = self.calculate_snr(sig_td=rxtd[0,:,:self.n_samples_trx], sig_sc_range=self.sc_range)
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


        self.save_dict_to_json(self.optimal_gains, self.optimal_gains_path)
        self.print("Calculated and saved optimal TX/RX gains...", thr=1)

        return self.optimal_gains
    


    def set_optimal_gain_piradio(self, client_piradio, client_controller):
        self.print("Setting optimal TX/RX gains in Pi-Radio", thr=0)

        freq_list = list(self.optimal_gains[self.tx_rx_distance].keys())
        nearest_fc = min(freq_list, key=lambda x: abs(x - self.stable_fc_piradio/1e9))

        rx_gain_optimal = self.optimal_gains[self.tx_rx_distance][nearest_fc]['rx_gain']
        tx_gain_optimal = self.optimal_gains[self.tx_rx_distance][nearest_fc]['tx_gain']

        client_piradio.set_gain(trx='rx', chan=0, gain_db=rx_gain_optimal)
        client_piradio.set_gain(trx='rx', chan=1, gain_db=rx_gain_optimal)
        client_piradio.set_gain(trx='tx', chan=0, gain_db=tx_gain_optimal)
        client_piradio.set_gain(trx='tx', chan=1, gain_db=tx_gain_optimal)
        if 'master' in self.mode:
            client_controller.set_gain_piradio(trx='tx', chan=0, gain_db=tx_gain_optimal)
            client_controller.set_gain_piradio(trx='tx', chan=1, gain_db=tx_gain_optimal)

        # if client_piradio.gain_sw_dly == 0:
        #     time.sleep(self.piradio_gain_sw_dly)



    def set_optimal_losupp_piradio(self, client_piradio, client_controller, fc=None):
        if fc is None:
            fc = self.fc

        self.print("Setting optimal LO suppression for TX and RX in Pi-Radio", thr=1)
        
        lo_supp_lut = {6.5: [-0.026, -0.021], 7.5: [-0.025, -0.016], 8.5: [-0.001, -0.036], 9.5: [0.078, -0.045],
                        10.5: [0.192, -0.146], 11.5: [0.113, -0.08], 12.5: [0.055, -0.03], 13.5: [0.04, 0.008],
                        14.5: [0.016, -0.002], 15.5: [-0.002, -0.022], 16.5: [0.004, -0.065], 17.5: [0.034, -0.065],
                        18.5: [0.049, -0.005], 19.5: [0.075, 0.003], 20.5: [0.116, 0.049], 21.5: [0.07, 0.027], 22.5: [-0.025, -0.027]}
        
        nearest_fc = min(lo_supp_lut.keys(), key=lambda x: abs(x - fc / 1e9))
        optimal_lo_supp = lo_supp_lut[nearest_fc]

        self.print("Nearest frequency: {} GHz, Optimal LO suppression: {}".format(nearest_fc, optimal_lo_supp), thr=1)
        client_piradio.set_bias(chan=0, iq='I', bias_voltage=optimal_lo_supp[0])
        client_piradio.set_bias(chan=0, iq='Q', bias_voltage=optimal_lo_supp[1])
        client_piradio.set_bias(chan=1, iq='I', bias_voltage=optimal_lo_supp[0])
        client_piradio.set_bias(chan=1, iq='Q', bias_voltage=optimal_lo_supp[1])
        if 'master' in self.mode:
            client_controller.set_bias_piradio(chan=0, iq='I', bias_voltage=optimal_lo_supp[0])
            client_controller.set_bias_piradio(chan=0, iq='Q', bias_voltage=optimal_lo_supp[1])
            client_controller.set_bias_piradio(chan=1, iq='I', bias_voltage=optimal_lo_supp[0])
            client_controller.set_bias_piradio(chan=1, iq='Q', bias_voltage=optimal_lo_supp[1])



    def rx_operations(self, txtd_base, rxtd):
        # Expand the dimension for 1 frame received signals
        if len(rxtd.shape)<3:
            rxtd = np.expand_dims(rxtd, axis=0)
        sparse_est_params = None
        plt_frm_id = self.plt_frame_id
        n_rd_rep = rxtd.shape[0]

        for ant_id in range(self.n_rx_ant):
            title = 'RX signal spectrum for antenna {}'.format(ant_id)
            xlabel = 'Frequency (MHz)'
            ylabel = 'Magnitude (dB)'
            self.plot_signal(x=self.freq_rx, sigs=rxtd[plt_frm_id, ant_id], mode='fft', scale='dB20', title=title, xlabel=xlabel, ylabel=ylabel, plot_level=4)

            title = 'RX signal in time domain (zoomed) for antenna {}'.format(ant_id)
            xlabel = 'Time (s)'
            ylabel = 'Magnitude'
            n = 4*int(np.round(self.fs_rx/self.f_max))
            self.plot_signal(x=self.t_rx[:n], sigs=rxtd[plt_frm_id, ant_id,:n], mode='time_IQ', scale='linear', title=title, xlabel=xlabel, ylabel=ylabel, legend=True, plot_level=4)

        if self.mixer_mode == 'digital' and self.mix_freq!=0:
            rxtd_base = np.zeros_like(rxtd)
            for ant_id in range(self.n_rx_ant):
                for frm_id in range(n_rd_rep):
                    rxtd_base[frm_id, ant_id,:] = self.freq_shift(rxtd[frm_id, ant_id], shift=-1*self.mix_freq, fs=self.fs_rx)

                title = 'RX signal spectrum after downconversion for antenna {}'.format(ant_id)
                xlabel = 'Frequency (MHz)'
                ylabel = 'Magnitude (dB)'
                self.plot_signal(x=self.freq_rx, sigs=rxtd_base[plt_frm_id, ant_id], mode='fft', scale='dB20', title=title, xlabel=xlabel, ylabel=ylabel, plot_level=4)
        else:
            rxtd_base = rxtd.copy()

        if 'filter' in self.rx_chain:
            for ant_id in range(self.n_rx_ant):
                for frm_id in range(n_rd_rep):
                    cf = (self.filter_bw_range[0]+self.filter_bw_range[1])/2
                    cutoff = self.filter_bw_range[1] - self.filter_bw_range[0]
                    rxtd_base[frm_id, ant_id,:] = self.filter(rxtd_base[frm_id, ant_id,:], center_freq=cf, cutoff=cutoff, fil_order=64, plot=False)

                title = 'RX signal spectrum after filtering in base-band for antenna {}'.format(ant_id)
                xlabel = 'Frequency (MHz)'
                ylabel = 'Magnitude (dB)'
                self.plot_signal(x=self.freq_rx, sigs=rxtd_base[0, ant_id], mode='fft', scale='dB20', title=title, xlabel=xlabel, ylabel=ylabel, plot_level=4)

        for ant_id in range(self.n_rx_ant):
            # n_samples = min(len(txtd_base), len(rxtd_base))
            txfd_base_ = np.abs(fftshift(fft(txtd_base[ant_id,:self.n_samples])))
            rxfd_base_ = np.abs(fftshift(fft(rxtd_base[plt_frm_id, ant_id,:self.n_samples])))

            title = 'TX and RX signals spectrum in base-band for antenna {}'.format(ant_id)
            xlabel = 'Frequency (MHz)'
            ylabel = 'Magnitude (dB)'
            scale = np.max(txfd_base_)/np.max(rxfd_base_)
            self.print("TX to RX spectrum scale for antenna {}: {:0.3f}".format(ant_id, scale), thr=4)
            xlim=(-2*self.f_max/1e6, 2*self.f_max/1e6)
            f1=np.abs(self.freq - xlim[0]).argmin()
            f2=np.abs(self.freq - xlim[1]).argmin()
            ylim=(np.min(rxfd_base_[f1:f2]*scale), 1.1*np.max(rxfd_base_[f1:f2]*scale))
            self.plot_signal(x=self.freq, sigs={"txfd_base":txfd_base_, "Scaled rxfd_base":rxfd_base_*scale}, scale='dB20', title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, legend=True, plot_level=5)
            self.print("txfd_base max freq for antenna {}: {} MHz".format(ant_id, self.freq[(self.nfft>>1)+np.argmax(txfd_base_[self.nfft>>1:])]), thr=4)
            self.print("rxfd_base max freq for antenna {}: {} MHz".format(ant_id, self.freq[(self.nfft>>1)+np.argmax(rxfd_base_[self.nfft>>1:])]), thr=4)


        if 'pilot_separate' in self.rx_chain:
            n_samples_rx = self.n_samples_trx * 2
        else:
            n_samples_rx = self.n_samples_trx

        txtd_base = txtd_base[:,:self.n_samples_trx]
        if 'integrate' in self.rx_chain:
            rxtd_base = self.integrate_signal(rxtd_base, n_samples=n_samples_rx)

        if 'sync_time' in self.rx_chain:
            rxtd_base_s = []
            for frm_id in range(n_rd_rep):
                if 'sync_time_frac' in self.rx_chain:
                    sync_frac = True
                else:
                    sync_frac = False
                rxtd_base_s_ = self.sync_time(rxtd_base[frm_id], txtd_base, sc_range=self.sc_range, rx_same_delay=self.rx_same_delay, sync_frac=sync_frac)
                rxtd_base_s.append(rxtd_base_s_)
            rxtd_base_s = np.array(rxtd_base_s)
        else:
            rxtd_base_s = rxtd_base.copy()
            rxtd_base_s = np.stack((rxtd_base_s, rxtd_base_s), axis=2)
        
        if 'sync_freq' in self.rx_chain:
            cfo_coarse = self.estimate_cfo(txtd_base, rxtd_base_s, mode='coarse', sc_range=self.sc_range)
            rxtd_base_t = self.sync_frequency(rxtd_base_s, cfo_coarse, mode='time')
            cfo_fine = self.estimate_cfo(txtd_base, rxtd_base_t, mode='fine', sc_range=self.sc_range)
            cfo = cfo_coarse + cfo_fine
            rxtd_base_s = self.sync_frequency(rxtd_base_s, cfo, mode='time')

        if 'pilot_separate' in self.rx_chain:
            rxtd_pilot_s = rxtd_base_s[:,:,:,:n_samples_rx//2]
            rxtd_base_s = rxtd_base_s[:,:,:,n_samples_rx//2:]
        else:
            rxtd_pilot_s = rxtd_base_s.copy()
        

        rxtd_base = np.stack((rxtd_base_s[:,0,0,:self.n_samples_trx], rxtd_base_s[:,1,0,:self.n_samples_trx]), axis=1)
        rxtd_pilot = np.stack((rxtd_pilot_s[:,0,0,:self.n_samples_trx], rxtd_pilot_s[:,1,0,:self.n_samples_trx]), axis=1)
        
        if 'channel_est' in self.rx_chain:
            if 'sys_res_deconv' in self.rx_chain:
                self.process_sys_response()
            else:
                self.sys_response = None
            snr_est = self.db_to_lin(self.snr_est_db, mode='pow')

            if 'sparse_est' in self.rx_chain:
                h = []
                for frm_id in range(n_rd_rep):
                    h_est_full, H_est, H_est_max = self.channel_estimate(txtd_base, rxtd_pilot_s[frm_id], sys_response=self.sys_response, sc_range_ch=self.sc_range_ch, snr_est=snr_est)
                    h.append(h_est_full)
                h = np.array(h)
                h = h.transpose(3,1,2,0)
                g = self.sys_response.copy()
                if g is not None:
                    g = g.transpose(2,0,1)
                ndly = 5000
                sparse_est_params = self.sparse_est(h=h, g=g, sc_range_ch=self.sc_range_ch, npaths=self.nf_npath_max, nframe_avg=1, ndly=ndly, drange=self.sparse_ch_samp_range, cv=True, n_ignore=self.sparse_ch_n_ignore)
            else:
                h_est_full, H_est, H_est_max = self.channel_estimate(txtd_base, rxtd_pilot_s, sys_response=self.sys_response, sc_range_ch=self.sc_range_ch, snr_est=snr_est)
            
            self.rx_phase_list, self.aoa_list = self.estimate_mimo_params(txtd_base, rxtd_pilot, self.fc, h_est_full, H_est_max, self.rx_phase_list, self.aoa_list)
            if len(self.rx_phase_list)>self.nfft_trx//10:
                self.rx_phase_list.pop(0)
            if len(self.aoa_list)>self.nfft_trx//10:
                self.aoa_list.pop(0)
        else:
            h_est_full = np.ones((self.n_rx_ant, self.n_tx_ant, self.n_samples_ch), dtype=complex)
            H_est = np.ones((self.n_rx_ant, self.n_tx_ant), dtype=complex)
            H_est_max = H_est.copy()
        if 'channel_eq' in self.rx_chain and 'channel_est' in self.rx_chain:
            rxtd_base = self.channel_equalize(txtd_base, rxtd_base[plt_frm_id], h_est_full, H_est, sc_range=self.sc_range, sc_range_ch=self.sc_range_ch, null_sc_range=self.null_sc_range, n_rx_ch_eq=self.n_rx_ch_eq)
            

        if len(rxtd_base.shape)==3:
            rxtd_base = rxtd_base[plt_frm_id]

        return (rxtd_base, h_est_full, H_est, H_est_max, sparse_est_params)
    


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
                sig = self.psd(sig, fs=self.fs_rx, nfft=nfft)
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
                sig = sig[self.sc_range[0]+n_samples//2:self.sc_range[1]+n_samples//2+1]
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

    




def wrap_angle_rad(a): return (a + np.pi) % (2*np.pi) - np.pi
def wrap_angle_deg(a): return (a + 180.0) % 360.0 - 180.0

class Aoa1sKF:
    """
    Wrapped-angle Kalman filter with a *persistent prior* across windows.
    State is [angle; angular_rate]. All internal math uses radians; inputs/outputs
    to the public API are in degrees where noted.

    Parameters
    ----------
    dt : float
        Sampling period (in seconds) of the AoA measurements inside *one* fusion
        window (e.g., 0.1 for 100 ms). This also sets the discrete-time model step.
    sigma_meas_deg : float
        Standard deviation of the AoA measurement noise in **degrees**.
        Smaller -> the filter trusts measurements more; larger -> trusts the model more.
    sigma_acc_deg : float, optional (default=0.3)
        Standard deviation (in **deg/s^2**) of the *angular acceleration* driving
        the process noise. Larger -> more responsive to rapid changes (less smooth);
        smaller -> smoother output (more model-trusting).
    init_angle_deg : float or None, optional
        If provided, the filter is initialized with this AoA (in **degrees**).
        If None, the first observed sample in `step()` will be used to initialize.

    Notes
    -----
    - Angles are wrapped to (-pi, pi] internally to avoid discontinuities.
    - The constant-velocity (angle-rate) model is used:
        x_k = [ angle_k, angular_rate_k ]^T
        x_{k+1} = F x_k + w_k,  z_k = H x_k + v_k
      with F = [[1, dt], [0, 1]], H = [[1, 0]].
    """

    def __init__(self, dt, sigma_meas_deg, sigma_acc_deg=0.3, init_angle_deg=None):
        self.dt = float(dt)
        self.sigma_meas = float(np.deg2rad(sigma_meas_deg))
        self.sigma_acc = float(np.deg2rad(sigma_acc_deg))
        self.F = np.array([[1.0, self.dt],[0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = self.sigma_acc**2 * np.array([[self.dt**3/3.0, self.dt**2/2.0],
                                               [self.dt**2/2.0, self.dt]])
        self.R = self.sigma_meas**2
        self.P = np.diag([np.deg2rad(30.0)**2, np.deg2rad(2.0)**2])
        self.initialized = False
        self.x = None
        if init_angle_deg is not None:
            self.reset(init_angle_deg)
    def reset(self, init_angle_deg):
        self.x = np.array([np.deg2rad(init_angle_deg), 0.0])
        self.x[0] = wrap_angle_rad(self.x[0])
        self.P = np.diag([np.deg2rad(30.0)**2, np.deg2rad(2.0)**2])
        self.initialized = True
    def step(self, angles_deg_1s):
        z_list = np.deg2rad(np.asarray(angles_deg_1s, dtype=float))
        if not self.initialized:
            self.reset(np.rad2deg(z_list[0]))
        for z in z_list:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            innov = wrap_angle_rad(z - (self.H @ self.x)[0])
            S = (self.H @ self.P @ self.H.T)[0, 0] + self.R
            K = (self.P @ self.H.T)[:, 0] / S
            self.x = self.x + K * innov
            self.P = (np.eye(2) - K[:, None] @ self.H) @ self.P
            self.x[0] = wrap_angle_rad(self.x[0])
        return float(np.rad2deg(self.x[0]))
    



class Animate_Plot(Signal_Utils_Rfsoc):
    def __init__(self, params, txtd_base):
        super().__init__(params)

        self.animate_plot_mode = getattr(params, 'animate_plot_mode', [])
        self.plot_fonts_dict = getattr(params, 'plot_fonts_dict', None)
        self.txtd_base = txtd_base

        self.mag_filter_list = {"process_list": ['fft'], "signal_name": ['h', 'H']}
        self.untoched_plot_list = {"process_list": ['IQ'], "signal_name": ['aoa_gauge', 'nf_loc']}

        self.anim_paused = False
        self.read_id = -1
        self.n_plots_row = len(self.animate_plot_mode)
        self.n_plots_col = len(self.freq_hop_list)

        self.plt_n_samples_rx = self.n_samples_trx
        self.n_samp_ch_sp = self.n_samples_ch // 2

        self.kf = Aoa1sKF(dt=0.1, sigma_meas_deg=np.sqrt(5.0), sigma_acc_deg=0.3)
        if len(self.turtlebot_publish_list)>0:
            from tb4_aoa_viz.aoa_bridge import get_publish_fn
        if "aoa" in self.turtlebot_publish_list:
            self.publish_aoa_turtlebot = get_publish_fn("/aoa_angle")
        else:
            self.publish_aoa_turtlebot = lambda x: None



    def process_signals_for_plot(self, txtd_base, rxtd_base, h_est_full, H_est_full, sparse_est_params):

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

        supported_operations = ['+', '-', '*', '/']
        signals=[]
        for plot in self.animate_plot_mode:
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
                    x = self.t_tx[:self.n_samples_tx]
                    sig = txtd_base[tx_id]
                    title += "TX"
                    if 'fft' in signal_process_list:
                        x = self.freq_tx
                        xlabel_mode = 'freq'
                        title += "-FD"
                    else:
                        x = self.t_tx*1e9
                        xlabel_mode = 'time'
                        title += "-TD"
                elif signal_name == 'rxtd':
                    sig = rxtd_base[rx_id]
                    title += "RX"
                    if 'fft' in signal_process_list:
                        x = self.freq_trx
                        xlabel_mode = 'freq'
                        title += "-FD"
                    else:
                        x = self.t_rx[:self.plt_n_samples_rx]*1e9
                        xlabel_mode = 'time'
                        title += "-TD"
                elif signal_name == 'h':
                    x = self.t_trx[:self.n_samples_ch]*1e9
                    sig = h_est_full[rx_id, tx_id]
                    title += "Channel"
                    if 'fft' in signal_process_list:
                        xlabel_mode = 'freq'
                        title += "-FD"
                    else:
                        xlabel_mode = 'time'
                        title += "-TD"
                elif signal_name == 'H':
                    x = self.freq_trx[(self.sc_range_ch[0]+self.n_samples_trx//2):(self.sc_range_ch[1]+self.n_samples_trx//2+1)]
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
                    sig = self.rx_phase_list
                    title += "RX-Phase Diff-TD"
                    xlabel_mode = 'id'
                    ylabel_mode = 'phase'
                elif signal_name == 'aoa_gauge':
                    # sig = self.aoa_list[-1]
                    window_deg = np.rad2deg(self.aoa_list[-10:])
                    sig = wrap_angle_deg(self.kf.step(window_deg))
                    sig = np.deg2rad(sig)
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
                
                
                sig, title_post = self.process_sig(sig, process_list=signal_process_list)
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
                    # if "phase" in signal_process_list:
                    #     sig_final = np.unwrap(sig_final)
                    plot_signals.append({'signal_name': signal_name, 'trx_id':[rx_id, tx_id], 'process_list': signal_process_list, 'x': x, 'data': sig_final, 'label': label_final})
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


            signals.append({'plot_signals': plot_signals, 'title': title, 'x_label': xlabel, 'y_label': ylabel})

        return signals
    


    def receive_data_anim(self, txtd_base):

        self.read_id+=1
        sigs_save = None
        channels_save = None
        if 'channel' in self.saved_sig_plot:
            channels_save = np.load(self.channel_save_path)
        if 'signal' in self.saved_sig_plot:
            sigs_save = np.load(self.sig_save_path)

        if sigs_save is None:
            if channels_save is None:
                rxtd = self.receive_data(self.client_rfsoc, n_rd_rep=self.n_rd_rep, mode='once')
            else:
                rxtd = None
        else:
            rxtd = sigs_save['rxtd_{:.1f}'.format(self.fc/1e9)][self.read_id*self.n_rd_rep:(self.read_id+1)*self.n_rd_rep]
            txtd_base = sigs_save['txtd'][0]

        if channels_save is None:
            while True:
                (rxtd_base, h_est_full, H_est, H_est_max, sparse_est_params) = self.rx_operations(txtd_base, rxtd)

                if self.enable_matlab_stream:
                    j = self.fc_id
                    print(self.freq_hop_list[j])
                    print("Streaming RX TD data to MATLAB...")
                    # (txtd_base,txtd) = self.gen_tx_signal()
                    # self.stream_rx_td_to_matlab(txtd)    
                    # self.stream_rx_td_to_matlab(self.freq_hop_list[j])
                    self.stream_rx_td_to_matlab(rxtd,self.freq_hop_list[j])
                    print("Streaming RX TD data to MATLAB done.")

                H_est_full = fft(h_est_full, axis=-1)
                if sparse_est_params is not None:
                    (h_tr, dly_est, peaks, npath_est) = sparse_est_params
                    if np.min(npath_est) > 0:
                        break
                    else:
                        self.print("Re-estimating channel due to zero paths", thr=0)
                        rxtd = self.receive_data(self.client_rfsoc, n_rd_rep=self.n_rd_rep, mode='once')
                else:
                    break
        else:
            h_est_full = channels_save['h_est_full_{:.1f}'.format(self.fc/1e9)]

            h = h_est_full.copy()[self.read_id*self.n_rd_rep:(self.read_id+1)*self.n_rd_rep]
            h = h.transpose(3,1,2,0)
            g = None
            ndly = 5000
            sparse_est_params = self.sparse_est(h=h, g=g, sc_range_ch=self.sc_range_ch, npaths=1, nframe_avg=1, ndly=ndly, drange=self.sparse_ch_samp_range, cv=True, n_ignore=self.sparse_ch_n_ignore)

            h_est_full = h_est_full[self.read_id]
            H_est_full = fft(h_est_full, axis=-1)


        signals = self.process_signals_for_plot(txtd_base, rxtd_base, h_est_full, H_est_full, sparse_est_params)
        
        return signals, h_est_full, sparse_est_params



    def toggle_pause(self, event):
        if event.key == 'p':  # Press 'p' to pause/resume
            self.anim_paused = not self.anim_paused



    def update(self, frame):

        if self.anim_paused:
            return self.line
        
        signals, h_est_full, sparse_est_params = self.receive_data_anim(self.txtd_base)

        self.hop_freq(self.client_piradio, self.client_controller)

        # if self.use_turntable:
        #     angle = self.rotation_angles[self.rot_angle_id]
        #     client_turntable.move_to_position(angle)
        #     self.rot_angle_id = (self.rot_angle_id + 1) % len(self.rotation_angles)

        self.handle_nf(h_est_full, sparse_est_params)


        line_id = 0
        for i in range(self.n_plots_row):
            j = self.fc_id - 1

            for signal in signals[i]['plot_signals']:

                signal_name = signal['signal_name']
                rx_id = signal['trx_id'][0]
                tx_id = signal['trx_id'][1]
                signal_data = signal['data']
                signal_process_list = signal['process_list']
                
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
                    self.publish_aoa_turtlebot(signal_data)
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
                    dly = np.arange(self.n_samples_ch)
                    dly = dly - self.n_samples_ch*(dly > self.n_samples_ch/2)
                    dly = dly / self.fs_trx *1e9
                    chan_pow = self.lin_to_db(np.abs(h_tr), mode='mag')

                    # Roll the response and shift the response
                    rots = self.n_samp_ch_sp//4
                    yshift = np.percentile(chan_pow, 25)
                    chan_powr = np.roll(chan_pow, rots) - yshift
                    dlyr = np.roll(dly, rots)
                    self.line[line_id][j].set_data(dlyr[:self.n_samp_ch_sp], chan_powr[:self.n_samp_ch_sp])
                    line_id+=1

                    # Compute the axes
                    ymax = np.max(chan_powr)+5
                    ymin = -10

                    # Plot the locations of the detected peaks
                    peaks_ = np.abs(peaks)**2
                    peaks_  = self.lin_to_db(peaks_, mode='pow')-yshift
                    dly_est = dly_est*1e9
                    dly_est = dly_est[dly_est<=np.max(dlyr[:self.n_samp_ch_sp])]
                    self.line[line_id][j].set_data(dly_est, peaks_)
                    line_id+=1
                    # for dly, peak in zip(dly_est, peaks_):
                    #     # self.line[line_id][j].set_ydata([dly, peak])
                    #     self.line[line_id][j].set_segments([[[dly, ymin], [dly, peak]]])
                    self.line[line_id][j].set_segments([[[i,ymin], [i,j]] for i,j in zip(dly_est, peaks_)])
                    line_id+=1
                    self.ax[i][j].set_ylim([ymin, ymax])
                elif signal_name == 'nf_loc':
                    self.nf_model.plot_results(self.ax[i][j], RoomModel=self.RoomModel, plot_type='init_est')
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
            elif not (signal_name in self.untoched_plot_list['signal_name'] or any(item in signal_process_list for item in self.untoched_plot_list['process_list'])):
                try:
                    self.ax[i][j].relim()
                    self.ax[i][j].autoscale_view()
                except Exception as e:
                    print("Error in autoscale {}".format(e))


        return self.line




    def init_plots(self):
        if self.plot_level<0:
            return
        
        signals, _, _ = self.receive_data_anim(self.txtd_base)
        
        # Set up the figure and plot
        self.line = [[None for j in range(self.n_plots_col)] for i in range(3*self.n_plots_row)]
        self.fig, self.ax = plt.subplots(self.n_plots_row, self.n_plots_col)
        if type(self.ax) is not np.ndarray:
            self.ax = np.array([self.ax])
        if len(self.ax.shape)<2:
            self.ax = self.ax.reshape(-1, 1)
        self.fig.canvas.mpl_connect('key_press_event', self.toggle_pause)


        for j in range(self.n_plots_col):
            line_id = 0
            for i in range(self.n_plots_row):
                for signal in signals[i]['plot_signals']:

                    signal_name = signal['signal_name']
                    label = signal['label']
                    signal_process_list = signal['process_list']
                    signal_data = signal['data']
                    x_data = signal['x']
                
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
                        self.draw_half_gauge(self.ax[i][j], min_val=-90, max_val=90)
                        self.gauge_update_needle(self.ax[i][j], 0, min_val=-90, max_val=90)
                        self.ax[i][j].set_xlim(0, 1)
                        self.ax[i][j].set_ylim(0.5, 1)
                        self.ax[i][j].axis('off')
                        
                    elif signal_name=='nf_loc':
                        self.ax[i][j] = self.nf_model.plot_results(self.ax[i][j], RoomModel=self.RoomModel, plot_type='init_est')
                        self.ax[i][j].set_yticks([])

                        self.ax[i][j].set_xlim(self.nf_region[0])
                        self.ax[i][j].set_ylim(self.nf_region[1])
                        self.ax[i][j].set_xticks(np.arange(self.nself.n_plots_rowf_region[0,0], self.nf_region[0,1], 1.0))
                        self.ax[i][j].set_yticks(np.arange(self.nf_region[1,0], self.nf_region[1,1], 2.0))

                    else:
                        self.line[line_id][j], = self.ax[i][j].plot(x_data, signal_data, label=label)
                        line_id+=1


                # Truncate the title to a maximum of 30 characters
                title = (signals[i]['title'][:self.plot_fonts_dict['title_max_chars']] + '...') if len(signals[i]['title']) > self.plot_fonts_dict['title_max_chars'] else signals[i]['title']
                title = title + "\n Carrier Frequency: {} GHz".format(self.freq_hop_list[j]/1e9)
                x_label = signals[i]['x_label']
                y_label = signals[i]['y_label']
                self.ax[i][j].set_title(title)
                self.ax[i][j].set_xlabel(x_label)
                self.ax[i][j].set_ylabel(y_label)


                # self.ax[i][j].title.set_fontsize(35-5*self.n_plots_row-3*self.n_plots_col)
                # self.ax[i][j].xaxis.label.set_fontsize(30-4*self.n_plots_row-2*self.n_plots_col)
                # self.ax[i][j].yaxis.label.set_fontsize(30-4*self.n_plots_row-2*self.n_plots_col)
                # self.ax[i][j].tick_params(axis='both', which='major', labelsize=25-4*self.n_plots_row-2*self.n_plots_col)  # For major ticks
                # self.ax[i][j].legend(fontsize=30-4*self.n_plots_row-2.5*self.n_plots_col)
                self.ax[i][j].title.set_fontsize(self.plot_fonts_dict['title_size'])
                self.ax[i][j].xaxis.label.set_fontsize(self.plot_fonts_dict['xaxis_size'])
                self.ax[i][j].yaxis.label.set_fontsize(self.plot_fonts_dict['yaxis_size'])
                self.ax[i][j].tick_params(axis='both', which='major', labelsize=self.plot_fonts_dict['ticks_size'])  # For major ticks
                self.ax[i][j].legend(fontsize=self.plot_fonts_dict['legend_size'])

                self.ax[i][j].grid(True)
                if not (signal_name in self.untoched_plot_list['signal_name'] or any(item in signal_process_list for item in self.untoched_plot_list['process_list'])):
                    self.ax[i][j].relim()
                    self.ax[i][j].autoscale_view()
                self.ax[i][j].minorticks_on()

        for j in range(self.n_plots_col):
            for i in range(len(self.line)):
                if self.line[i][j] is not None:
                    # self.line[i][j].set_linewidth(3.0-0.5*self.n_plots_row-0.3*self.n_plots_col)
                    self.line[i][j].set_linewidth(self.plot_fonts_dict['line_width'])

        # Create the animation
        plt.tight_layout()
        plt.subplots_adjust(hspace=self.plot_fonts_dict['hspace'], wspace=self.plot_fonts_dict['wspace'])
        anim = animation.FuncAnimation(self.fig, self.update, frames=int(1e9), interval=self.anim_interval, blit=False)
        plt.show()
        self.fig.savefig(self.figs_save_path, dpi=300)



    def stream_rx_td_to_matlab(self,rxtd,freq):
        import io
        import socket
        import scipy.io

        try:
            #s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #s.connect((self.matlab_stream_ip, self.matlab_stream_port))
            #print("Connected to MATLAB stream at {}:{}".format(self.matlab_stream_ip, self.matlab_stream_port))
            #s.close()
            buf = io.BytesIO()
            scipy.io.savemat(buf, {'rxtd': rxtd},do_compression=True)
            scipy.io.savemat(buf, {'freq': freq},do_compression=True)
            data = buf.getvalue()
            print(len(data), "bytes of rxtd data to be sent to MATLAB stream")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(.1)
                sock.connect((self.matlab_stream_ip, self.matlab_stream_port))
                sock.sendall(len(data).to_bytes(8, byteorder='big'))  # Send the length of the data first
                sock.sendall(data)
                self.print("rxtd data sent to MATLAB stream at {}:{}".format(self.matlab_stream_ip, self.matlab_stream_port), thr=1)
        
        except Exception as e:
            self.print("Error in streaming rxtd to MATLAB: {}".format(e), thr=0)



if __name__ == "__main__":
    from params import Params_Class
    params = Params_Class()

    signals_inst = Signal_Utils_Rfsoc(params)
    # signals_inst.collect_signals()
    signals_inst.compute_sys_response()




