import time
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.fft import fft


from signal_utilsrfsoc import Signal_Utils_Rfsoc, SignalUtilsRFSoCConfig
from sigcom_toolkit.signal_utils import Signal_Utils, AoAKalmanFilter, SignalUtilsConfig

try:
    from sigcom_toolkit.near_field import Sim as Near_Field_Model, RoomModel
except:
    pass



# TODO Clean this file and check dependencies

@dataclass
class NearFieldSignalUtilsConfig(SignalUtilsRFSoCConfig):
    nf_param_estimate = False                  # If True, performs near field parameter estimation
    nf_walls = np.array([[-5,4], [-1,6]])      # Near field walls coordinates in meters
    nf_rx_sep_dir = np.array([1,0])            # Direction of the RX antenna separation
    nf_tx_sep_dir = np.array([1,0])            # Direction of the TX antenna separation
    nf_npath_max = [20,5]                      # 1st number is the maximum number to extract at the 1st round, 2nd number is the maximum number to extract at the 2nd round
    nf_stop_thr = 0.03                         # Stopping threshold for the near field parameter estimation
    nf_tx_loc = np.array([[0.3,1.0]])          # TX antenna location in meters
    nf_rx_loc_sep = np.array([0,0.2,0.4])      # RX locations separation in meters
    nf_tx_ant_sep = 0.5                        # TX antenna separation in meters
    nf_rx_ant_sep = 0.5 * np.array([1,2,4])    # RX antenna separation in meters

    def __post_init__(self):
        super().__post_init__()

        self.nf_n_rx_loc_sep = len(self.nf_rx_loc_sep)
        self.nf_n_ant_sep = len(self.nf_rx_ant_sep)
        self.nf_n_meas = self.nf_n_rx_loc_sep * self.nf_n_ant_sep
        p = len(self.nf_rx_sep_dir)
        # Generate the RX antenna positions
        self.nf_rx_ant_loc = np.zeros((self.n_rx_ant, self.nf_n_meas, p))
        self.nf_tx_ant_loc = np.zeros((self.n_tx_ant, self.nf_n_meas, p))
        for k in range(self.nf_n_rx_loc_sep):
            for i in range(self.nf_n_ant_sep):
                m = k*self.nf_n_ant_sep + i
                # Linear distance of the RX antennas from the origin
                t = self.nf_rx_loc_sep[k] + self.nf_rx_ant_sep[i]*np.arange(self.n_rx_ant)*self.wl
                # Position of the RX antennas
                self.nf_rx_ant_loc[:,m,:] = t[:,None]*self.nf_rx_sep_dir[None,:]

                t = self.ant_d_m[0] * np.arange(self.n_tx_ant)
                self.nf_tx_ant_loc[:,m,:] = self.nf_tx_loc + t[:,None]*self.nf_tx_sep_dir[None,:]


class NearFieldSignalUtils(Signal_Utils_Rfsoc):
    def __init__(self, config: NearFieldSignalUtilsConfig, **overrides: Any):        
        # strict override: only allow existing fields
        super().__init__(config, **overrides)

        self.nf_loc_idx = 0
        self.nf_sep_idx = 0

        self.create_near_field_model()


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
                        rxlocsep=self.nf_rx_loc_sep, sepdir=self.nf_rx_sep_dir, antsep=self.nf_rx_ant_sep, npath_est=self.npath_max[1], 
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
    


    def handle_nf(self, h_est_full, sparse_est_params, client_lintrack):
        use_linear_track = True

        if self.nf_param_estimate:
            # h_index = self.animate_plot_mode.index('h')
            if self.nf_loc_idx==0:
                self.nf_sep_idx = 0

                if use_linear_track:
                    client_lintrack.return2home(lin_track_id=0)
                    client_lintrack.return2home(lin_track_id=1)
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
                    if use_linear_track:
                        distance = 1000*(self.nf_rx_ant_sep[0]*self.wl - self.nf_rx_ant_sep[-1]*self.wl)
                        distance = np.round(distance, 2)
                        client_lintrack.move(lin_track_id=1, distance=distance)
                        time.sleep(0.5)
                        self.ant_d[0] = self.nf_rx_ant_sep[0]

                        if self.nf_loc_idx < len(self.nf_rx_loc):
                            distance = 1000*(self.nf_rx_loc[self.nf_loc_idx,0] - self.nf_rx_loc[self.nf_loc_idx-1,0])
                            distance = np.round(distance, 2)
                            client_lintrack.move(lin_track_id=1, distance=distance)
                            client_lintrack.move(lin_track_id=0, distance=distance)
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

                    if use_linear_track:
                        if self.nf_sep_idx < len(self.nf_rx_ant_sep):
                            distance = 1000*(self.nf_rx_ant_sep[self.nf_sep_idx]*self.wl - self.nf_rx_ant_sep[self.nf_sep_idx-1]*self.wl)
                            distance = np.round(distance, 2)
                            client_lintrack.move(lin_track_id=1, distance=distance)
                            time.sleep(0.5)
                            self.ant_d[0] = self.nf_rx_ant_sep[self.nf_sep_idx]
                    
                    self.nf_sep_idx+=1
            
                self.ant_d_m[0] = self.ant_d[0] * self.wl

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
            ant_d_m = [np.linalg.norm(self.nf_rx_ant_loc[1,m,:] - self.nf_rx_ant_loc[0,m,:], axis=-1)]
            aoa[:,m] = self.phase_to_aoa(phase_diff[:,m], wl=self.wl, ant_d_m=ant_d_m)
            # aoa = self.phase_to_aoa(phase_diff, wl=self.wl, ant_d_m=self.ant_d_m)
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


