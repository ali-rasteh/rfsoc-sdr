from backend import *
from backend import be_np as np, be_scp as scipy
from sigcom_toolkit.general import General, GeneralConfig
from dataclasses import dataclass, fields, replace



@dataclass
class TCPComConfig(GeneralConfig):
    server_ip : str = '0.0.0.0'
    TCP_port_Cmd : int = 8080
    TCP_port_Data : int = 8081
    tcp_localIP : str = '0.0.0.0'
    tcp_bufferSize : int = 2**10
    after_idle_sec : int = 1
    interval_sec : int = 3
    max_fails : int = 5
    nbytes : int = 2
    
    invalidCommandMessage : str = "ERROR: Invalid command"
    invalidNumberOfArgumentsMessage : str = "ERROR: Invalid number of arguments"
    successMessage : str = "Successully executed"
    droppedMessage : str = "Connection dropped?"


class Tcp_Comm(General):
    def __init__(self, config: TCPComConfig, **overrides):
        super().__init__(config, **overrides)

        self.print("Tcp_Comm object init done", thr=5)

    def close(self):
        self.radio_control.close()
        self.radio_data.close()
        self.print("Client object closed", thr=1)

    def __del__(self):
        self.close()
        self.print("Client object deleted", thr=1)

    def init_tcp_server(self):
        ## TCP Server
        self.print("Starting TCP server", thr=1)
        
        ## Command
        self.TCPServerSocketCmd = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)# Create a datagram socket
        self.TCPServerSocketCmd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.TCPServerSocketCmd.bind((self.config.tcp_localIP, self.config.TCP_port_Cmd)) # Bind to address and ip
        
        ## Data
        self.TCPServerSocketData = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)         # Create a datagram socket
        self.TCPServerSocketData.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.TCPServerSocketData.bind((self.config.tcp_localIP, self.config.TCP_port_Data))                # Bind to address and ip

        bufsize = self.TCPServerSocketData.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF) 
        # self.print ("Buffer size [Before]:%d" %bufsize, thr=2)
        self.print("TCP server is up", thr=1)
    
    def run_tcp_server(self, call_back_func):
        # Listen for incoming connections
        self.TCPServerSocketCmd.listen(1)
        self.TCPServerSocketData.listen(1)
        
        while True:
            # Wait for a connection
            self.print('\nWaiting for a connection', thr=2)
            self.connectionCMD, addrCMD = self.TCPServerSocketCmd.accept()
            self.connectionData, addrDATA = self.TCPServerSocketData.accept()
            self.print('\nConnection established', thr=2)
            
            
            self.connectionData.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.connectionData.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.config.after_idle_sec)
            self.connectionData.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.config.interval_sec)
            self.connectionData.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.config.max_fails)
            
            self.connectionCMD.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.connectionCMD.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.config.after_idle_sec)
            self.connectionCMD.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.config.interval_sec)
            self.connectionCMD.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.config.max_fails)            
            
            try:
                while True:
                    try:
                        receivedCMD = self.connectionCMD.recv(self.config.tcp_bufferSize)
                        if receivedCMD:
                            self.print("\nClient CMD:{}".format(receivedCMD.decode()), thr=5)
                            responseToCMDinBytes = call_back_func(receivedCMD)
                            self.connectionCMD.sendall(responseToCMDinBytes)
                        else:
                            break
                    except:
                        break
            finally:
                # Clean up the connection
                self.print('\nConnection is closed.', thr=2)
                self.connectionCMD.close()                  
                self.connectionData.close()

    def init_tcp_client(self):
        self.radio_control = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.radio_control.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.radio_control.connect((self.config.server_ip, self.config.TCP_port_Cmd))

        self.radio_data = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.radio_data.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.radio_data.connect((self.config.server_ip, self.config.TCP_port_Data))

        self.print("Client succesfully connected to the server", thr=1)



@dataclass
class TCPComRFSoCConfig(TCPComConfig):
    fc : float = 6.0e9
    beam_test : list = None
    adc_bits : int = 14
    dac_bits : int = 14
    RFFE : str = None
    n_frame_rd : int = 2
    n_samples : int = 1024
    n_tx_ant : int = 2
    n_rx_ant : int = 2


class Tcp_Comm_RFSoC(Tcp_Comm):
    def __init__(self, config: TCPComRFSoCConfig, **overrides):
        super().__init__(config, **overrides)

        self.obj_rfsoc = None

        if self.config.RFFE=='sivers':
            self.tx_bb_gain = 0x3
            self.tx_bb_phase = 0x0
            self.tx_bb_iq_gain = 0x77
            self.tx_bfrf_gain = 0x7F
            self.rx_gain_ctrl_bb1 = 0x33
            self.rx_gain_ctrl_bb2 = 0x00
            self.rx_gain_ctrl_bb3 = 0x33
            self.rx_gain_ctrl_bfrf = 0x7F

        self.nread = self.config.n_rx_ant * self.config.n_frame_rd * self.config.n_samples

        self.print("Tcp_Comm_RFSoC object init done", thr=1)

    def set_mode_sivers(self, mode):
        if mode == 'RXen0_TXen1' or mode == 'RXen1_TXen0' or mode == 'RXen0_TXen0':
            self.radio_control.sendall(b"setModeSiver "+str.encode(str(mode)))
            data = self.radio_control.recv(1024)
            self.print("Result of set_mode_sivers: {}".format(data),thr=3)
            return data
        
    def set_frequency_sivers(self, fc):
        self.radio_control.sendall(b"setCarrierFrequencySivers "+str.encode(str(fc)))
        data = self.radio_control.recv(1024)
        self.print("Result of set_frequency_sivers: {}".format(data),thr=3)
        return data
    
    def set_frequency_mixer_rfsoc(self, f_mixer_dac, f_mixer_adc):
        self.radio_control.sendall(b"setFrequencyMixer "+str.encode(str(f_mixer_dac) + " ") + str.encode(str(f_mixer_adc)))
        data = self.radio_control.recv(1024)
        self.print("Result of set_frequency_mixer_rfsoc: {}".format(data),thr=3)
        return data

    def set_tx_gain_sivers(self):
        self.radio_control.sendall(b"setGainTXSivers " + str.encode(str(int(self.tx_bb_gain)) + " ") \
                                                    + str.encode(str(int(self.tx_bb_phase)) + " ") \
                                                    + str.encode(str(int(self.tx_bb_iq_gain)) + " ") \
                                                    + str.encode(str(int(self.tx_bfrf_gain))))
        data = self.radio_control.recv(1024)
        self.print("Result of set_tx_gain_sivers: {}".format(data),thr=3)
        return data

    def set_rx_gain_sivers(self):
        self.radio_control.sendall(b"setGainRXSivers " + str.encode(str(int(self.rx_gain_ctrl_bb1)) + " ") \
                                                    + str.encode(str(int(self.rx_gain_ctrl_bb2)) + " ") \
                                                    + str.encode(str(int(self.rx_gain_ctrl_bb3)) + " ") \
                                                    + str.encode(str(int(self.rx_gain_ctrl_bfrf))))
        data = self.radio_control.recv(1024)
        self.print("Result of set_rx_gain_sivers: {}".format(data),thr=3)
        return data

    def transmit_data_default_rfsoc(self):
        self.radio_control.sendall(b"transmitSamplesDefault")
        data = self.radio_control.recv(1024)
        self.print("Result of transmit_data_default_rfsoc: {}".format(data),thr=3)
        return data
    
    def transmit_data_rfsoc(self, txtd):
        txtd = txtd.copy()
        txtd = np.array(txtd).flatten()
        txtd = txtd * (2 ** (self.config.dac_bits + 1) - 1)
        re = txtd.real.astype(np.int16)
        im = txtd.imag.astype(np.int16)
        txtd = np.concatenate((re, im))

        self.radio_control.sendall(b"transmitSamples")
        self.radio_data.sendall(txtd.tobytes())
        data = self.radio_control.recv(1024)
        self.print("Result of transmit_data_rfsoc: {}".format(data),thr=3)
        return data

    def receive_data_rfsoc(self, mode='once'):
        if mode=='once':
            nbeams = 1
            self.radio_control.sendall(b"receiveSamplesOnce")
        elif mode=='beams':
            nbeams = len(self.config.beam_test)
            self.radio_control.sendall(b"receiveSamples")
        nbytes = nbeams * self.config.nbytes * self.nread * 2
        buf = bytearray()

        while len(buf) < nbytes:
            data = self.radio_data.recv(nbytes)
            buf.extend(data)
        data = np.frombuffer(buf, dtype=np.int16)
        data = data/(2 ** (self.config.adc_bits + 1) - 1)
        rxtd = data[:self.nread*nbeams] + 1j*data[self.nread*nbeams:]
        rxtd = rxtd.reshape(nbeams, self.config.n_rx_ant, self.nread//self.config.n_rx_ant)
        return rxtd
    

    def parse_and_execute(self, receivedCMD):
        clientMsg = receivedCMD.decode()
        clientMsgParsed = clientMsg.split()

        if clientMsgParsed[0] == "receiveSamplesOnce":
            if len(clientMsgParsed) == 1:
                iq_data = self.obj_rfsoc.recv_frame_one(n_frame=self.obj_rfsoc.n_frame_rd)
                iq_data = np.array(iq_data).flatten()
                iq_data = iq_data * (2 ** (self.obj_rfsoc.adc_bits + 1) - 1)
                re = iq_data.real.astype(np.int16)
                im = iq_data.imag.astype(np.int16)
                iq_data = np.concatenate((re, im))
                self.connectionData.sendall(iq_data.tobytes())
                responseToCMD = "Success"
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "receiveSamples":
            if len(clientMsgParsed) == 1:
                iq_data = self.obj_rfsoc.recv_frame(n_frame=self.obj_rfsoc.n_frame_rd)
                re = iq_data.real.astype(np.int16)
                im = iq_data.imag.astype(np.int16)
                iq_data = np.concatenate((re, im))
                self.connectionData.sendall(iq_data.tobytes())
                responseToCMD = "Success"
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "transmitSamplesDefault":
            if len(clientMsgParsed) == 1:
                self.obj_rfsoc.send_frame(txtd=self.obj_rfsoc.txtd)
                responseToCMD = 'Success'
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "transmitSamples":
            if len(clientMsgParsed) == 1:
                nread = self.obj_rfsoc.n_tx_ant * self.obj_rfsoc.n_samples_tx
                nbytes = self.config.nbytes * nread * 2
                buf = bytearray()

                while len(buf) < nbytes:
                    data = self.connectionData.recv(nbytes)
                    buf.extend(data)
                data = np.frombuffer(buf, dtype=np.int16)
                data = data/(2 ** (self.obj_rfsoc.dac_bits + 1) - 1)
                txtd = data[:nread] + 1j*data[nread:]
                txtd = txtd.reshape(self.obj_rfsoc.n_tx_ant, nread//self.obj_rfsoc.n_tx_ant)

                self.obj_rfsoc.send_frame(txtd=txtd)
                responseToCMD = 'Success'
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "getBeamIndexTXSivers":
            if len(clientMsgParsed) == 1:
                responseToCMD = str(self.obj_rfsoc.siversControllerObj.getBeamIndexTX())
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage 
        elif clientMsgParsed[0] == "setBeamIndexTXSivers":
            if len(clientMsgParsed) == 2:
                beamIndex = int(clientMsgParsed[1])
                success, status = self.obj_rfsoc.siversControllerObj.setBeamIndexTX(beamIndex)
                if success == True:
                    responseToCMD = self.config.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage  
        elif clientMsgParsed[0] == "getBeamIndexRXSivers":
            if len(clientMsgParsed) == 1:
                responseToCMD = str(self.obj_rfsoc.siversControllerObj.getBeamIndexRX())
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage 
        elif clientMsgParsed[0] == "setBeamIndexRXSivers":
            if len(clientMsgParsed) == 2:
                beamIndex = int(clientMsgParsed[1])
                success, status = self.obj_rfsoc.siversControllerObj.setBeamIndexRX(beamIndex)
                if success == True:
                    responseToCMD = self.config.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "getModeSiver":
            if len(clientMsgParsed) == 1:
                responseToCMD = self.obj_rfsoc.siversControllerObj.getMode()
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage 
        elif clientMsgParsed[0] == "setModeSiver":
            if len(clientMsgParsed) == 2:
                mode = clientMsgParsed[1]
                success,status = self.obj_rfsoc.siversControllerObj.setMode(mode)
                if success == True:
                    responseToCMD = self.config.successMessage 
                else:
                    responseToCMD = status                  
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage    
        elif clientMsgParsed[0] == "getGainRXSivers":
            if len(clientMsgParsed) == 1:
                rx_gain_ctrl_bb1, rx_gain_ctrl_bb2, rx_gain_ctrl_bb3, rx_gain_ctrl_bfrf,agc_int_bfrf_gain_lvl, agc_int_bb3_gain_lvl = self.obj_rfsoc.siversControllerObj.getGainRX()
                responseToCMD = 'rx_gain_ctrl_bb1:' + str(hex(rx_gain_ctrl_bb1)) + \
                                ', rx_gain_ctrl_bb2:' +  str(hex(rx_gain_ctrl_bb2)) + \
                                ', rx_gain_ctrl_bb3:' +   str(hex(rx_gain_ctrl_bb3)) + \
                                ', rx_gain_ctrl_bfrf:' +   str(hex(rx_gain_ctrl_bfrf)) +\
                                ', agc_int_bfrf_gain_lvl:' +   str(hex(agc_int_bfrf_gain_lvl)) +\
                                ', agc_int_bb3_gain_lvl:' +   str(hex(agc_int_bb3_gain_lvl))
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage 
        elif clientMsgParsed[0] == "setGainRXSivers":
            if len(clientMsgParsed) == 5:
                rx_gain_ctrl_bb1 = int(clientMsgParsed[1])
                rx_gain_ctrl_bb2 = int(clientMsgParsed[2])
                rx_gain_ctrl_bb3 = int(clientMsgParsed[3])
                rx_gain_ctrl_bfrf = int(clientMsgParsed[4])
                
                success,status = self.obj_rfsoc.siversControllerObj.setGainRX(rx_gain_ctrl_bb1, rx_gain_ctrl_bb2, rx_gain_ctrl_bb3, rx_gain_ctrl_bfrf)
                if success == True:
                    responseToCMD = self.config.successMessage 
                else:
                    responseToCMD = status                  
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage      
        elif clientMsgParsed[0] == "getGainTXSivers":
            if len(clientMsgParsed) == 1:
                tx_bb_gain, tx_bb_phase, tx_bb_iq_gain, tx_bfrf_gain, tx_ctrl = self.obj_rfsoc.siversControllerObj.getGainTX()
                responseToCMD = 'tx_bb_gain:' + str(hex(tx_bb_gain)) + \
                                ', tx_bb_phase:' +  str(hex(tx_bb_phase)) + \
                                ', tx_bb_gain:' +   str(hex(tx_bb_iq_gain)) + \
                                ', tx_bfrf_gain:' +   str(hex(tx_bfrf_gain)) + \
                                ', tx_ctrl:' +   str(hex(tx_ctrl))
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage 
        elif clientMsgParsed[0] == "setGainTXSivers":
            if len(clientMsgParsed) == 5:
                self.print(clientMsgParsed[1], thr=2)
                
                tx_bb_gain = int(clientMsgParsed[1])
                tx_bb_phase = int(clientMsgParsed[2])
                tx_bb_iq_gain = int(clientMsgParsed[3])
                tx_bfrf_gain = int(clientMsgParsed[4])
                
                success,status = self.obj_rfsoc.siversControllerObj.setGainTX(tx_bb_gain, tx_bb_phase, tx_bb_iq_gain, tx_bfrf_gain)
                if success == True:
                    responseToCMD = self.config.successMessage 
                else:
                    responseToCMD = status                  
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage   
        elif clientMsgParsed[0] == "getCarrierFrequencySivers":
            if len(clientMsgParsed) == 1:
                responseToCMD = str(self.obj_rfsoc.siversControllerObj.getFrequency())
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage 
        elif clientMsgParsed[0] == "setCarrierFrequencySivers":
            if len(clientMsgParsed) == 2:
                self.print(clientMsgParsed[1], thr=2)
                fc = float(clientMsgParsed[1])
                success, status = self.obj_rfsoc.siversControllerObj.setFrequency(fc)
                if success == True:
                    responseToCMD = self.config.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "setFrequencyMixer":
            if len(clientMsgParsed) == 3:
                self.print(clientMsgParsed[1], thr=2)

                f_mixer_dac = float(clientMsgParsed[1])
                f_mixer_adc = float(clientMsgParsed[2])
                success = self.obj_rfsoc.set_dac_mixer(rfsoc_mix_freq=f_mixer_dac, do_rfsoc_mixer_settings=True)
                success &= self.obj_rfsoc.set_adc_mixer(rfsoc_mix_freq=f_mixer_adc, do_rfsoc_mixer_settings=True)
                if success == True:
                    responseToCMD = self.config.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage
                
        #######################
        else:
            responseToCMD = self.config.invalidCommandMessage
        
        responseToCMDInBytes = str.encode(responseToCMD + " (" + clientMsg + ")" )  
        return responseToCMDInBytes
    


@dataclass
class TCPComLinTrackConfig(TCPComConfig):
    pass

class Tcp_Comm_LinTrack(Tcp_Comm):
    def __init__(self, config: TCPComLinTrackConfig, **overrides):
        super().__init__(config, **overrides)
        self.obj_lintrack = None

        self.print("Tcp_Comm_LinTrack object init done", thr=1)

    def move(self, lin_track_id=0, distance=0.0):
        self.print("Moving linear track {} by {} mm".format(lin_track_id, distance), thr=2)
        self.radio_control.sendall(b"Move "+str.encode(str(lin_track_id)+' ')+str.encode(str(distance)))
        data = self.radio_control.recv(1024)
        self.print("Result of move_forward: {}".format(data), thr=3)
        return data
    
    def return2home(self, lin_track_id=0):
        self.print("Returning linear track {} to home".format(lin_track_id), thr=2)
        self.radio_control.sendall(b"Return2home "+str.encode(str(lin_track_id)))
        data = self.radio_control.recv(1024)
        self.print("Result of Return2home: {}".format(data), thr=3)
        return data
    
    def go2end(self, lin_track_id=0):
        self.print("Going to the end of line on linear track {}".format(lin_track_id), thr=2)
        self.radio_control.sendall(b"Go2end "+str.encode(str(lin_track_id)))
        data = self.radio_control.recv(1024)
        self.print("Result of Go2end: {}".format(data), thr=3)
        return data
    

    def parse_and_execute(self, receivedCMD):
        clientMsg = receivedCMD.decode()
        clientMsgParsed = clientMsg.split()

        if clientMsgParsed[0] == "Move":
            if len(clientMsgParsed) == 3:
                self.print('{}, {}'.format(clientMsgParsed[1], clientMsgParsed[2]), thr=5)
                motor_id = int(clientMsgParsed[1])
                distance = float(clientMsgParsed[2])
                success, status = self.obj_lintrack.displace(motor_id=motor_id, dis=distance)
                if success == True:
                    responseToCMD = self.config.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage

        elif clientMsgParsed[0] == "Return2home":
            if len(clientMsgParsed) == 2:
                motor_id = int(clientMsgParsed[1])
                success, status = self.obj_lintrack.return2home(motor_id=motor_id)
                if success == True:
                    responseToCMD = self.config.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage

        elif clientMsgParsed[0] == "Go2end":
            if len(clientMsgParsed) == 2:
                motor_id = int(clientMsgParsed[1])
                success, status = self.obj_lintrack.go2end(motor_id=motor_id)
                if success == True:
                    responseToCMD = self.config.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage
        else:
            responseToCMD = self.config.invalidCommandMessage
        
        responseToCMDInBytes = str.encode(responseToCMD + " (" + clientMsg + ")" )  
        return responseToCMDInBytes


@dataclass
class TCPComControllerConfig(TCPComConfig):
    pass

class Tcp_Comm_Controller(Tcp_Comm):
    def __init__(self, config: TCPComControllerConfig, **overrides):
        super().__init__(config, **overrides)

        self.obj_rfsoc = None
        self.obj_piradio = None

        self.print("Tcp_Comm_Controller object init done", thr=1)

    def set_frequency_piradio(self, fc=6.0e9, lo='high'):
        self.print("Setting frequency to {} GHz".format(fc/1e9), thr=3)
        self.radio_control.sendall(b"setFrequencyPiradio "+str.encode(str(fc) + " ") + str.encode(str(lo)))
        data = self.radio_control.recv(1024)
        self.print("Result of set_frequency_piradio: {}".format(data), thr=3)
        return data
    
    def set_gain_piradio(self, trx='tx', chan=0, gain_db=0):
        self.print("Setting gain to {} dB for {}-{}".format(gain_db, trx, chan), thr=3)
        self.radio_control.sendall(b"setGainPiradio "+str.encode(str(trx) + " ") + str.encode(str(chan) + " ") + str.encode(str(gain_db)))
        data = self.radio_control.recv(1024)
        self.print("Result of set_gain_piradio: {}".format(data), thr=3)
        return data
    
    def set_bias_piradio(self, chan, iq='I', bias_voltage=0):
        self.print("Setting bias to {} V for tx-{}-{}".format(bias_voltage, chan, iq), thr=3)
        self.radio_control.sendall(b"setBiasPiradio "+str.encode(str(chan) + " ") + str.encode(str(iq) + " ") + str.encode(str(bias_voltage)))
        data = self.radio_control.recv(1024)
        self.print("Result of set_bias_piradio: {}".format(data), thr=3)
        return data

    def transmit_data_rfsoc(self, txtd):
        txtd = txtd.copy()
        txtd = np.array(txtd).flatten()
        txtd = txtd * (2 ** (self.config.dac_bits + 1) - 1)
        re = txtd.real.astype(np.int16)
        im = txtd.imag.astype(np.int16)
        txtd = np.concatenate((re, im))

        self.radio_control.sendall(b"transmitSamplesRfsoc")
        self.radio_data.sendall(txtd.tobytes())
        data = self.radio_control.recv(1024)
        self.print("Result of transmit_data_rfsoc: {}".format(data),thr=3)
        return data

    def parse_and_execute(self, receivedCMD):
        clientMsg = receivedCMD.decode()
        clientMsgParsed = clientMsg.split()

        if clientMsgParsed[0] == "setFrequencyPiradio":
            if len(clientMsgParsed) == 3:
                freq = float(clientMsgParsed[1])
                lo = clientMsgParsed[2]
                result, response = self.obj_piradio.set_frequency_piradio(freq, lo=lo)
                responseToCMD = self.config.successMessage
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "setGainPiradio":
            if len(clientMsgParsed) == 4:
                trx = clientMsgParsed[1]
                chan = int(clientMsgParsed[2])
                gain_db = float(clientMsgParsed[3])
                result, response = self.obj_piradio.set_gain_piradio(trx=trx, chan=chan, gain_db=gain_db)
                responseToCMD = self.config.successMessage
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "setBiasPiradio":
            if len(clientMsgParsed) == 4:
                chan = int(clientMsgParsed[1])
                iq = clientMsgParsed[2]
                bias_voltage = float(clientMsgParsed[3])
                result, response = self.obj_piradio.set_bias_piradio(chan=chan, iq=iq, bias_voltage=bias_voltage)
                responseToCMD = self.config.successMessage
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "transmitSamplesRfsoc":
            if len(clientMsgParsed) == 1:
                nread = self.obj_rfsoc.n_tx_ant * self.obj_rfsoc.n_samples_tx
                nbytes = self.config.nbytes * nread * 2
                buf = bytearray()

                while len(buf) < nbytes:
                    data = self.connectionData.recv(nbytes)
                    buf.extend(data)
                data = np.frombuffer(buf, dtype=np.int16)
                data = data/(2 ** (self.obj_rfsoc.dac_bits + 1) - 1)
                txtd = data[:nread] + 1j*data[nread:]
                txtd = txtd.reshape(self.obj_rfsoc.n_tx_ant, nread//self.obj_rfsoc.n_tx_ant)

                self.obj_rfsoc.send_frame(txtd=txtd)
                responseToCMD = 'Success'
            else:
                responseToCMD = self.config.invalidNumberOfArgumentsMessage
        else:
            responseToCMD = self.config.invalidCommandMessage
        
        responseToCMDInBytes = str.encode(responseToCMD + " (" + clientMsg + ")" )  
        return responseToCMDInBytes



@dataclass
class SshComConfig(GeneralConfig):
    host_ip: str = '0.0.0.0'
    port: int = 22
    username: str = 'root'
    password: str = ' root'

class ssh_Com(General):
    def __init__(self, config: SshComConfig, **overrides):
        super().__init__(config, **overrides)

        self.print("ssh_Com object init done", thr=1)


    def init_ssh_client(self):
        try:
            # Initialize SSH client
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect to the remote server
            self.client.connect(hostname=self.config.host_ip, port=self.config.port, username=self.config.username, password=self.config.password, look_for_keys=False, allow_agent=False)

        except paramiko.AuthenticationException:
            print("Authentication failed. Please check your credentials.")
            traceback.print_exc()
        except paramiko.SSHException as e:
            print(f"SSH Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        self.print("ssh_Com client init done", thr=1)


    def close(self):
        self.client.close()
        self.print("SSH Client object closed", thr=1)


    def __del__(self):
        self.close()
        self.print("SSH Client object deleted", thr=1)


    def exec_command(self, command, verif_keyword=''):
        # Execute the command
        stdin, stdout, stderr = self.client.exec_command(command)

        # Capture command output and errors
        output = stdout.read().decode()
        errors = stderr.read().decode()

        if errors:
            self.print(f"Error: {errors}", thr=3)
        else:
            self.print(f"Command Output:\n{output}", thr=3)

        # Search for the keyword in the output
        if verif_keyword in output:
            self.print(f"Keyword '{verif_keyword}' found in the output.", thr=3)
            result = True
        else:
            self.print(f"Keyword '{verif_keyword}' not found in the output.", thr=3)
            result = False

        return result


@dataclass
class ScpComConfig(SshComConfig):
    pass

class Scp_Com(ssh_Com):
    def __init__(self, config: ScpComConfig, **overrides):
        super().__init__(config, **overrides)

        self.init_ssh_client()
        self.scp_clinet = SCPClient(self.client.get_transport())
        self.print("Scp_Com object init done", thr=1)


    # SCP files from the remote host
    def download_files(self, remote_files, local_dir):
        try:
            for remote_file in remote_files:
                try:
                    self.scp_clinet.get(remote_file, local_path=os.path.join(local_dir, os.path.basename(remote_file)))
                except:
                    self.print(f"Failed to download {remote_file}", thr=0)
            self.print("Files downloaded successfully!", thr=3)
        except:
            self.print("Files download failed!", thr=0)


    def download_files_with_pattern(self, remote_base_dir, remote_patterns, local_base_dir):
        try:
            for pattern in remote_patterns:
                pattern_ = os.path.join(remote_base_dir, pattern)
                remote_files = self.client.exec_command(f'ls {pattern_}')[1].read().decode().split()
                for remote_file in remote_files:
                    remote_file = os.path.join(remote_base_dir, remote_file) if not os.path.isabs(remote_file) else remote_file
                    relative_path = os.path.relpath(remote_file, remote_base_dir)
                    local_path = os.path.join(local_base_dir, relative_path)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    try:
                        self.scp_clinet.get(remote_file, local_path=local_path)
                    except:
                        self.print(f"Failed to download {remote_file}", thr=0)
            self.print("Files at {} downloaded successfully!".format(remote_patterns), thr=3)
        except:
            self.print("Files download failed!", thr=0)


    def close(self):
        self.scp_clinet.close()
        self.client.close()
        self.print("SCP Client object closed", thr=1)


@dataclass
class RestComConfig(GeneralConfig):
    ip_address: str = '0.0.0.0'
    port: int = 5000
    protocol: str = 'http'
    timeout: float = 5.0

class REST_Com(General):
    def __init__(self, config: RestComConfig, **overrides):
        super().__init__(config, **overrides)

        self._s = requests.Session()
        self.url_base = f"{self.config.protocol}://{self.config.ip_address}:{self.config.port}/"

        self.print("REST_Com object init done", thr=1)


    def init_rest_client(self):
        self.print("REST_Com client init done", thr=1)


    def close(self):
        self.print("REST_Com object closed", thr=1)


    def __del__(self):
        self.close()
        self.print("REST_Com object deleted", thr=1)


    def call_rest_api(self, url, params=None, verif_keyword=None):
        try:
            # response = requests.get(url, timeout=self.config.timeout)
            if params is None:
                response = self._s.get(url, timeout=self.config.timeout)
            else:
                response = self._s.get(url, params=params, timeout=self.config.timeout)

            response.raise_for_status()  # Raise an HTTPError for bad responses
            self.print("Successfully called the REST API:{}".format(response.json()), thr=3)

            # response =  response.json()
            response = json.loads(response.text)
            if type(response) == int or type(response) == float:
                response = str(response)
        except requests.exceptions.RequestException as e:
            self.print(f"Error executing REST API: {e}", thr=0)
            response = ''


        # Search for the keyword in the output
        if verif_keyword is None:
            result = True
        elif verif_keyword in response:
            self.print(f"Keyword '{verif_keyword}' found in the output.", thr=3)
            result = True
        else:
            self.print(f"Keyword '{verif_keyword}' not found in the output.", thr=3)
            result = False

        return result, response
    

@dataclass
class PiradioRestComConfig(RestComConfig):
    freq_sw_dly: float = 1.0
    gain_sw_dly: float = 1.0
    bias_sw_dly: float = 1.0
    freq_range: list = [6.0, 22.5]

class REST_Com_Piradio(REST_Com):
    def __init__(self, config: RestComConfig, **overrides):
        super().__init__(config, **overrides)

        self.print("REST_Com_Piradio object init done", thr=1)


    def initialize(self, verif_keyword='done'):
        self.print("Pi-Radio REST Comm Initialization done", thr=3)


    def set_frequency_piradio(self, fc=6.0e9, lo='high', verif_keyword=None):
        # command = f'high_lo?freq={fc}'
        url = f'{self.url_base}{lo}_lo'
        params = {'freq': fc}
        result, response = self.call_rest_api(url, params=params, verif_keyword=verif_keyword)
        if response == '':
            result = False
        else:
            result = (float(response['frequency']) == fc)
        if result:
            time.sleep(self.config.freq_sw_dly)
            self.print(f"Frequency set to {fc/1e9} GHz", thr=3)
        else:
            self.print(f"Failed to set frequency to {fc/1e9} GHz", thr=0)
        return result, response
    

    def set_gain_piradio(self, trx='tx', chan=0, gain_db=0, verif_keyword=None):
        chan = str(chan)
        url = self.url_base + 'gain'
        params={'trx': trx, 'chan': chan, 'v': gain_db}
        result, response = self.call_rest_api(url, params=params, verif_keyword=verif_keyword)
        if response == '':
            result = False
        else:
            result = (float(response[trx][chan]) == gain_db)
        if result:
            time.sleep(self.config.gain_sw_dly)
            self.print(f"Gain set to {gain_db} dB", thr=3)
        else:
            self.print(f"Failed to set gain to {gain_db} dB", thr=0)
        return result, response
    

    def set_bias_piradio(self, chan=0, iq='I', bias_voltage=0, verif_keyword=None):
        chan = str(chan)
        url = self.url_base + "bias"
        params={'iq': iq, 'chan': chan, 'v': bias_voltage}
        result, response = self.call_rest_api(url, params=params, verif_keyword=verif_keyword)
        if response == '':
            result = False
        else:
            result = (float(response[chan][iq]) == bias_voltage)
        if result:
            time.sleep(self.config.bias_sw_dly)
            self.print(f"Bias voltage set to {bias_voltage} V", thr=3)
        else:
            self.print(f"Failed to set bias voltage to {bias_voltage} V", thr=0)
        return result, response




