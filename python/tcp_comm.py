from backend import *
from backend import be_np as np, be_scp as scipy
from sigcom-toolkit.general import General




class Tcp_Comm(General):
    def __init__(self, params):
        super().__init__(params)

        self.server_ip = getattr(params, 'server_ip', '0.0.0.0')
        self.TCP_port_Cmd = getattr(params, 'TCP_port_Cmd', 8080)
        self.TCP_port_Data = getattr(params, 'TCP_port_Data', 8081)
        self.tcp_localIP = getattr(params, 'tcp_localIP', '0.0.0.0')
        self.tcp_bufferSize = getattr(params, 'tcp_bufferSize', 2**10)
        self.after_idle_sec = 1
        self.interval_sec = 3
        self.max_fails = 5

        self.nbytes = 2

        self.invalidCommandMessage = "ERROR: Invalid command"
        self.invalidNumberOfArgumentsMessage = "ERROR: Invalid number of arguments"
        self.successMessage = "Successully executed"
        self.droppedMessage = "Connection dropped?"

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
        self.TCPServerSocketCmd.bind((self.tcp_localIP, self.TCP_port_Cmd)) # Bind to address and ip
        
        ## Data
        self.TCPServerSocketData = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)         # Create a datagram socket
        self.TCPServerSocketData.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.TCPServerSocketData.bind((self.tcp_localIP, self.TCP_port_Data))                # Bind to address and ip

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
            self.connectionData.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.after_idle_sec)
            self.connectionData.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.interval_sec)
            self.connectionData.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.max_fails)
            
            self.connectionCMD.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.connectionCMD.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.after_idle_sec)
            self.connectionCMD.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.interval_sec)
            self.connectionCMD.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.max_fails)            
            
            try:
                while True:
                    try:
                        receivedCMD = self.connectionCMD.recv(self.tcp_bufferSize)
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
        self.radio_control.connect((self.server_ip, self.TCP_port_Cmd))

        self.radio_data = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.radio_data.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.radio_data.connect((self.server_ip, self.TCP_port_Data))

        self.print("Client succesfully connected to the server", thr=1)



class Tcp_Comm_RFSoC(Tcp_Comm):
    def __init__(self, params):
        params = params.copy()
        params.server_ip = params.rfsoc_server_ip
        super().__init__(params)

        self.obj_rfsoc = None

        self.fc = params.fc
        self.beam_test = params.beam_test
        self.adc_bits = params.adc_bits
        self.dac_bits = params.dac_bits
        self.RFFE = params.RFFE
        self.n_frame_rd = params.n_frame_rd
        self.n_samples = params.n_samples
        self.n_tx_ant = params.n_tx_ant
        self.n_rx_ant = params.n_rx_ant

        if self.RFFE=='sivers':
            self.tx_bb_gain = 0x3
            self.tx_bb_phase = 0x0
            self.tx_bb_iq_gain = 0x77
            self.tx_bfrf_gain = 0x7F
            self.rx_gain_ctrl_bb1 = 0x33
            self.rx_gain_ctrl_bb2 = 0x00
            self.rx_gain_ctrl_bb3 = 0x33
            self.rx_gain_ctrl_bfrf = 0x7F

        self.nread = self.n_rx_ant * self.n_frame_rd * self.n_samples

        self.print("Tcp_Comm_RFSoC object init done", thr=1)

    def set_mode(self, mode):
        if mode == 'RXen0_TXen1' or mode == 'RXen1_TXen0' or mode == 'RXen0_TXen0':
            self.radio_control.sendall(b"setModeSiver "+str.encode(str(mode)))
            data = self.radio_control.recv(1024)
            self.print("Result of set_mode: {}".format(data),thr=3)
            return data
        
    def set_frequency_sivers(self, fc):
        self.radio_control.sendall(b"setCarrierFrequencySivers "+str.encode(str(fc)))
        data = self.radio_control.recv(1024)
        self.print("Result of set_frequency_sivers: {}".format(data),thr=3)
        return data
    
    def set_frequency_mixer(self, f_mixer_dac, f_mixer_adc):
        self.radio_control.sendall(b"setFrequencyMixer "+str.encode(str(f_mixer_dac) + " ") + str.encode(str(f_mixer_adc)))
        data = self.radio_control.recv(1024)
        self.print("Result of set_frequency_mixer: {}".format(data),thr=3)
        return data

    def set_tx_gain(self):
        self.radio_control.sendall(b"setGainTXSivers " + str.encode(str(int(self.tx_bb_gain)) + " ") \
                                                    + str.encode(str(int(self.tx_bb_phase)) + " ") \
                                                    + str.encode(str(int(self.tx_bb_iq_gain)) + " ") \
                                                    + str.encode(str(int(self.tx_bfrf_gain))))
        data = self.radio_control.recv(1024)
        self.print("Result of set_tx_gain: {}".format(data),thr=3)
        return data

    def set_rx_gain(self):
        self.radio_control.sendall(b"setGainRXSivers " + str.encode(str(int(self.rx_gain_ctrl_bb1)) + " ") \
                                                    + str.encode(str(int(self.rx_gain_ctrl_bb2)) + " ") \
                                                    + str.encode(str(int(self.rx_gain_ctrl_bb3)) + " ") \
                                                    + str.encode(str(int(self.rx_gain_ctrl_bfrf))))
        data = self.radio_control.recv(1024)
        self.print("Result of set_rx_gain: {}".format(data),thr=3)
        return data

    def transmit_data_default(self):
        self.radio_control.sendall(b"transmitSamplesDefault")
        data = self.radio_control.recv(1024)
        self.print("Result of transmit_data_default: {}".format(data),thr=3)
        return data
    
    def transmit_data(self, txtd):
        txtd = txtd.copy()
        txtd = np.array(txtd).flatten()
        txtd = txtd * (2 ** (self.dac_bits + 1) - 1)
        re = txtd.real.astype(np.int16)
        im = txtd.imag.astype(np.int16)
        txtd = np.concatenate((re, im))

        self.radio_control.sendall(b"transmitSamples")
        self.radio_data.sendall(txtd.tobytes())
        data = self.radio_control.recv(1024)
        self.print("Result of transmit_data: {}".format(data),thr=3)
        return data

    def receive_data(self, mode='once'):
        if mode=='once':
            nbeams = 1
            self.radio_control.sendall(b"receiveSamplesOnce")
        elif mode=='beams':
            nbeams = len(self.beam_test)
            self.radio_control.sendall(b"receiveSamples")
        nbytes = nbeams * self.nbytes * self.nread * 2
        buf = bytearray()

        while len(buf) < nbytes:
            data = self.radio_data.recv(nbytes)
            buf.extend(data)
        data = np.frombuffer(buf, dtype=np.int16)
        data = data/(2 ** (self.adc_bits + 1) - 1)
        rxtd = data[:self.nread*nbeams] + 1j*data[self.nread*nbeams:]
        rxtd = rxtd.reshape(nbeams, self.n_rx_ant, self.nread//self.n_rx_ant)
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
                responseToCMD = self.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "receiveSamples":
            if len(clientMsgParsed) == 1:
                iq_data = self.obj_rfsoc.recv_frame(n_frame=self.obj_rfsoc.n_frame_rd)
                re = iq_data.real.astype(np.int16)
                im = iq_data.imag.astype(np.int16)
                iq_data = np.concatenate((re, im))
                self.connectionData.sendall(iq_data.tobytes())
                responseToCMD = "Success"
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "transmitSamplesDefault":
            if len(clientMsgParsed) == 1:
                self.obj_rfsoc.send_frame(txtd=self.obj_rfsoc.txtd)
                responseToCMD = 'Success'
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "transmitSamples":
            if len(clientMsgParsed) == 1:
                nread = self.obj_rfsoc.n_tx_ant * self.obj_rfsoc.n_samples_tx
                nbytes = self.nbytes * nread * 2
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
                responseToCMD = self.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "getBeamIndexTXSivers":
            if len(clientMsgParsed) == 1:
                responseToCMD = str(self.obj_rfsoc.siversControllerObj.getBeamIndexTX())
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage 
        elif clientMsgParsed[0] == "setBeamIndexTXSivers":
            if len(clientMsgParsed) == 2:
                beamIndex = int(clientMsgParsed[1])
                success, status = self.obj_rfsoc.siversControllerObj.setBeamIndexTX(beamIndex)
                if success == True:
                    responseToCMD = self.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage  
        elif clientMsgParsed[0] == "getBeamIndexRXSivers":
            if len(clientMsgParsed) == 1:
                responseToCMD = str(self.obj_rfsoc.siversControllerObj.getBeamIndexRX())
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage 
        elif clientMsgParsed[0] == "setBeamIndexRXSivers":
            if len(clientMsgParsed) == 2:
                beamIndex = int(clientMsgParsed[1])
                success, status = self.obj_rfsoc.siversControllerObj.setBeamIndexRX(beamIndex)
                if success == True:
                    responseToCMD = self.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "getModeSiver":
            if len(clientMsgParsed) == 1:
                responseToCMD = self.obj_rfsoc.siversControllerObj.getMode()
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage 
        elif clientMsgParsed[0] == "setModeSiver":
            if len(clientMsgParsed) == 2:
                mode = clientMsgParsed[1]
                success,status = self.obj_rfsoc.siversControllerObj.setMode(mode)
                if success == True:
                    responseToCMD = self.successMessage 
                else:
                    responseToCMD = status                  
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage    
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
                responseToCMD = self.invalidNumberOfArgumentsMessage 
        elif clientMsgParsed[0] == "setGainRXSivers":
            if len(clientMsgParsed) == 5:
                rx_gain_ctrl_bb1 = int(clientMsgParsed[1])
                rx_gain_ctrl_bb2 = int(clientMsgParsed[2])
                rx_gain_ctrl_bb3 = int(clientMsgParsed[3])
                rx_gain_ctrl_bfrf = int(clientMsgParsed[4])
                
                success,status = self.obj_rfsoc.siversControllerObj.setGainRX(rx_gain_ctrl_bb1, rx_gain_ctrl_bb2, rx_gain_ctrl_bb3, rx_gain_ctrl_bfrf)
                if success == True:
                    responseToCMD = self.successMessage 
                else:
                    responseToCMD = status                  
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage      
        elif clientMsgParsed[0] == "getGainTXSivers":
            if len(clientMsgParsed) == 1:
                tx_bb_gain, tx_bb_phase, tx_bb_iq_gain, tx_bfrf_gain, tx_ctrl = self.obj_rfsoc.siversControllerObj.getGainTX()
                responseToCMD = 'tx_bb_gain:' + str(hex(tx_bb_gain)) + \
                                ', tx_bb_phase:' +  str(hex(tx_bb_phase)) + \
                                ', tx_bb_gain:' +   str(hex(tx_bb_iq_gain)) + \
                                ', tx_bfrf_gain:' +   str(hex(tx_bfrf_gain)) + \
                                ', tx_ctrl:' +   str(hex(tx_ctrl))
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage 
        elif clientMsgParsed[0] == "setGainTXSivers":
            if len(clientMsgParsed) == 5:
                self.print(clientMsgParsed[1], thr=2)
                
                tx_bb_gain = int(clientMsgParsed[1])
                tx_bb_phase = int(clientMsgParsed[2])
                tx_bb_iq_gain = int(clientMsgParsed[3])
                tx_bfrf_gain = int(clientMsgParsed[4])
                
                success,status = self.obj_rfsoc.siversControllerObj.setGainTX(tx_bb_gain, tx_bb_phase, tx_bb_iq_gain, tx_bfrf_gain)
                if success == True:
                    responseToCMD = self.successMessage 
                else:
                    responseToCMD = status                  
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage   
        elif clientMsgParsed[0] == "getCarrierFrequencySivers":
            if len(clientMsgParsed) == 1:
                responseToCMD = str(self.obj_rfsoc.siversControllerObj.getFrequency())
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage 
        elif clientMsgParsed[0] == "setCarrierFrequencySivers":
            if len(clientMsgParsed) == 2:
                self.print(clientMsgParsed[1], thr=2)
                fc = float(clientMsgParsed[1])
                success, status = self.obj_rfsoc.siversControllerObj.setFrequency(fc)
                if success == True:
                    responseToCMD = self.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "setFrequencyMixer":
            if len(clientMsgParsed) == 3:
                self.print(clientMsgParsed[1], thr=2)

                f_mixer_dac = float(clientMsgParsed[1])
                f_mixer_adc = float(clientMsgParsed[2])
                success = self.obj_rfsoc.set_dac_mixer(mix_freq=f_mixer_dac, do_mixer_settings=True)
                success &= self.obj_rfsoc.set_adc_mixer(mix_freq=f_mixer_adc, do_mixer_settings=True)
                if success == True:
                    responseToCMD = self.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage
                
        #######################
        else:
            responseToCMD = self.invalidCommandMessage
        
        responseToCMDInBytes = str.encode(responseToCMD + " (" + clientMsg + ")" )  
        return responseToCMDInBytes
    


class Tcp_Comm_LinTrack(Tcp_Comm):
    def __init__(self, params):
        params = params.copy()
        params.server_ip = params.lintrack_server_ip
        super().__init__(params)
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
                    responseToCMD = self.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage

        elif clientMsgParsed[0] == "Return2home":
            if len(clientMsgParsed) == 2:
                motor_id = int(clientMsgParsed[1])
                success, status = self.obj_lintrack.return2home(motor_id=motor_id)
                if success == True:
                    responseToCMD = self.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage

        elif clientMsgParsed[0] == "Go2end":
            if len(clientMsgParsed) == 2:
                motor_id = int(clientMsgParsed[1])
                success, status = self.obj_lintrack.go2end(motor_id=motor_id)
                if success == True:
                    responseToCMD = self.successMessage 
                else:
                    responseToCMD = status 
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage

        else:
            responseToCMD = self.invalidCommandMessage
        
        responseToCMDInBytes = str.encode(responseToCMD + " (" + clientMsg + ")" )  
        return responseToCMDInBytes


class Tcp_Comm_Controller(Tcp_Comm):
    def __init__(self, params):
        params = params.copy()
        params.server_ip = params.controller_slave_ip
        super().__init__(params)

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
    
    def parse_and_execute(self, receivedCMD):
        clientMsg = receivedCMD.decode()
        clientMsgParsed = clientMsg.split()

        if clientMsgParsed[0] == "setFrequencyPiradio":
            if len(clientMsgParsed) == 3:
                freq = float(clientMsgParsed[1])
                lo = clientMsgParsed[2]
                result, response = self.obj_piradio.set_frequency(freq, lo=lo)
                responseToCMD = self.successMessage
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "setGainPiradio":
            if len(clientMsgParsed) == 4:
                trx = clientMsgParsed[1]
                chan = int(clientMsgParsed[2])
                gain_db = float(clientMsgParsed[3])
                result, response = self.obj_piradio.set_gain(trx=trx, chan=chan, gain_db=gain_db)
                responseToCMD = self.successMessage
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "setBiasPiradio":
            if len(clientMsgParsed) == 4:
                chan = int(clientMsgParsed[1])
                iq = clientMsgParsed[2]
                bias_voltage = float(clientMsgParsed[3])
                result, response = self.obj_piradio.set_bias(chan=chan, iq=iq, bias_voltage=bias_voltage)
                responseToCMD = self.successMessage
            else:
                responseToCMD = self.invalidNumberOfArgumentsMessage
        else:
            responseToCMD = self.invalidCommandMessage
        
        responseToCMDInBytes = str.encode(responseToCMD + " (" + clientMsg + ")" )  
        return responseToCMDInBytes



class ssh_Com(General):
    def __init__(self, params):
        super().__init__(params)

        self.host_ip = getattr(params, 'host_ip', '0.0.0.0')
        self.port = getattr(params, 'port', 22)
        self.username = getattr(params, 'username', 'root')
        self.password = getattr(params, 'password', ' root')

        self.print("ssh_Com object init done", thr=1)


    def init_ssh_client(self):
        try:
            # Initialize SSH client
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect to the remote server
            self.client.connect(hostname=self.host_ip, port=self.port, username=self.username, password=self.password, look_for_keys=False, allow_agent=False)

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





class Scp_Com(ssh_Com):
    def __init__(self, params):
        super().__init__(params)

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



class REST_Com(General):
    def __init__(self, params):
        super().__init__(params)

        self.host_ip = getattr(params, 'host_ip', '0.0.0.0')
        self.port = getattr(params, 'port', 5000)
        self.protocol = getattr(params, 'protocol', 'http')
        self.timeout = getattr(params, 'timeout', 5)

        self._s = requests.Session()
        self.url_base = f"{self.protocol}://{self.host_ip}:{self.port}/"

        self.print("REST_Com object init done", thr=1)


    def init_rest_client(self):
        self.print("REST_Com client init done", thr=1)


    def close(self):
        self.print("REST_Com object closed", thr=1)


    def __del__(self):
        self.close()
        self.print("REST_Com object deleted", thr=1)


    def call_rest_api(self, url, params=None, verif_keyword=None):
        # url = f"{self.protocol}://{self.host_ip}:{self.port}/{command}"

        try:
            # response = requests.get(url, timeout=self.timeout)
            if params is None:
                response = self._s.get(url, timeout=self.timeout)
            else:
                response = self._s.get(url, params=params, timeout=self.timeout)

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
    


class REST_Com_Piradio(REST_Com):
    def __init__(self, params):
        params = params.copy()
        params.host_ip = params.piradio_host
        params.port = params.piradio_rest_port
        params.protocol = getattr(params, 'piradio_rest_protocol', 'http')
        super().__init__(params)

        self.freq_sw_dly = getattr(params, 'piradio_freq_sw_dly', 1.0)
        self.gain_sw_dly = getattr(params, 'piradio_gain_sw_dly', 1.0)
        self.bias_sw_dly = getattr(params, 'piradio_bias_sw_dly', 1.0)
        self.freq_range = getattr(params, 'piradio_freq_range', [6.0, 22.5])

        self.print("REST_Com_Piradio object init done", thr=1)


    def initialize(self, verif_keyword='done'):
        self.print("Pi-Radio REST Comm Initialization done", thr=3)


    def set_frequency(self, fc=6.0e9, lo='high', verif_keyword=None):
        # command = f'high_lo?freq={fc}'
        url = f'{self.url_base}{lo}_lo'
        params = {'freq': fc}
        result, response = self.call_rest_api(url, params=params, verif_keyword=verif_keyword)
        if response == '':
            result = False
        else:
            result = (float(response['frequency']) == fc)
        if result:
            time.sleep(self.freq_sw_dly)
            self.print(f"Frequency set to {fc/1e9} GHz", thr=3)
        else:
            self.print(f"Failed to set frequency to {fc/1e9} GHz", thr=0)
        return result, response
    

    def set_gain(self, trx='tx', chan=0, gain_db=0, verif_keyword=None):
        chan = str(chan)
        url = self.url_base + 'gain'
        params={'trx': trx, 'chan': chan, 'v': gain_db}
        result, response = self.call_rest_api(url, params=params, verif_keyword=verif_keyword)
        if response == '':
            result = False
        else:
            result = (float(response[trx][chan]) == gain_db)
        if result:
            time.sleep(self.gain_sw_dly)
            self.print(f"Gain set to {gain_db} dB", thr=3)
        else:
            self.print(f"Failed to set gain to {gain_db} dB", thr=0)
        return result, response
    

    def set_bias(self, chan=0, iq='I', bias_voltage=0, verif_keyword=None):
        chan = str(chan)
        url = self.url_base + "bias"
        params={'iq': iq, 'chan': chan, 'v': bias_voltage}
        result, response = self.call_rest_api(url, params=params, verif_keyword=verif_keyword)
        if response == '':
            result = False
        else:
            result = (float(response[chan][iq]) == bias_voltage)
        if result:
            time.sleep(self.bias_sw_dly)
            self.print(f"Bias voltage set to {bias_voltage} V", thr=3)
        else:
            self.print(f"Failed to set bias voltage to {bias_voltage} V", thr=0)
        return result, response




