#!/usr/bin/env python

import pynq
from pynq import Overlay
from pynq import allocate
from pynq import MMIO
from pynq import Clocks
from pynq import GPIO
from pyftdi.ftdi import Ftdi
from siversController import *

import os
import time
import xrfclk
import xrfdc
import numpy as np
import socket
import matplotlib.pyplot as plt

run_sivers = True
_THIS_DIR = os.path.dirname(__file__)

CLOCKWIZARD_LOCK_ADDRESS = 0x0004
CLOCKWIZARD_RESET_ADDRESS = 0x0000
CLOCKWIZARD_RESET_TOKEN = 0x000A


class objNetworkInterface:
    def __init__(self):
        self.beam_test = np.array([1, 5, 9, 13, 17, 21, 25, 29, 32, 35, 39, 43, 47, 51, 55, 59, 63])

        # FPGA
        print("Starting the FPGA controller")
        self.overlay = Overlay(os.path.join(_THIS_DIR, "rfsoc4x2.xsa"))
        self.init_gpio()
        self.init_rf_clocks()
        self.verify_clock_tree()
        self.rfdc = self.overlay.usp_rf_data_converter_0
        self.rx_dma = self.overlay.adc_path.axi_dma_0
        self.tx_dma = self.overlay.dac_path.axi_dma_0
        self.rx_dma.set_up_rx_channel()
        self.tx_dma.set_up_tx_channel()
        self.rx_reg = self.overlay.adc_path.axis_flow_ctrl_0
        self.init_tile_sync()
        self.sync_tiles(dacTiles=0x5, adcTiles=0x5)
        print("FPGA controller is loaded.")

        # Sivers
        if run_sivers:
            print("Starting Sivers EVK controller")
            allDevices = Ftdi.list_devices()
            Ftdi.show_devices()
            strFTDIdesc = str(allDevices[0][0])
            snStr = strFTDIdesc[strFTDIdesc.find("sn=") + 4 : strFTDIdesc.find("sn=") + 14]
            siverEVKAddr = "ftdi://ftdi:4232:" + snStr
            print(siverEVKAddr)
            self.siversControllerObj = siversController(siverEVKAddr)
            self.siversControllerObj.init()
            print("Sivers EVK controller is loaded")

        ## TCP Server
        print("Starting TCP server")
        self.localIP = "0.0.0.0"
        self.bufferSize = 2**10

        ## Command
        self.localPort = 8080
        self.TCPServerSocket = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_STREAM
        )  # Create a datagram socket
        self.TCPServerSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.TCPServerSocket.bind((self.localIP, self.localPort))  # Bind to address and ip

        ## Data
        self.localPortData = 8081
        self.TCPServerSocketData = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_STREAM
        )  # Create a datagram socket
        self.TCPServerSocketData.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.TCPServerSocketData.bind((self.localIP, self.localPortData))  # Bind to address and ip

        bufsize = self.TCPServerSocketData.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        print("Buffer size [Before]:%d" % bufsize)
        print("TCP server up...")

        self.txtd = np.load(os.path.join(_THIS_DIR, "txtd.npy"))  # self.wideband()
        self.txtd /= np.max([np.abs(self.txtd.real), np.abs(self.txtd.imag)])
        self.txfd = np.fft.fft(self.txtd)
        self.txtd *= 2**13 - 1

    def onetone(self, sc=400, nfft=1024):
        # Create a tone in frequency-domain
        fd = np.zeros((nfft,), dtype="complex")
        fd[(nfft >> 1) + sc] = 1
        fd = np.fft.fftshift(fd, axes=0)

        # Convert the waveform to time-domain
        td = np.fft.ifft(fd, axis=0)

        # Normalize the signal
        td /= np.max([np.abs(td.real), np.abs(td.imag)])
        return td

    def wideband(self, sc_min=-400, sc_max=400, nfft=1024, mod="qam", seed=100):
        np.random.seed(seed)
        qam = (1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j)  # QAM symbols

        # Create the wideband sequence in frequency-domain
        fd = np.zeros((nfft,), dtype="complex")
        if mod == "qam":
            fd[((nfft >> 1) + sc_min) : ((nfft >> 1) + sc_max)] = np.random.choice(
                qam, len(range(sc_min, sc_max))
            )
        else:
            fd[((nfft >> 1) + sc_min) : ((nfft >> 1) + sc_max)] = 1

        fd[((nfft >> 1) - 10) : ((nfft >> 1) + 10)] = 0

        fd = np.fft.fftshift(fd, axes=0)

        # Convert the waveform to time-domain
        td = np.fft.ifft(fd, axis=0)

        # Normalize the signal
        td /= np.max([np.abs(td.real), np.abs(td.imag)])

        return td

    def init_gpio(self):
        # Todo: LMK reset should be changed here : Done
        # self.lmk_reset = GPIO(GPIO.get_gpio_pin(84), 'out')
        # 78: number of PS_MIO pins, 7: index of lmk_reset MIO pin
        self.lmk_reset = GPIO(GPIO.get_gpio_pin(-78 + 7), "out")
        self.adc_reset = GPIO(GPIO.get_gpio_pin(32), "out")
        self.dac_reset = GPIO(GPIO.get_gpio_pin(0), "out")
        self.dac_mux = GPIO(GPIO.get_gpio_pin(3), "out")
        self.adc_enable = GPIO(GPIO.get_gpio_pin(34), "out")
        self.dac_enable = GPIO(GPIO.get_gpio_pin(2), "out")
        self.led = GPIO(GPIO.get_gpio_pin(80), "out")

        # Init Values
        self.led.write(0)
        self.adc_enable.write(0)
        self.dac_enable.write(0)
        self.adc_reset.write(1)
        self.dac_reset.write(1)

    def init_rf_clocks(self):
        self.lmk_reset.write(1)
        self.lmk_reset.write(0)
        # Todo: Frequencies and xrfclk should be checked here : Done
        xrfclk.set_ref_clks(lmk_freq=122.88, lmx_freq=3932.16)

    def verify_clock_tree(self):
        status = self.overlay.clocktreeMTS.clk_wiz_0.read(CLOCKWIZARD_LOCK_ADDRESS)
        if status != 1:
            raise Exception(
                "The MTS ClockTree has failed to LOCK. Please verify board clocking configuration"
            )

    def init_tile_sync(self):
        self.sync_tiles(0x1, 0x1)
        self.overlay.clocktreeMTS.clk_wiz_0.mmio.write_reg(
            CLOCKWIZARD_RESET_ADDRESS, CLOCKWIZARD_RESET_TOKEN
        )
        time.sleep(0.1)

        self.rfdc.dac_tiles[0].Reset()
        self.rfdc.dac_tiles[2].Reset()

        for toggleValue in range(0, 1):
            self.rfdc.adc_tiles[0].SetupFIFO(toggleValue)
            self.rfdc.adc_tiles[2].SetupFIFO(toggleValue)

    # Todo: Tiles indices should be checked, solve the problem of mts_dac_config unknown : Done
    def sync_tiles(self, dacTiles=0, adcTiles=0):
        self.rfdc.mts_dac_config.RefTile = 0  # MTS starts at DAC Tile 228
        self.rfdc.mts_adc_config.RefTile = 0  # MTS starts at ADC Tile 224
        self.rfdc.mts_dac_config.Target_Latency = -1
        self.rfdc.mts_adc_config.Target_Latency = -1
        if dacTiles > 0:
            self.rfdc.mts_dac_config.Tiles = dacTiles  # group defined in binary 0b1111
            self.rfdc.mts_dac_config.SysRef_Enable = 1
            self.rfdc.mts_dac()
        else:
            self.rfdc.mts_dac_config.Tiles = 0x0
            self.rfdc.mts_dac_config.SysRef_Enable = 0

        if adcTiles > 0:
            self.rfdc.mts_adc_config.Tiles = adcTiles
            self.rfdc.mts_adc_config.SysRef_Enable = 1
            self.rfdc.mts_adc()
        else:
            self.rfdc.mts_adc_config.Tiles = 0x0
            self.rfdc.mts_adc_config.SysRef_Enable = 0

    def send(self, mode="ChannelSounder", freq=100, amp=(0.5**4) * (2**16 - 1)):
        npar = 4

        if mode == "ChannelSounder":
            x_real = np.int16(self.txtd.real).reshape(-1, npar)
            x_imag = np.int16(self.txtd.imag).reshape(-1, npar)
        elif mode == "Sinusoid":
            t = np.linspace(0, 1, (self.txtd.real).shape[0])
            wt = np.multiply(t, 2 * np.pi * freq)
            x_real = amp * (np.cos(wt))
            x_real = np.int16(x_real).reshape(-1, npar)
            x_imag = 0 * amp * (np.sin(wt))
            x_imag = np.int16(x_imag).reshape(-1, npar)
        else:
            print("Mode is not supported!")
            return

        # Combine the real and imaginary data. Flatten the output buffer.
        data = np.zeros((x_real.shape[0] * 2, x_real.shape[1]), dtype="int16")
        data[1::2, :] = np.int16(x_real)
        data[::2, :] = np.int16(x_imag)
        data = data.flatten()
        input_buffer = allocate(shape=(x_real.shape[0] * x_real.shape[1] * 2,), dtype=np.int16)
        input_buffer[:] = data[:]
        self.dac_mux.write(0)
        self.dac_enable.write(0)
        self.dac_reset.write(1)
        time.sleep(0.5)
        self.dac_reset.write(0)
        self.tx_dma.sendchannel.transfer(input_buffer)
        self.tx_dma.sendchannel.wait()
        self.dac_mux.write(1)
        self.dac_enable.write(1)

    def recv_once(self, nframe=1):
        nread = 1024
        nskip = 0
        npar = 4

        output_buffer = allocate(
            shape=(nread * nframe * 2,), target=self.overlay.ddr4_0, dtype=np.int16
        )

        self.rx_reg.write(0, nread // npar)
        self.rx_reg.write(4, nskip // npar)
        self.rx_reg.write(8, nframe * nread * 4)

        self.adc_reset.write(0)
        self.rx_dma.recvchannel.transfer(output_buffer)
        self.adc_enable.write(1)
        self.rx_dma.recvchannel.wait()
        self.adc_enable.write(0)
        self.adc_reset.write(1)
        data = np.array(output_buffer).astype("int16")
        data = data.reshape(-1, npar)
        data = data[::2, :] + 1j * data[1::2, :]
        # data = data[::2,:]
        return data.reshape(-1)

    def recv(self, nframe=1):
        nread = 1024
        nskip = 0
        npar = 4

        output_buffer = allocate(
            shape=(nread * nframe * 2,), target=self.overlay.ddr4_0, dtype=np.int16
        )

        self.rx_reg.write(0, nread // npar)
        self.rx_reg.write(4, nskip // npar)
        self.rx_reg.write(8, nframe * nread * 4)

        rxtd = np.zeros((len(self.beam_test), nread), dtype="complex")
        for i, beam_index in enumerate(self.beam_test):
            if run_sivers:
                self.siversControllerObj.setBeamIndexRX(beam_index)
            self.adc_reset.write(0)
            self.rx_dma.recvchannel.transfer(output_buffer)
            self.adc_enable.write(1)
            self.rx_dma.recvchannel.wait()
            self.adc_enable.write(0)
            self.adc_reset.write(1)
            data = np.array(output_buffer).astype("int16")
            data = data.reshape(-1, npar)
            data = data[::2, :] + 1j * data[1::2, :]
            rxtd[i, :] = data.reshape(-1)
        # return rxtd
        rxfd = np.fft.fft(rxtd, axis=1)
        rxfd = np.roll(rxfd, 1, axis=1)
        Hest = rxfd * np.conj(self.txfd)
        hest = np.fft.ifft(Hest, axis=1)
        hest = hest.flatten()
        re = hest.real.astype(np.int16)
        im = hest.imag.astype(np.int16)
        return np.concatenate((re, im))

    def run(self):
        # Listen for incoming connections
        self.TCPServerSocket.listen(1)
        self.TCPServerSocketData.listen(1)

        while True:
            # Wait for a connection
            print("\nWaiting for a connection")
            self.connectionCMD, addrCMD = self.TCPServerSocket.accept()
            self.connectionData, addrDATA = self.TCPServerSocketData.accept()

            after_idle_sec = 1
            interval_sec = 3
            max_fails = 5
            self.connectionData.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.connectionData.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, after_idle_sec)
            self.connectionData.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval_sec)
            self.connectionData.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, max_fails)

            self.connectionCMD.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.connectionCMD.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, after_idle_sec)
            self.connectionCMD.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval_sec)
            self.connectionCMD.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, max_fails)

            try:
                while True:
                    try:
                        receivedCMD = self.connectionCMD.recv(self.bufferSize)
                        if receivedCMD:
                            print("\nClient CMD:{}".format(receivedCMD.decode()))
                            responseToCMDinBytes = self.parseAndExecute(receivedCMD)
                            self.connectionCMD.sendall(responseToCMDinBytes)
                        else:
                            break
                    except:
                        break
            finally:
                # Clean up the connection
                print("\nConnection is closed.")
                self.connectionCMD.close()
                self.connectionData.close()

    def parseAndExecute(self, receivedCMD):
        clientMsg = receivedCMD.decode()
        invalidCommandMessage = "ERROR: Invalid command"
        invalidNumberOfArgumentsMessage = "ERROR: Invalid number of arguments"
        successMessage = "Successully executed"
        droppedMessage = "Connection dropped?"
        clientMsgParsed = clientMsg.split()
        if clientMsgParsed[0] == "receiveSamplesOnce":
            if len(clientMsgParsed) == 1:
                iq_data = self.recv_once(1)
                self.connectionData.sendall(iq_data.tobytes())
                responseToCMD = "Success"
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        if clientMsgParsed[0] == "receiveSamples":
            if len(clientMsgParsed) == 1:
                iq_data = self.recv(1)
                self.connectionData.sendall(iq_data.tobytes())
                responseToCMD = "Success"
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "transmitSamples":
            if len(clientMsgParsed) == 1:
                self.send()
                responseToCMD = "Success"
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "getBeamIndexTX":
            if len(clientMsgParsed) == 1:
                responseToCMD = str(self.siversControllerObj.getBeamIndexTX())
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "setBeamIndexTX":
            if len(clientMsgParsed) == 2:
                beamIndex = int(clientMsgParsed[1])
                success, status = self.siversControllerObj.setBeamIndexTX(beamIndex)
                if success == True:
                    responseToCMD = successMessage
                else:
                    responseToCMD = status
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "getBeamIndexRX":
            if len(clientMsgParsed) == 1:
                responseToCMD = str(self.siversControllerObj.getBeamIndexRX())
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "setBeamIndexRX":
            if len(clientMsgParsed) == 2:
                beamIndex = int(clientMsgParsed[1])
                success, status = self.siversControllerObj.setBeamIndexRX(beamIndex)
                if success == True:
                    responseToCMD = successMessage
                else:
                    responseToCMD = status
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "getModeSiver":
            if len(clientMsgParsed) == 1:
                responseToCMD = self.siversControllerObj.getMode()
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "setModeSiver":
            if len(clientMsgParsed) == 2:
                mode = clientMsgParsed[1]
                success, status = self.siversControllerObj.setMode(mode)
                if success == True:
                    responseToCMD = successMessage
                else:
                    responseToCMD = status
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "getGainRX":
            if len(clientMsgParsed) == 1:
                (
                    rx_gain_ctrl_bb1,
                    rx_gain_ctrl_bb2,
                    rx_gain_ctrl_bb3,
                    rx_gain_ctrl_bfrf,
                    agc_int_bfrf_gain_lvl,
                    agc_int_bb3_gain_lvl,
                ) = self.siversControllerObj.getGainRX()
                responseToCMD = (
                    "rx_gain_ctrl_bb1:"
                    + str(hex(rx_gain_ctrl_bb1))
                    + ", rx_gain_ctrl_bb2:"
                    + str(hex(rx_gain_ctrl_bb2))
                    + ", rx_gain_ctrl_bb3:"
                    + str(hex(rx_gain_ctrl_bb3))
                    + ", rx_gain_ctrl_bfrf:"
                    + str(hex(rx_gain_ctrl_bfrf))
                    + ", agc_int_bfrf_gain_lvl:"
                    + str(hex(agc_int_bfrf_gain_lvl))
                    + ", agc_int_bb3_gain_lvl:"
                    + str(hex(agc_int_bb3_gain_lvl))
                )
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "setGainRX":
            if len(clientMsgParsed) == 5:
                rx_gain_ctrl_bb1 = int(clientMsgParsed[1])
                rx_gain_ctrl_bb2 = int(clientMsgParsed[2])
                rx_gain_ctrl_bb3 = int(clientMsgParsed[3])
                rx_gain_ctrl_bfrf = int(clientMsgParsed[4])

                success, status = self.siversControllerObj.setGainRX(
                    rx_gain_ctrl_bb1, rx_gain_ctrl_bb2, rx_gain_ctrl_bb3, rx_gain_ctrl_bfrf
                )
                if success == True:
                    responseToCMD = successMessage
                else:
                    responseToCMD = status
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "getGainTX":
            if len(clientMsgParsed) == 1:
                tx_bb_gain, tx_bb_phase, tx_bb_iq_gain, tx_bfrf_gain, tx_ctrl = (
                    self.siversControllerObj.getGainTX()
                )
                responseToCMD = (
                    "tx_bb_gain:"
                    + str(hex(tx_bb_gain))
                    + ", tx_bb_phase:"
                    + str(hex(tx_bb_phase))
                    + ", tx_bb_gain:"
                    + str(hex(tx_bb_iq_gain))
                    + ", tx_bfrf_gain:"
                    + str(hex(tx_bfrf_gain))
                    + ", tx_ctrl:"
                    + str(hex(tx_ctrl))
                )
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "setGainTX":
            if len(clientMsgParsed) == 5:
                print(clientMsgParsed[1])

                tx_bb_gain = int(clientMsgParsed[1])
                tx_bb_phase = int(clientMsgParsed[2])
                tx_bb_iq_gain = int(clientMsgParsed[3])
                tx_bfrf_gain = int(clientMsgParsed[4])

                success, status = self.siversControllerObj.setGainTX(
                    tx_bb_gain, tx_bb_phase, tx_bb_iq_gain, tx_bfrf_gain
                )
                if success == True:
                    responseToCMD = successMessage
                else:
                    responseToCMD = status
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "getCarrierFrequency":
            if len(clientMsgParsed) == 1:
                responseToCMD = str(self.siversControllerObj.getFrequency())
            else:
                responseToCMD = invalidNumberOfArgumentsMessage
        elif clientMsgParsed[0] == "setCarrierFrequency":
            if len(clientMsgParsed) == 2:
                print(clientMsgParsed[1])
                fc = float(clientMsgParsed[1])
                success, status = self.siversControllerObj.setFrequency(fc)
                if success == True:
                    responseToCMD = successMessage
                else:
                    responseToCMD = status
            else:
                responseToCMD = invalidNumberOfArgumentsMessage

        #######################
        else:
            responseToCMD = invalidCommandMessage

        responseToCMDInBytes = str.encode(responseToCMD + " (" + clientMsg + ")")
        return responseToCMDInBytes


if __name__ == "__main__":
    networkInterfaceObj = objNetworkInterface()

    networkInterfaceObj.run()
