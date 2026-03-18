import contextlib
import json
import os
import socket
import time
import traceback
from dataclasses import dataclass

import numpy as np
import paramiko
import requests
from scp import SCPClient
from sigcom_toolkit.general import General, GeneralConfig


@dataclass(kw_only=True)
class TCPComConfig(GeneralConfig):
    server_ip: str = "0.0.0.0"
    TCP_port_Cmd: int = 8080
    TCP_port_Data: int = 8081
    tcp_local_ip: str = "0.0.0.0"
    tcp_buffer_size: int = 2**10
    after_idle_sec: int = 1
    interval_sec: int = 3
    max_fails: int = 5
    nbytes: int = 2
    timeout: float = 5.0

    invalid_command_message: str = "ERROR: Invalid command"
    invalid_number_of_arguments_message: str = "ERROR: Invalid number of arguments"
    success_message: str = "Successully executed"
    dropped_message: str = "Connection dropped?"


class TcpComm(General):
    def __init__(self, config: TCPComConfig, **overrides):
        super().__init__(config, **overrides)

        self._command_handlers = {}

        self.print("TcpComm object init done", thr=5)

    def close(self):
        sockets_to_close = (
            "radio_control", "radio_data",
            "TCPServerSocketCmd", "TCPServerSocketData",
            "connectionCMD", "connectionData"
        )
        for attr_name in (sockets_to_close):
            sock = getattr(self, attr_name, None)
            if sock is not None:
                with contextlib.suppress(Exception):
                    sock.close()
        self.print("Client object closed", thr=1)

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
        self.print("Client object deleted", thr=1)

    def init_tcp_server(self):
        ## TCP Server
        self.print("Starting TCP server", thr=1)

        ## Command
        self.TCPServerSocketCmd = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_STREAM
        )  # Create a datagram socket
        self.TCPServerSocketCmd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.TCPServerSocketCmd.bind(
            (self.config.tcp_local_ip, self.config.TCP_port_Cmd)
        )  # Bind to address and ip

        ## Data
        self.TCPServerSocketData = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_STREAM
        )  # Create a datagram socket
        self.TCPServerSocketData.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.TCPServerSocketData.bind(
            (self.config.tcp_local_ip, self.config.TCP_port_Data)
        )  # Bind to address and ip

        _ = self.TCPServerSocketData.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        # self.print ("Buffer size [Before]:%d" %bufsize, thr=2)
        self.print("TCP server is up", thr=1)

    def run_tcp_server(self, call_back_func):
        # Listen for incoming connections
        self.TCPServerSocketCmd.listen(1)
        self.TCPServerSocketData.listen(1)

        while True:
            # Wait for a connection
            self.print("\nWaiting for a connection", thr=2)
            self.connectionCMD, addrCMD = self.TCPServerSocketCmd.accept()
            self.connectionData, addrDATA = self.TCPServerSocketData.accept()
            self.connectionData.settimeout(self.config.timeout)
            self.print("\nConnection established", thr=2)

            self.connectionData.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.connectionData.setsockopt(
                socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.config.after_idle_sec
            )
            self.connectionData.setsockopt(
                socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.config.interval_sec
            )
            self.connectionData.setsockopt(
                socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.config.max_fails
            )

            self.connectionCMD.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.connectionCMD.setsockopt(
                socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.config.after_idle_sec
            )
            self.connectionCMD.setsockopt(
                socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.config.interval_sec
            )
            self.connectionCMD.setsockopt(
                socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.config.max_fails
            )

            try:
                while True:
                    try:
                        received_command = self.connectionCMD.recv(self.config.tcp_buffer_size)
                        if received_command:
                            self.print(f"\nClient CMD:{received_command.decode()}", thr=5)
                            responseToCMDinBytes = call_back_func(received_command)
                            self.connectionCMD.sendall(responseToCMDinBytes)
                        else:
                            break
                    except Exception:
                        break
            finally:
                # Clean up the connection
                self.print("\nConnection is closed.", thr=2)
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

    def parse_and_execute(self, received_command):
        client_msg = received_command.decode()
        parts = client_msg.split()

        if not parts:
            response = self.config.invalid_command_message
            return str.encode(f"{response} ({client_msg})")

        cmd, args = parts[0], parts[1:]
        handler = self._command_handlers.get(cmd)

        if handler is None:
            response = self.config.invalid_command_message
        else:
            try:
                response = handler(args)
            except (ValueError, TypeError):
                response = self.config.invalid_number_of_arguments_message
            except Exception as exc:
                self.print(f"Command '{cmd}' failed: {exc}", thr=0)
                response = self.config.invalid_command_message

        return str.encode(f"{response} ({client_msg})")


@dataclass(kw_only=True)
class TCPComRFSoCConfig(TCPComConfig):
    beam_test: tuple = None
    adc_bits: int = 14
    dac_bits: int = 14
    RFFE: str = None

    n_frame_wr: int = 1
    n_frame_rd: int = 2
    n_samples: int = 1024
    n_samples_tx: int = 1024
    n_samples_rx: int = 1024
    n_tx_ant: int = 2
    n_rx_ant: int = 2


class TcpCommRFSoC(TcpComm):
    def __init__(self, config: TCPComRFSoCConfig, **overrides):
        super().__init__(config, **overrides)

        self.obj_rfsoc = None
        self.methods_suffix = "_rfsoc"

        if self.config.RFFE == "sivers":
            self.tx_bb_gain = 0x3
            self.tx_bb_phase = 0x0
            self.tx_bb_iq_gain = 0x77
            self.tx_bfrf_gain = 0x7F
            self.rx_gain_ctrl_bb1 = 0x33
            self.rx_gain_ctrl_bb2 = 0x00
            self.rx_gain_ctrl_bb3 = 0x33
            self.rx_gain_ctrl_bfrf = 0x7F

        self.nread = self.config.n_rx_ant * self.config.n_frame_rd * self.config.n_samples

        # command -> handler
        self._command_handlers.update({
            "receive_samples_once": self._handle_receive_samples_once,
            "receive_samples": self._handle_receive_samples,
            "transmit_samples_default": self._handle_transmit_samples_default,
            "transmit_samples": self._handle_transmit_samples,
            "get_beam_index_tx_sivers": self._handle_get_beam_index_tx_sivers,
            "set_beam_index_tx_sivers": self._handle_set_beam_index_tx_sivers,
            "get_beam_index_rx_sivers": self._handle_get_beam_index_rx_sivers,
            "set_beam_index_rx_sivers": self._handle_set_beam_index_rx_sivers,
            "get_mode_sivers": self._handle_get_mode_sivers,
            "set_mode_sivers": self._handle_set_mode_sivers,
            "get_gain_rx_sivers": self._handle_get_gain_rx_sivers,
            "set_gain_rx_sivers": self._handle_set_gain_rx_sivers,
            "get_gain_tx_sivers": self._handle_get_gain_tx_sivers,
            "set_gain_tx_sivers": self._handle_set_gain_tx_sivers,
            "get_carrier_frequency_sivers": self._handle_get_carrier_frequency_sivers,
            "set_carrier_frequency_sivers": self._handle_set_carrier_frequency_sivers,
            "set_frequency_mixer": self._handle_set_frequency_mixer,
        })

        self.print("TcpCommRFSoC object init done", thr=1)

    def set_mode_sivers(self, mode):
        if mode == "RXen0_TXen1" or mode == "RXen1_TXen0" or mode == "RXen0_TXen0":
            self.radio_control.sendall(b"set_mode_sivers " + str.encode(str(mode)))
            data = self.radio_control.recv(1024)
            self.print(f"Result of set_mode_sivers: {data}", thr=3)
            return data

    def set_frequency_sivers(self, fc):
        self.radio_control.sendall(b"set_carrier_frequency_sivers " + str.encode(str(fc)))
        data = self.radio_control.recv(1024)
        self.print(f"Result of set_frequency_sivers: {data}", thr=3)
        return data

    def set_frequency_mixer_rfsoc(self, f_mixer_dac, f_mixer_adc):
        self.radio_control.sendall(
            b"set_frequency_mixer "
            + str.encode(str(f_mixer_dac) + " ")
            + str.encode(str(f_mixer_adc))
        )
        data = self.radio_control.recv(1024)
        self.print(f"Result of set_frequency_mixer_rfsoc: {data}", thr=3)
        return data

    def set_tx_gain_sivers(self):
        self.radio_control.sendall(
            b"set_gain_tx_sivers "
            + str.encode(str(int(self.tx_bb_gain)) + " ")
            + str.encode(str(int(self.tx_bb_phase)) + " ")
            + str.encode(str(int(self.tx_bb_iq_gain)) + " ")
            + str.encode(str(int(self.tx_bfrf_gain)))
        )
        data = self.radio_control.recv(1024)
        self.print(f"Result of set_tx_gain_sivers: {data}", thr=3)
        return data

    def set_rx_gain_sivers(self):
        self.radio_control.sendall(
            b"set_gain_rx_sivers "
            + str.encode(str(int(self.rx_gain_ctrl_bb1)) + " ")
            + str.encode(str(int(self.rx_gain_ctrl_bb2)) + " ")
            + str.encode(str(int(self.rx_gain_ctrl_bb3)) + " ")
            + str.encode(str(int(self.rx_gain_ctrl_bfrf)))
        )
        data = self.radio_control.recv(1024)
        self.print(f"Result of set_rx_gain_sivers: {data}", thr=3)
        return data

    def transmit_data_default_rfsoc(self):
        self.radio_control.sendall(b"transmit_samples_default")
        data = self.radio_control.recv(1024)
        self.print(f"Result of transmit_data_default_rfsoc: {data}", thr=3)
        return data

    def transmit_data_rfsoc(self, txtd):
        txtd = txtd.copy()
        txtd = np.array(txtd).flatten()
        txtd = txtd * (2 ** (self.config.dac_bits + 1) - 1)
        re = txtd.real.astype(np.int16)
        im = txtd.imag.astype(np.int16)
        txtd = np.concatenate((re, im))

        self.radio_control.sendall(b"transmit_samples")
        self.radio_data.sendall(txtd.tobytes())
        data = self.radio_control.recv(1024)
        self.print(f"Result of transmit_data_rfsoc: {data}", thr=3)
        return data

    def receive_data_rfsoc_once(self, mode="once"):
        if mode == "once":
            nbeams = 1
            self.radio_control.sendall(b"receive_samples_once")
        elif mode == "beams":
            if self.config.beam_test is None:
                raise ValueError("Cannot use 'beams' mode: config.beam_test is None")
            nbeams = len(self.config.beam_test)
            self.radio_control.sendall(b"receive_samples")
        nbytes = nbeams * self.config.nbytes * self.nread * 2
        buf = bytearray()

        while len(buf) < nbytes:
            data = self.radio_data.recv(nbytes - len(buf))
            if not data:
                break # Connection dropped, break to avoid infinite loop
            buf.extend(data)
        data = np.frombuffer(buf, dtype=np.int16)
        data = data / (2 ** (self.config.adc_bits + 1) - 1)
        rxtd = data[: self.nread * nbeams] + 1j * data[self.nread * nbeams :]
        rxtd = rxtd.reshape(nbeams, self.config.n_rx_ant, self.nread // self.config.n_rx_ant)

        resp = self.radio_control.recv(1024)
        self.print(f"Result of receive_data_rfsoc_once: {resp}", thr=3)
        return rxtd

    def receive_data_rfsoc(self, n_rd_rep=1, mode="once", verbose=False):
        rxtd = []
        for i in range(n_rd_rep):
            if verbose:
                self.print(f"Reading iteration: {i + 1}", thr=0)
            rxtd_ = self.receive_data_rfsoc_once(mode=mode)
            rxtd_ = rxtd_.squeeze(axis=0)
            rxtd.append(rxtd_)
        rxtd = np.array(rxtd)
        if len(rxtd.shape) != 3:
            rxtd = rxtd.reshape(n_rd_rep, rxtd.shape[-2], rxtd.shape[-1])
        self.last_rxtd = rxtd.copy()
        return rxtd

    def _handle_receive_samples_once(self, args):
        if len(args) != 0:
            return self.config.invalid_number_of_arguments_message
        iq_data = self.obj_rfsoc.recv_frame_once(n_frame=self.config.n_frame_rd)
        iq_data = np.array(iq_data).flatten()
        iq_data = iq_data * (2 ** (self.config.adc_bits + 1) - 1)
        re = iq_data.real.astype(np.int16)
        im = iq_data.imag.astype(np.int16)
        iq_data = np.concatenate((re, im))
        self.connectionData.sendall(iq_data.tobytes())
        return self.config.success_message

    def _handle_receive_samples(self, args):
        if len(args) != 0:
            return self.config.invalid_number_of_arguments_message
        iq_data = self.obj_rfsoc.recv_frame(n_frame=self.config.n_frame_rd)
        re = iq_data.real.astype(np.int16)
        im = iq_data.imag.astype(np.int16)
        iq_data = np.concatenate((re, im))
        self.connectionData.sendall(iq_data.tobytes())
        return self.config.success_message

    def _handle_transmit_samples_default(self, args):
        if len(args) != 0:
            return self.config.invalid_number_of_arguments_message
        self.obj_rfsoc.send_frame(txtd=self.obj_rfsoc.txtd)
        return self.config.success_message

    def _handle_transmit_samples(self, args):
        if len(args) != 0:
            return self.config.invalid_number_of_arguments_message

        nread = self.config.n_tx_ant * self.config.n_samples_tx
        nbytes = self.config.nbytes * nread * 2
        buf = bytearray()

        while len(buf) < nbytes:
            data = self.connectionData.recv(nbytes - len(buf))
            if not data:
                break # Connection dropped, break to avoid infinite loop
            buf.extend(data)
        data = np.frombuffer(buf, dtype=np.int16)
        data = data / (2 ** (self.config.dac_bits + 1) - 1)
        txtd = data[:nread] + 1j * data[nread:]
        txtd = txtd.reshape(self.config.n_tx_ant, nread // self.config.n_tx_ant)

        self.obj_rfsoc.send_frame(txtd=txtd)
        return self.config.success_message

    def _handle_get_beam_index_tx_sivers(self, args):
        if len(args) != 0:
            return self.config.invalid_number_of_arguments_message
        responseToCMD = str(self.obj_rfsoc.siversControllerObj.get_beam_index_tx())
        return responseToCMD

    def _handle_set_beam_index_tx_sivers(self, args):
        if len(args) != 1:
            return self.config.invalid_number_of_arguments_message
        beamIndex = int(args[0])
        success, status = self.obj_rfsoc.siversControllerObj.set_beam_index_tx(beamIndex)
        responseToCMD = self.config.success_message if success else status
        return responseToCMD

    def _handle_get_beam_index_rx_sivers(self, args):
        if len(args) != 0:
            return self.config.invalid_number_of_arguments_message
        responseToCMD = str(self.obj_rfsoc.siversControllerObj.get_beam_index_rx())
        return responseToCMD

    def _handle_set_beam_index_rx_sivers(self, args):
        if len(args) != 1:
            return self.config.invalid_number_of_arguments_message
        beamIndex = int(args[0])
        success, status = self.obj_rfsoc.siversControllerObj.set_beam_index_rx(beamIndex)
        responseToCMD = self.config.success_message if success else status
        return responseToCMD

    def _handle_get_mode_sivers(self, args):
        if len(args) != 0:
            return self.config.invalid_number_of_arguments_message
        responseToCMD = self.obj_rfsoc.siversControllerObj.get_mode()
        return responseToCMD

    def _handle_set_mode_sivers(self, args):
        if len(args) != 1:
            return self.config.invalid_number_of_arguments_message
        mode = args[0]
        success, status = self.obj_rfsoc.siversControllerObj.set_mode(mode)
        responseToCMD = self.config.success_message if success else status
        return responseToCMD

    def _handle_get_gain_rx_sivers(self, args):
        if len(args) != 0:
            return self.config.invalid_number_of_arguments_message
        (
            rx_gain_ctrl_bb1,
            rx_gain_ctrl_bb2,
            rx_gain_ctrl_bb3,
            rx_gain_ctrl_bfrf,
            agc_int_bfrf_gain_lvl,
            agc_int_bb3_gain_lvl,
        ) = self.obj_rfsoc.siversControllerObj.get_gain_rx()
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
        return responseToCMD

    def _handle_set_gain_rx_sivers(self, args):
        if len(args) != 4:
            return self.config.invalid_number_of_arguments_message
        rx_gain_ctrl_bb1 = int(args[0])
        rx_gain_ctrl_bb2 = int(args[1])
        rx_gain_ctrl_bb3 = int(args[2])
        rx_gain_ctrl_bfrf = int(args[3])

        success, status = self.obj_rfsoc.siversControllerObj.set_gain_rx(
            rx_gain_ctrl_bb1, rx_gain_ctrl_bb2, rx_gain_ctrl_bb3, rx_gain_ctrl_bfrf
        )
        responseToCMD = self.config.success_message if success else status
        return responseToCMD

    def _handle_get_gain_tx_sivers(self, args):
        if len(args) != 0:
            return self.config.invalid_number_of_arguments_message

        tx_bb_gain, tx_bb_phase, tx_bb_iq_gain, tx_bfrf_gain, tx_ctrl = (
            self.obj_rfsoc.siversControllerObj.get_gain_tx()
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
        return responseToCMD

    def _handle_set_gain_tx_sivers(self, args):
        if len(args) != 4:
            return self.config.invalid_number_of_arguments_message

        tx_bb_gain = int(args[0])
        tx_bb_phase = int(args[1])
        tx_bb_iq_gain = int(args[2])
        tx_bfrf_gain = int(args[3])

        success, status = self.obj_rfsoc.siversControllerObj.set_gain_tx(
            tx_bb_gain, tx_bb_phase, tx_bb_iq_gain, tx_bfrf_gain
        )
        responseToCMD = self.config.success_message if success else status
        return responseToCMD

    def _handle_get_carrier_frequency_sivers(self, args):
        if len(args) != 0:
            return self.config.invalid_number_of_arguments_message
        responseToCMD = str(self.obj_rfsoc.siversControllerObj.get_frequency())
        return responseToCMD

    def _handle_set_carrier_frequency_sivers(self, args):
        if len(args) != 1:
            return self.config.invalid_number_of_arguments_message
        fc = float(args[0])
        success, status = self.obj_rfsoc.siversControllerObj.set_frequency(fc)
        responseToCMD = self.config.success_message if success else status
        return responseToCMD

    def _handle_set_frequency_mixer(self, args):
        if len(args) != 2:
            return self.config.invalid_number_of_arguments_message

        f_mixer_dac = float(args[0])
        f_mixer_adc = float(args[1])
        success = self.obj_rfsoc.set_dac_mixer(mix_freq=f_mixer_dac, do_rfsoc_mixer_settings=True)
        success &= self.obj_rfsoc.set_adc_mixer(mix_freq=f_mixer_adc, do_rfsoc_mixer_settings=True)
        responseToCMD = (
            self.config.success_message if success else "Failed to set mixer frequencies"
        )
        return responseToCMD


@dataclass(kw_only=True)
class TCPComLinTrackConfig(TCPComConfig):
    pass


class TcpCommLinTrack(TcpComm):
    def __init__(self, config: TCPComLinTrackConfig, **overrides):
        super().__init__(config, **overrides)
        self.obj_lintrack = None
        self.methods_suffix = "_lintrack"

        # command -> handler
        self._command_handlers.update({
            "move": self._handle_move,
            "return2home": self._handle_return2home,
            "go2end": self._handle_go2end,
        })

        self.print("TcpCommLinTrack object init done", thr=1)

    def move(self, lintrack_id=0, distance=0.0):
        self.print(f"Moving linear track {lintrack_id} by {distance} mm", thr=2)
        self.radio_control.sendall(
            b"move " + str.encode(str(lintrack_id) + " ") + str.encode(str(distance))
        )
        data = self.radio_control.recv(1024)
        self.print(f"Result of move_forward: {data}", thr=3)
        return data

    def go2pos(self, lintrack_id=0, position=0.0):
        self.print(f"Moving linear track {lintrack_id} to position {position} mm", thr=2)
        self.radio_control.sendall(
            b"go2pos " + str.encode(str(lintrack_id) + " ") + str.encode(str(position))
        )
        data = self.radio_control.recv(1024)
        self.print(f"Result of go2pos: {data}", thr=3)
        return data

    def return2home(self, lintrack_id=0):
        self.print(f"Returning linear track {lintrack_id} to home", thr=2)
        self.radio_control.sendall(b"return2home " + str.encode(str(lintrack_id)))
        data = self.radio_control.recv(1024)
        self.print(f"Result of return2home: {data}", thr=3)
        return data

    def go2end(self, lintrack_id=0):
        self.print(f"Going to the end of line on linear track {lintrack_id}", thr=2)
        self.radio_control.sendall(b"go2end " + str.encode(str(lintrack_id)))
        data = self.radio_control.recv(1024)
        self.print(f"Result of go2end: {data}", thr=3)
        return data

    def _handle_move(self, args):
        if len(args) != 2:
            return self.config.invalid_number_of_arguments_message
        motor_id = int(args[0])
        distance = float(args[1])
        success, status = self.obj_lintrack.displace(motor_id=motor_id, dis=distance)
        return self.config.success_message if success else status

    def _handle_go2pos(self, args):
        if len(args) != 2:
            return self.config.invalid_number_of_arguments_message
        motor_id = int(args[0])
        position = float(args[1])
        success, status = self.obj_lintrack.go2pos(motor_id=motor_id, pos=position)
        return self.config.success_message if success else status

    def _handle_return2home(self, args):
        if len(args) != 1:
            return self.config.invalid_number_of_arguments_message
        motor_id = int(args[0])
        success, status = self.obj_lintrack.return2home(motor_id=motor_id)
        return self.config.success_message if success else status

    def _handle_go2end(self, args):
        if len(args) != 1:
            return self.config.invalid_number_of_arguments_message
        motor_id = int(args[0])
        success, status = self.obj_lintrack.go2end(motor_id=motor_id)
        return self.config.success_message if success else status


@dataclass(kw_only=True)
class TCPComControllerConfig(TCPComRFSoCConfig, TCPComLinTrackConfig):
    pass


class TcpCommController(TcpCommRFSoC, TcpCommLinTrack):
    def __init__(self, config: TCPComControllerConfig, **overrides):
        super().__init__(config, **overrides)
        self.methods_suffix = "_controller"

        self.obj_piradio = None
        self.obj_gimbal = None

        self._command_handlers.update({
            "set_frequency_piradio": self._handle_set_frequency_piradio,
            "set_gain_piradio": self._handle_set_gain_piradio,
            "set_bias_piradio": self._handle_set_bias_piradio,
            "set_gimbal_deg_az": self._handle_set_gimbal_deg_az,
            "set_gimbal_deg_el": self._handle_set_gimbal_deg_el,
        })

        self.print("TcpCommController object init done", thr=1)

    def set_frequency_piradio(self, fc=6.0e9, lo="high"):
        self.print(f"Setting frequency to {fc / 1e9} GHz", thr=3)
        self.radio_control.sendall(
            b"set_frequency_piradio " + str.encode(str(fc) + " ") + str.encode(str(lo))
        )
        data = self.radio_control.recv(1024)
        self.print(f"Result of set_frequency_piradio: {data}", thr=3)
        return data

    def set_gain_piradio(self, trx="tx", chan=0, gain_db=0):
        self.print(f"Setting gain to {gain_db} dB for {trx}-{chan}", thr=3)
        self.radio_control.sendall(
            b"set_gain_piradio "
            + str.encode(str(trx) + " ")
            + str.encode(str(chan) + " ")
            + str.encode(str(gain_db))
        )
        data = self.radio_control.recv(1024)
        self.print(f"Result of set_gain_piradio: {data}", thr=3)
        return data

    def set_bias_piradio(self, chan, iq="I", bias_voltage=0):
        self.print(f"Setting bias to {bias_voltage} V for tx-{chan}-{iq}", thr=3)
        self.radio_control.sendall(
            b"set_bias_piradio "
            + str.encode(str(chan) + " ")
            + str.encode(str(iq) + " ")
            + str.encode(str(bias_voltage))
        )
        data = self.radio_control.recv(1024)
        self.print(f"Result of set_bias_piradio: {data}", thr=3)
        return data

    def set_gimbal_deg_az(self, angle_deg):
        self.print(f"Setting gimbal azimuth to {angle_deg} degrees", thr=3)
        self.radio_control.sendall(b"set_gimbal_deg_az " + str.encode(str(angle_deg)))
        data = self.radio_control.recv(1024)
        self.print(f"Result of set_gimbal_deg_az: {data}", thr=3)
        return data

    def set_gimbal_deg_el(self, angle_deg):
        self.print(f"Setting gimbal elevation to {angle_deg} degrees", thr=3)
        self.radio_control.sendall(b"set_gimbal_deg_el " + str.encode(str(angle_deg)))
        data = self.radio_control.recv(1024)
        self.print(f"Result of set_gimbal_deg_el: {data}", thr=3)
        return data

    def _handle_set_frequency_piradio(self, args):
        if len(args) != 2:
            return self.config.invalid_number_of_arguments_message
        freq = float(args[0])
        lo = args[1]
        result, response = self.obj_piradio.set_frequency_piradio(freq, lo=lo)
        responseToCMD = self.config.success_message
        return responseToCMD

    def _handle_set_gain_piradio(self, args):
        if len(args) != 3:
            return self.config.invalid_number_of_arguments_message
        trx = args[0]
        chan = int(args[1])
        gain_db = float(args[2])
        result, response = self.obj_piradio.set_gain_piradio(trx=trx, chan=chan, gain_db=gain_db)
        responseToCMD = self.config.success_message
        return responseToCMD

    def _handle_set_bias_piradio(self, args):
        if len(args) != 3:
            return self.config.invalid_number_of_arguments_message
        chan = int(args[0])
        iq = args[1]
        bias_voltage = float(args[2])
        result, response = self.obj_piradio.set_bias_piradio(
            chan=chan, iq=iq, bias_voltage=bias_voltage
        )
        responseToCMD = self.config.success_message
        return responseToCMD

    def _handle_set_gimbal_deg_az(self, args):
        if len(args) != 1:
            return self.config.invalid_number_of_arguments_message
        angle_deg = float(args[0])
        current_deg = self.obj_gimbal.get_deg()
        self.obj_gimbal.set_deg([angle_deg, current_deg[1]])

    def _handle_set_gimbal_deg_el(self, args):
        if len(args) != 1:
            return self.config.invalid_number_of_arguments_message
        angle_deg = float(args[0])
        current_deg = self.obj_gimbal.get_deg()
        self.obj_gimbal.set_deg([current_deg[0], angle_deg])


@dataclass(kw_only=True)
class SshComConfig(GeneralConfig):
    host_ip: str = "0.0.0.0"
    port: int = 22
    username: str = "root"
    password: str = " root"


class SshCom(General):
    def __init__(self, config: SshComConfig, **overrides):
        super().__init__(config, **overrides)

        self.print("SshCom object init done", thr=1)

    def init_ssh_client(self):
        try:
            # Initialize SSH client
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect to the remote server
            self.client.connect(
                hostname=self.config.host_ip,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                look_for_keys=False,
                allow_agent=False,
            )

        except paramiko.AuthenticationException:
            self.print("Authentication failed. Please check your credentials.", thr=0)
            traceback.print_exc()
        except paramiko.SSHException as e:
            self.print(f"SSH Error: {e}", thr=0)
        except Exception as e:
            self.print(f"Unexpected error: {e}", thr=0)

        self.print("SshCom client init done", thr=1)

    def close(self):
        self.client.close()
        self.print("SSH Client object closed", thr=1)

    def __del__(self):
        self.close()
        self.print("SSH Client object deleted", thr=1)

    def exec_command(self, command, verif_keyword=""):
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


@dataclass(kw_only=True)
class ScpComConfig(SshComConfig):
    pass


class ScpCom(SshCom):
    def __init__(self, config: ScpComConfig, **overrides):
        super().__init__(config, **overrides)

        self.init_ssh_client()
        self.scp_clinet = SCPClient(self.client.get_transport())
        self.print("ScpCom object init done", thr=1)

    # SCP files from the remote host
    def download_files(self, remote_files, local_dir):
        try:
            for remote_file in remote_files:
                try:
                    self.scp_clinet.get(
                        remote_file,
                        local_path=os.path.join(local_dir, os.path.basename(remote_file)),
                    )
                except Exception:
                    self.print(f"Failed to download {remote_file}", thr=0)
            self.print("Files downloaded successfully!", thr=3)
        except Exception:
            self.print("Files download failed!", thr=0)

    def download_files_with_pattern(self, remote_base_dir, remote_patterns, local_base_dir):
        try:
            for pattern in remote_patterns:
                pattern_ = os.path.join(remote_base_dir, pattern)
                remote_files = self.client.exec_command(f"ls {pattern_}")[1].read().decode().split()
                for remote_file in remote_files:
                    remote_file = (
                        os.path.join(remote_base_dir, remote_file)
                        if not os.path.isabs(remote_file)
                        else remote_file
                    )
                    relative_path = os.path.relpath(remote_file, remote_base_dir)
                    local_path = os.path.join(local_base_dir, relative_path)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    try:
                        self.scp_clinet.get(remote_file, local_path=local_path)
                    except Exception:
                        self.print(f"Failed to download {remote_file}", thr=0)
            self.print(f"Files at {remote_patterns} downloaded successfully!", thr=3)
        except Exception:
            self.print("Files download failed!", thr=0)

    def close(self):
        self.scp_clinet.close()
        self.client.close()
        self.print("SCP Client object closed", thr=1)


@dataclass(kw_only=True)
class RestComConfig(GeneralConfig):
    ip_address: str = "0.0.0.0"
    port: int = 5000
    protocol: str = "http"
    timeout: float = 5.0


class RESTCom(General):
    def __init__(self, config: RestComConfig, **overrides):
        super().__init__(config, **overrides)

        self._s = requests.Session()
        self.url_base = f"{self.config.protocol}://{self.config.ip_address}:{self.config.port}/"

        self.print("RESTCom object init done", thr=1)

    def init_rest_client(self):
        self.print("RESTCom client init done", thr=1)

    def close(self):
        self.print("RESTCom object closed", thr=1)

    def __del__(self):
        self.close()
        self.print("RESTCom object deleted", thr=1)

    def call_rest_api(self, url, params=None, verif_keyword=None):
        try:
            # response = requests.get(url, timeout=self.config.timeout)
            if params is None:
                response = self._s.get(url, timeout=self.config.timeout)
            else:
                response = self._s.get(url, params=params, timeout=self.config.timeout)

            response.raise_for_status()  # Raise an HTTPError for bad responses
            self.print(f"Successfully called the REST API:{response.json()}", thr=3)

            # response =  response.json()
            response = json.loads(response.text)
            if isinstance(response, (int, float)):
                response = str(response)
        except requests.exceptions.RequestException as e:
            self.print(f"Error executing REST API: {e}", thr=0)
            response = ""

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


@dataclass(kw_only=True)
class RestComPiradioConfig(RestComConfig):
    port: int = 5111
    freq_sw_dly: float = 0.1
    gain_sw_dly: float = 0.1
    bias_sw_dly: float = 0.1


class RESTComPiradio(RESTCom):
    def __init__(self, config: RestComPiradioConfig, **overrides):
        super().__init__(config, **overrides)

        self.methods_suffix = "_piradio"
        self.print("RESTComPiradio object init done", thr=1)

    def initialize(self, verif_keyword="done"):
        self.print("Pi-Radio REST Comm Initialization done", thr=3)

    def set_frequency(self, fc=6.0e9, lo="high", verif_keyword=None):
        # command = f'high_lo?freq={fc}'
        url = f"{self.url_base}{lo}_lo"
        params = {"freq": fc}
        result, response = self.call_rest_api(url, params=params, verif_keyword=verif_keyword)
        result = False if response == "" else float(response["frequency"]) == fc
        if result:
            time.sleep(self.config.freq_sw_dly)
            self.print(f"Frequency set to {fc / 1e9} GHz", thr=3)
        else:
            self.print(f"Failed to set frequency to {fc / 1e9} GHz", thr=0)
        return result, response

    def set_gain(self, trx="tx", chan=0, gain_db=0, verif_keyword=None):
        chan = str(chan)
        url = self.url_base + "gain"
        params = {"trx": trx, "chan": chan, "v": gain_db}
        result, response = self.call_rest_api(url, params=params, verif_keyword=verif_keyword)
        result = False if response == "" else float(response[trx][chan]) == gain_db
        if result:
            time.sleep(self.config.gain_sw_dly)
            self.print(f"Gain set to {gain_db} dB", thr=3)
        else:
            self.print(f"Failed to set gain to {gain_db} dB", thr=0)
        return result, response

    def set_bias(self, chan=0, iq="I", bias_voltage=0, verif_keyword=None):
        chan = str(chan)
        url = self.url_base + "bias"
        params = {"iq": iq, "chan": chan, "v": bias_voltage}
        result, response = self.call_rest_api(url, params=params, verif_keyword=verif_keyword)
        result = False if response == "" else float(response[chan][iq]) == bias_voltage
        if result:
            time.sleep(self.config.bias_sw_dly)
            self.print(f"Bias voltage set to {bias_voltage} V", thr=3)
        else:
            self.print(f"Failed to set bias voltage to {bias_voltage} V", thr=0)
        return result, response
