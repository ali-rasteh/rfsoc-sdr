import time
import socket

fc = 57.51e9

ip_address = "127.0.0.1"
radio_control_port = 8080
radio_data_port = 8081


def set_mode(mode):
    if mode == "RXen0_TXen1" or mode == "RXen1_TXen0" or mode == "RXen0_TXen0":
        radio_control.sendall(b"setModeSiver " + str.encode(str(mode)))
        data = radio_control.recv(1024)
        print(data)
        return data


def set_frequency(fc):
    radio_control.sendall(b"setCarrierFrequency " + str.encode(str(fc)))
    data = radio_control.recv(1024)
    print(data)
    return data


def set_tx_gain():
    tx_bb_gain = 0x3
    tx_bb_phase = 0x0
    tx_bb_iq_gain = 0x77
    tx_bfrf_gain = 0x7F

    radio_control.sendall(
        b"setGainTX "
        + str.encode(str(int(tx_bb_gain)) + " ")
        + str.encode(str(int(tx_bb_phase)) + " ")
        + str.encode(str(int(tx_bb_iq_gain)) + " ")
        + str.encode(str(int(tx_bfrf_gain)))
    )
    data = radio_control.recv(1024)
    print(data)
    return data


def transmit_data():
    radio_control.sendall(b"transmitSamples")
    data = radio_control.recv(1024)
    print(data)
    return data


radio_control = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
radio_control.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
radio_control.connect((ip_address, radio_control_port))


radio_data = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
radio_data.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
radio_data.connect((ip_address, radio_data_port))

transmit_data()

# set_mode('RXen0_TXen1')
# set_frequency(fc)
# set_tx_gain()
