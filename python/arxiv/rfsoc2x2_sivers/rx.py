import time
import socket
import numpy as np
import matplotlib.pyplot as plt


fc = 57.51e9

ip_address = "10.1.1.30"
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


def set_rx_gain():
    rx_gain_ctrl_bb1 = 0x33
    rx_gain_ctrl_bb2 = 0x00
    rx_gain_ctrl_bb3 = 0x33
    rx_gain_ctrl_bfrf = 0x7F
    radio_control.sendall(
        b"setGainRX "
        + str.encode(str(int(rx_gain_ctrl_bb1)) + " ")
        + str.encode(str(int(rx_gain_ctrl_bb2)) + " ")
        + str.encode(str(int(rx_gain_ctrl_bb3)) + " ")
        + str.encode(str(int(rx_gain_ctrl_bfrf)))
    )
    data = radio_control.recv(1024)
    print(data)
    return data


def receive_data():
    nbeams = 17
    nbytes = 2
    nread = 1024
    radio_control.sendall(b"receiveSamples")
    nbytes = nbeams * nbytes * nread * 2
    buf = bytearray()

    while len(buf) < nbytes:
        data = radio_data.recv(nbytes)
        buf.extend(data)
    data = np.frombuffer(buf, dtype=np.int16)
    rxtd = data[: nread * nbeams] + 1j * data[nread * nbeams :]
    rxtd = rxtd.reshape(nbeams, nread)
    return rxtd


radio_control = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
radio_control.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
radio_control.connect((ip_address, radio_control_port))


radio_data = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
radio_data.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
radio_data.connect((ip_address, radio_data_port))


set_mode("RXen1_TXen0")
set_frequency(fc)
set_rx_gain()

hest = receive_data()

for i in range(17):
    mag = np.abs(hest[i, :])
    mag = mag / np.max(mag)
    mag_db = 20 * np.log10(mag + 1e-17)
    plt.plot(mag_db)
    plt.ylabel("Magnitude, dB")
    plt.title(f"Beam index {i}")
    plt.grid(0.4)
    plt.show()
