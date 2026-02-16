import time
from dataclasses import dataclass

import serial
import serial.tools.list_ports

from sigcom_toolkit.general import General, GeneralConfig


@dataclass
class SerialComConfig(GeneralConfig):
    port: str = "COM6"
    baudrate: int = 115200
    timeout: float = 1.0


class SerialComm(General):
    def __init__(self, config: SerialComConfig, **overrides):
        """
        Initialize the connection to the Target.

        :param port: The COM port to connect to (default: 'COM6').
        :param baudrate: The communication baud rate (default: 115200).
        :param timeout: Read timeout in seconds (default: 1).
        """

        super().__init__(config, **overrides)

        self.client = None
        self.print("SerialComm Client object created", thr=1)

    def close(self):
        """Close the connection to the Target."""
        if self.client and self.client.is_open:
            self.client.close()
            self.print("Client serial closed", thr=1)

    def __del__(self):
        self.close()
        self.print("Client object deleted", thr=1)

    def connect(self):
        """Establish a connection to the target."""
        self.client = serial.Serial(
            port=self.config.port, baudrate=self.config.baudrate, timeout=self.config.timeout
        )
        time.sleep(1)  # Wait for target to reset
        if self.client.is_open:
            self.print("Client serial connected!", thr=1)
        else:
            self.print("Client serial connection failed.", thr=0)

    def list_ports(self):
        """
        List all available COM ports.
        """
        ports = serial.tools.list_ports.comports()
        for port, desc, hwid in ports:
            self.print(f"{port}: {desc} [{hwid}]", thr=0)

    def write(self, data):
        """
        Send data to the Target.

        :param data: The string data to send.
        """
        if self.client and self.client.is_open:
            self.client.write(data.encode())  # Convert string to bytes
            self.print("Finished writing to the Serial target", thr=5)

    def read_lines(self, max_lines=None, termination_signal="END"):
        """
        Read multiple lines of data from the Target.

        :param max_lines: Maximum number of lines to read (optional).
        :param termination_signal: A specific message that signals the end of the response.
        :return: A list of lines read from the Target.
        """
        responses = []
        lines_read = 0
        while True:
            if self.client.in_waiting > 0:  # Check if there is data to read
                line = self.client.readline().decode("utf-8").strip()  # Decode bytes to string
                responses.append(line)
                self.print(f"Target: {line}", thr=5)  # Debugging: print to console

                lines_read += 1
                if max_lines and lines_read >= max_lines:
                    break
                if line == termination_signal:  # Stop if termination signal is received
                    break
        self.print("Finished reading from the Serial target", thr=5)
        return responses


@dataclass
class SerialComTurnTableConfig(SerialComConfig):
    rotation_delay: float = 0.0


class SerialCommTurnTable(SerialComm):
    def __init__(self, config: SerialComTurnTableConfig, **overrides):
        """
        Initialize the connection to the Arduino.

        :param port: The COM port to connect to (default: 'COM6').
        :param baudrate: The communication baud rate (default: 115200).
        :param timeout: Read timeout in seconds (default: 1).
        """
        super().__init__(config, **overrides)

        self.position = 0.0
        self.print("SerialCommTurnTable Client object created", thr=1)

        # self.connect()

    def return2home(self):
        self.print("Starting turn-table homing procedure..", thr=2)
        self.move_to_position(position=0.0)
        self.print("turn-table homing procedure done.", thr=2)

    def move_to_position(self, position):
        self.print(f"Moving turn-table to position: {position}", thr=2)
        command = "moveToAngle=" + str(position)
        self.write(command)
        responses = self.read_lines(max_lines=1)
        # block until position is reached:
        isReady = False
        while not isReady:
            if responses[-1] == "done.":
                isReady = True
            else:
                self.print("waiting..", thr=3)
                time.sleep(0.1)
                responses = self.read_lines(max_lines=1)
        self.position = position
        if self.config.rotation_delay > 0.0:
            time.sleep(self.config.rotation_delay)
        self.print(f"Turn-table moved to position: {position}", thr=3)

    def set_home(self):
        self.print("Setting the current position as the home position", thr=2)
        command = "home"
        self.write(command)
        _ = self.read_lines(max_lines=1)
        self.position = 0.0
        self.print("Home position set", thr=3)

    def calibrate(self, mode="start"):
        self.print(f"Calibrating the turn-table with mode {mode}", thr=1)
        self.print("Try to set the angle at zero ...", thr=1)
        while True:
            angle_str = input(
                "Enter the angle to move in deg, empty if need to finish calibration: "
            )
            if angle_str == "":
                # if mode == 'start':
                #     self.position = 0.0
                # elif mode == 'end':
                #     self.position = 360.0
                self.set_home()
                break
            try:
                angle = float(angle_str)
            except Exception:
                self.print("Invalid angle, please enter a valid angle", thr=0)
                continue
            self.move_to_position(position=angle)

        self.print("Calibration for turn-table complete", thr=1)

    def interactive_move(self):
        self.print("Starting interactive move for TurnTable", thr=1)
        while True:
            angle_str = input("Enter the angle to move in degrees, empty if need to break: ")
            if angle_str == "":
                break
            try:
                angle = float(angle_str)
            except Exception:
                self.print("Invalid angle, please enter a valid angle", thr=0)
                continue
            self.move_to_position(position=angle)
