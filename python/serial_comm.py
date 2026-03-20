from __future__ import annotations

import contextlib
import math
import re
import time
from dataclasses import dataclass

import serial
import serial.tools.list_ports
from sigcom_toolkit.general import General, GeneralConfig


@dataclass(kw_only=True)
class SerialComConfig(GeneralConfig):
    port: str = "COM6"
    baudrate: int = 115200
    timeout: float = 1.0
    write_timeout: float | None = 0.0
    bytesize: int | None = None
    parity: str | None = None
    stopbits: int | None = None


class SerialCom(General):
    def __init__(self, config: SerialComConfig, **overrides):
        """
        Initialize the connection to the Target.

        :param port: The COM port to connect to (default: 'COM6').
        :param baudrate: The communication baud rate (default: 115200).
        :param timeout: Read timeout in seconds (default: 1).
        """

        super().__init__(config, **overrides)

        self.client = None
        self.print("SerialCom Client object created", thr=1)

    def close(self):
        """Close the connection to the Target."""
        if self.client and self.client.is_open:
            self.client.close()
            self.client = None
            self.print("Client serial closed", thr=1)

    def __del__(self):
        self.close()
        self.print("Client object deleted", thr=1)

    def connect(self):
        """Establish a connection to the target."""
        kwargs = {
            "port": self.config.port,
            "baudrate": self.config.baudrate,
            "timeout": self.config.timeout,
            "write_timeout": self.config.write_timeout,
            "bytesize": self.config.bytesize,
            "parity": self.config.parity,
            "stopbits": self.config.stopbits,
        }
        self.client = serial.Serial(**{k: v for k, v in kwargs.items() if v is not None})
        try:
            self.client.reset_input_buffer()
            self.client.reset_output_buffer()
            time.sleep(0.1)  # Wait for target to reset
        except Exception:
            pass
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


@dataclass(kw_only=True)
class SerialComTurnTableConfig(SerialComConfig):
    rotation_delay: float = 0.0


class SerialComTurnTable(SerialCom):
    def __init__(self, config: SerialComTurnTableConfig, **overrides):
        """
        Initialize the connection to the Arduino.

        :param port: The COM port to connect to (default: 'COM6').
        :param baudrate: The communication baud rate (default: 115200).
        :param timeout: Read timeout in seconds (default: 1).
        """
        super().__init__(config, **overrides)

        self.methods_suffix_list = ["_turntable"]
        self.position = 0.0
        self.print("SerialComTurnTable Client object created", thr=1)

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


@dataclass(kw_only=True)
class SerialComD48PTUConfig(SerialComConfig):
    baudrate: int = 9600
    timeout: float = 0.7
    write_timeout: float = 0.7

    arcsec_per_pos: float = 92.571429  # from your device
    line_ending: str = "\r"
    # Optional soft limits (in degrees) for safety
    pan_min_deg: float | None = -160.0
    pan_max_deg: float | None = 160.0
    tilt_min_deg: float | None = -35.1
    tilt_max_deg: float | None = 35.1


class SerialComD48PTU(SerialCom):
    """
    Public API:
        - get_deg() / get_rad() -> [azimuth, elevation]
        - set_deg([az, el]) / set_rad([az, el])
        - goto_deg(az=..., el=...) / goto_rad(az=..., el=...)
        - stop()
        - refresh_scale_from_device()
    """

    def __init__(self, config: SerialComD48PTUConfig, **overrides):
        super().__init__(config, **overrides)
        self.methods_suffix_list = ["_d48ptu"]

    # ----------------- serial I/O -----------------
    def _write_line(self, line: str) -> None:
        assert self.client is not None
        payload = (line.strip() + self.config.line_ending).encode("ascii", errors="ignore")
        self.client.write(payload)
        self.client.flush()

    def _read_all(self, max_wait: float = 0.4) -> str:
        """Read whatever comes back within a short window."""
        assert self.client is not None
        t0 = time.time()
        chunks = []
        while time.time() - t0 < max_wait:
            n = getattr(self.client, "in_waiting", 0)
            if n:
                chunks.append(self.client.read(n).decode("ascii", errors="ignore"))
                t0 = time.time()  # extend to catch trailing lines
            else:
                time.sleep(0.01)
        return "".join(chunks).strip()

    def cmd(self, line: str, read: bool = True) -> str:
        if not self.client or not self.client.is_open:
            self.connect()
        with contextlib.suppress(Exception):
            self.client.reset_input_buffer()
        self._write_line(line)
        return self._read_all() if read else ""

    # ----------------- conversion -----------------
    def deg_to_pos(self, deg: float) -> int:
        # 1 degree = 3600 arcsec
        pos = deg * 3600.0 / float(self.config.arcsec_per_pos)
        return int(round(pos))

    def pos_to_deg(self, pos: int) -> float:
        return float(pos) * float(self.config.arcsec_per_pos) / 3600.0

    def rad_to_pos(self, rad: float) -> int:
        return self.deg_to_pos(math.degrees(rad))

    def pos_to_rad(self, pos: int) -> float:
        return math.radians(self.pos_to_deg(pos))

    def _clamp_deg(self, axis: str, deg: float) -> float:
        if axis == "pan":
            mn, mx = self.config.pan_min_deg, self.config.pan_max_deg
        else:
            mn, mx = self.config.tilt_min_deg, self.config.tilt_max_deg
        if mn is not None and deg < mn:
            return mn
        if mx is not None and deg > mx:
            return mx
        return deg

    # ----------------- parsing -----------------
    _re_cur_pan = re.compile(r"Current\s+Pan\s+position\s+is\s+(-?\d+)", re.IGNORECASE)
    _re_cur_tilt = re.compile(r"Current\s+Tilt\s+position\s+is\s+(-?\d+)", re.IGNORECASE)
    _re_scale = re.compile(
        r"([0-9]+(?:\.[0-9]+)?)\s*seconds?\s*arc\s*per\s*position", re.IGNORECASE
    )

    def refresh_scale_from_device(self) -> float:
        out = self.cmd("PR")
        m = self._re_scale.search(out)
        if not m:
            raise RuntimeError(f"Could not parse scale from response:\n{out}")
        self.config.arcsec_per_pos = float(m.group(1))
        return self.config.arcsec_per_pos

    # ----------------- axis getters/setters (positions) -----------------
    def _get_pan_pos(self) -> int:
        out = self.cmd("pp")
        m = self._re_cur_pan.search(out)
        if not m:
            raise RuntimeError(f"Could not parse pan position from:\n{out}")
        return int(m.group(1))

    def _get_tilt_pos(self) -> int:
        out = self.cmd("tp")
        m = self._re_cur_tilt.search(out)
        if not m:
            raise RuntimeError(f"Could not parse tilt position from:\n{out}")
        return int(m.group(1))

    def _set_pan_pos(self, pos: int) -> str:
        return self.cmd(f"PP{pos}")

    def _set_tilt_pos(self, pos: int) -> str:
        return self.cmd(f"TP{pos}")

    # ----------------- unified angle API -----------------
    def get_deg(self) -> list[float]:
        """Return [azimuth_deg, elevation_deg]."""
        pan_pos = self._get_pan_pos()
        tilt_pos = self._get_tilt_pos()
        return [-self.pos_to_deg(pan_pos), self.pos_to_deg(tilt_pos)]

    def get_rad(self) -> list[float]:
        """Return [azimuth_rad, elevation_rad]."""
        pan_pos = self._get_pan_pos()
        tilt_pos = self._get_tilt_pos()
        return [-self.pos_to_rad(pan_pos), self.pos_to_rad(tilt_pos)]

    def set_deg(self, angles_deg: list[float]) -> None:
        """
        angles_deg = [azimuth_deg, elevation_deg]
        Sets target positions (non-blocking).
        """
        if len(angles_deg) != 2:
            raise ValueError("set_deg expects [azimuth_deg, elevation_deg]")
        az_deg, el_deg = float(-angles_deg[0]), float(angles_deg[1])

        az_deg = self._clamp_deg("pan", az_deg)
        el_deg = self._clamp_deg("tilt", el_deg)

        self._set_pan_pos(self.deg_to_pos(az_deg))
        self._set_tilt_pos(self.deg_to_pos(el_deg))

    def set_rad(self, angles_rad: list[float]) -> None:
        """
        angles_rad = [azimuth_rad, elevation_rad]
        Sets target positions (non-blocking).
        """
        if len(angles_rad) != 2:
            raise ValueError("set_rad expects [azimuth_rad, elevation_rad]")
        az_rad, el_rad = float(-angles_rad[0]), float(angles_rad[1])

        az_deg = math.degrees(az_rad)
        el_deg = math.degrees(el_rad)

        az_deg = self._clamp_deg("pan", az_deg)
        el_deg = self._clamp_deg("tilt", el_deg)

        self._set_pan_pos(self.deg_to_pos(az_deg))
        self._set_tilt_pos(self.deg_to_pos(el_deg))

    def goto_deg(
        self, azimuth_deg: float | None = None, elevation_deg: float | None = None
    ) -> None:
        """Convenience: set only one axis if you want."""
        if azimuth_deg is not None:
            azimuth_deg = -azimuth_deg
            az = self._clamp_deg("pan", float(azimuth_deg))
            self._set_pan_pos(self.deg_to_pos(az))
        if elevation_deg is not None:
            el = self._clamp_deg("tilt", float(elevation_deg))
            self._set_tilt_pos(self.deg_to_pos(el))

    def goto_rad(
        self, azimuth_rad: float | None = None, elevation_rad: float | None = None
    ) -> None:
        """Convenience: set only one axis if you want."""
        if azimuth_rad is not None:
            azimuth_rad =- azimuth_rad
            az_deg = self._clamp_deg("pan", math.degrees(float(azimuth_rad)))
            self._set_pan_pos(self.deg_to_pos(az_deg))
        if elevation_rad is not None:
            el_deg = self._clamp_deg("tilt", math.degrees(float(elevation_rad)))
            self._set_tilt_pos(self.deg_to_pos(el_deg))

    def stop(self) -> str:
        return self.cmd("ST")
