import atexit
import os
import time

try:
    import board  # type: ignore
    from adafruit_motor import stepper  # type: ignore
    from adafruit_motorkit import MotorKit  # type: ignore
except ImportError:
    print("Adafruit libraries not found.")
from sigcom_toolkit.general import General, GeneralConfig
from tcp_comm import TCPComLinTrackConfig, TcpCommLinTrack


class LinearTrackControllerConfig(GeneralConfig):
    output_mode: str = "dc"  # "stepper" or "dc"
    dis_per_rev: float = 8.0  # distance per revolution in mm
    pulse_per_rev: int = 400  # number of pulses per revolution
    pulse_freq: int = 1600  # frequency of the pulse in Hz
    dis_coeff: float = 0.972  # coefficient to convert the distance to time
    overhead_time: float = 0.0018 + 0.0061 + 0.0001  # overhead time for the motor to start and stop in seconds
    position_file_path: str = os.path.join(os.getcwd(), "lintrack_position.txt")  # file path to store the position of the linear track
    n_motors: int = 2  # number of motors to control

    total_length = 1500  # length of the linear track in mm
    plate_length = 125
    margin2edge = 5

    def __post_init__(self):
        super().__post_init__()
        self.travel_length = self.total_length - self.plate_length
        self.travel_length -= 2 * self.margin2edge


class LinearTrackController(General):
    def __init__(self, config: LinearTrackControllerConfig, **overrides):
        super().__init__(config, **overrides)

        if self.config.output_mode == "stepper":
            self.kit = stepper.StepperMotor(microsteps=2)
        elif self.config.output_mode == "dc":
            self.kit = MotorKit(i2c=board.I2C(), pwm_frequency=self.config.pulse_freq)

        self.pulse_pwm_1 = self.kit.motor1
        self.pulse_pwm_2 = self.kit.motor2
        self.pulse_pwm = [self.pulse_pwm_1, self.pulse_pwm_2]
        self.direction_out_1 = self.kit.motor3
        self.direction_out_2 = self.kit.motor4
        self.direction_out = [self.direction_out_1, self.direction_out_2]

        self.reset()
        self.position = self.read_position()

        config = TCPComLinTrackConfig().update_from_config(self.config)
        self.tcp_comm = TcpCommLinTrack(config)
        self.tcp_comm.init_tcp_server()

    def run_tcp(self):
        self.print("Running TCP server", thr=1)
        self.tcp_comm.obj_lintrack = self
        self.tcp_comm.run_tcp_server(self.tcp_comm.parse_and_execute)

    def calibrate(self, motor_id=0, mode="start"):
        self.print(f"Calibrating the linear track {motor_id} with mode {mode}", thr=1)
        while True:
            dis_str = input("Enter the distance to move in mm, empty if need to break: ")
            if dis_str == "":
                if mode == "start":
                    self.position[motor_id] = 0.0
                elif mode == "end":
                    self.position[motor_id] = self.config.travel_length
                self.write_position(self.position)
                break
            try:
                dis = float(dis_str)
            except Exception:
                self.print("Invalid distance entered", thr=0)
                continue
            self.displace(motor_id=motor_id, dis=dis, pos_check=False)

        self.print(f"Calibration for linear track {motor_id} complete", thr=1)

    def interactive_move(self, motor_id=0):
        self.print(f"Starting interactive move for linear track {motor_id}", thr=1)
        while True:
            dis_str = input("Enter the distance to move in mm, empty if need to break: ")
            if dis_str == "":
                break
            try:
                dis = float(dis_str)
            except Exception:
                self.print("Invalid distance entered", thr=0)
                continue
            self.displace(motor_id=motor_id, dis=dis)

    def read_position(self):
        self.position = [0.0] * self.config.n_motors
        with open(self.config.position_file_path) as f:
            for i in range(self.config.n_motors):
                self.position[i] = float(f.readline())
            # self.position = float(f.readline(4))
        return self.position

    def write_position(self, position):
        with open(self.config.position_file_path, "w") as f:
            for i in range(self.config.n_motors):
                f.write(str(position[i]))
                f.write("\n")
            # f.write(str(position))

    def set_direction(self, motor_id=0, direction="forward"):
        if direction == "forward":
            self.direction_out[motor_id].throttle = 0.0
        elif direction == "backward":
            self.direction_out[motor_id].throttle = 1.0

    def move(self, motor_id=0, move_time=0.0):
        self.pulse_pwm[motor_id].throttle = 0.5
        sleep_time = max(move_time - self.config.overhead_time, 0.0)
        time.sleep(sleep_time)
        self.stop(motor_id=motor_id)

    # def move(self, move_time=0.1):
    #     for i in range(int(move_time/delay)):
    #         kit.stepper1.onestep(style=stepper.DOUBLE)
    #         step_motor.onestep(style=stepper.DOUBLE)
    #         time.sleep(delay)

    def dis2time(self, dis=0.0):
        dis = self.config.dis_coeff * dis
        t = dis * (self.config.pulse_per_rev) / (self.config.pulse_freq * self.config.dis_per_rev)
        return t

    def time2dis(self, t=0.0):
        dis = t * (self.config.pulse_freq * self.config.dis_per_rev) / (self.config.pulse_per_rev)
        dis = dis / self.config.dis_coeff
        return dis

    def position_check(self, motor_id=0, dis=0.0):
        """
        The position valye is maintained and stored to keep track
        of where the linear track's gantry plate is positioned and can
        be used to bring the plate back to home position(if needed)
        """
        position = self.position[motor_id] + dis
        if position > self.config.travel_length or position < 0:
            raise Exception(f"Gantry plate at linear track {motor_id} already at the edge")
            success = False
        else:
            success = True

        self.print(f"The new distance from home for linear track {motor_id} is {position}mm", thr=2)
        return success, position

    def displace(self, motor_id=0, dis=0.0, pos_check=True):
        self.print(f"Displacing linear track {motor_id} by {dis}mm", thr=1)
        if pos_check:
            result, position = self.position_check(motor_id, dis)
        else:
            result = True
            position = 0.0

        if result:
            direction = "forward" if dis >= 0 else "backward"
            self.set_direction(motor_id=motor_id, direction=direction)
            move_time = self.dis2time(abs(dis))
            self.move(motor_id=motor_id, move_time=move_time)

            self.position[motor_id] = position
            self.write_position(self.position)

            success = True
            status = None
        else:
            success = False
            status = "invalid_distance"
        return success, status

    def return2home(self, motor_id=0):
        self.print(f"Returning to home position on linea track {motor_id}", thr=1)
        dis_from_home = self.position[motor_id]

        success = True
        status = None
        if dis_from_home > 0:
            success, status = self.displace(motor_id=motor_id, dis=-1 * dis_from_home)
        elif dis_from_home == 0:
            self.print(f"Gantry plate of linear track {motor_id} already at home")
        else:
            raise Exception(
                "The position status variable is negative. Please check the position file"
            )

        return success, status

    def go2end(self, motor_id=0):
        self.print(f"Going to the end of the line on linear track {motor_id}", thr=1)
        dis_from_end = self.config.travel_length - self.position[motor_id]

        success = True
        status = None
        if dis_from_end > 0:
            success, status = self.displace(motor_id=motor_id, dis=dis_from_end)
        elif dis_from_end == 0:
            self.print("Gantry plate on linear track {} already at the end", format(motor_id))
        else:
            raise Exception(
                "The position status variable is negative for gotoend. Please check the position file"
            )

        return success, status

    def back_and_forth(self, motor_id=0, distance=100.0, margin=100.0, repeats=8, delay=2.0):
        self.print(f"Moving linear track {motor_id} back and forth", thr=1)
        direction = "forward"
        rep_id = 0

        while True:
            time.sleep(delay)
            rep_id += 1

            if direction == "forward":
                direction = 1
            elif direction == "backward":
                direction = -1
            else:
                raise Exception("Invalid direction")
            dist = distance * direction
            success, status = self.displace(motor_id=motor_id, dis=dist)
            if not success:
                break

            if rep_id >= repeats:
                rep_id = 0
                if direction == "forward":
                    direction = "backward"
                elif direction == "backward":
                    direction = "forward"

            if self.position[motor_id] >= self.config.travel_length - margin:
                rep_id = 0
                direction = "backward"
            elif self.position[motor_id] <= margin:
                rep_id = 0
                direction = "forward"

            if rep_id == 0:
                time.sleep(5 * delay)

    def stop(self, motor_id=0):
        self.pulse_pwm[motor_id].throttle = 0.0
        self.direction_out[motor_id].throttle = 0.0

    def reset(self):
        self.print("Resetting all the motors", thr=1)
        self.kit.motor1.throttle = 0.0
        self.kit.motor2.throttle = 0.0
        self.kit.motor3.throttle = 0.0
        self.kit.motor4.throttle = 0.0


# def on_program_exit():
#     kit = MotorKit(i2c=board.I2C())
#     kit.motor1.throttle = 0.0
#     kit.motor2.throttle = 0.0
#     kit.motor3.throttle = 0.0
#     kit.motor4.throttle = 0.0
#     print("Exiting the program")


if __name__ == "__main__":
    lintrack_config = LinearTrackControllerConfig()
    lt = LinearTrackController(lintrack_config)

    atexit.register(lt.reset)
    # atexit.register(on_program_exit)

    # lt.calibrate(motor_id=0, mode='start')
    # lt.calibrate(motor_id=1, mode='start')
    # lt.calibrate(motor_id=0, mode='end')
    # lt.calibrate(motor_id=1, mode='end')
    # lt.return2home(motor_id=0)
    # lt.return2home(motor_id=1)
    # lt.go2end(motor_id=0)
    # lt.go2end(motor_id=1)

    lt.interactive_move(motor_id=0)
    lt.interactive_move(motor_id=1)
    # lt.back_and_forth(motor_id=0, distance=100.0, margin=100.0, repeats=8, delay=3.0)

    lt.run_tcp()
