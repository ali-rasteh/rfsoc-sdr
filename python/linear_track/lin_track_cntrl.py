import os
import time
from types import SimpleNamespace
import atexit

import board
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper

from tcp_comm import Tcp_Comm_LinTrack
from general import General




# TODO add config class and transfer all parameters

class Configs_Class(object):
    def __init__(self):
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--output_mode", type=str, default="dc", help="Type of the Raspberry Pi hat output to use")
        # parser.add_argument("--dis_per_rev", type=int, default=8, help="Distance on the linear track per stepper motor revolution")
        # parser.add_argument("--pulse_per_rev", type=int, default=400, help="Pulse needed on the logic for each revolution of the stepper motor")
        # parser.add_argument("--pulse_freq", type=float, default=1600, help="Pulse frequency of the logic circuit")
        # parser.add_argument("--plot_level", type=int, default=0, help="level of plotting outputs")
        # parser.add_argument("--verbose_level", type=int, default=0, help="level of printing output")
        # parser.add_argument("--run_tcp_server", action="store_true", default=False, help="If true, runs the TCP server")
        # config = parser.parse_args()
        config = SimpleNamespace()
        config.overwrite_configs=True

        if config.overwrite_configs:
            self.output_mode = 'dc'
            self.tcp_localIP = "0.0.0.0"
            self.tcp_bufferSize=2**10
            self.TCP_port_Cmd=8080
            self.TCP_port_Data=8081
            self.seed=100
            self.run_tcp_server=True
            self.position_file_path = os.path.join(os.getcwd(), 'position.txt')
            self.dis_coeff = 0.972
            self.overhead_time = 0.0018+0.0061+0.0001
            self.n_motors = 2
            
            self.dis_per_rev = 8
            self.pulse_per_rev = 400
            self.pulse_freq = 1600
            self.verbose_level = 4
            self.plot_level = 5




class LinearTrack(General):
    def __init__(self, config):
        super().__init__(config)

        self.run_tcp_server = config.run_tcp_server
        self.output_mode = config.output_mode
        self.dis_per_rev = config.dis_per_rev
        self.pulse_per_rev = config.pulse_per_rev
        self.pulse_freq = config.pulse_freq
        self.dis_coeff = config.dis_coeff
        self.overhead_time = config.overhead_time
        self.position_file_path = config.position_file_path
        self.n_motors = config.n_motors
        self.total_length = 1500      # length of the linear track in mm
        self.plate_length = 125
        self.travel_length = self.total_length - self.plate_length
        self.margin2edge = 5

        self.travel_length -= 2*self.margin2edge

        if self.output_mode == 'stepper':
            self.kit = stepper.StepperMotor(microsteps=2)
        elif self.output_mode == 'dc':
            self.kit = MotorKit(i2c=board.I2C(), pwm_frequency = self.pulse_freq)

        self.pulse_pwm_1 = self.kit.motor1
        self.pulse_pwm_2 = self.kit.motor2
        self.pulse_pwm = [self.pulse_pwm_1, self.pulse_pwm_2]
        self.direction_out_1 = self.kit.motor3
        self.direction_out_2 = self.kit.motor4
        self.direction_out = [self.direction_out_1, self.direction_out_2]

        self.reset()

        self.position = self.read_position()
        
        if self.run_tcp_server:
            self.tcp_comm = Tcp_Comm_LinTrack(config)
            self.tcp_comm.init_tcp_server()


    def run_tcp(self):
        self.print("Running TCP server", thr=1)
        self.tcp_comm.obj_lintrack = self
        self.tcp_comm.run_tcp_server(self.tcp_comm.parse_and_execute)


    def calibrate(self, motor_id=0, mode='start'):
        self.print("Calibrating the linear track {} with mode {}".format(motor_id, mode), thr=1)
        while True:
            dis_str = input("Enter the distance to move in mm, empty if need to break: ")
            if dis_str == '':
                if mode == 'start':
                    self.position[motor_id] = 0.0
                elif mode == 'end':
                    self.position[motor_id] = self.travel_length
                self.write_position(self.position)
                break
            try:
                dis = float(dis_str)
            except:
                self.print("Invalid distance entered", thr=0)
                continue
            self.displace(motor_id=motor_id, dis=dis, pos_check=False)

        self.print("Calibration for linear track {} complete".format(motor_id), thr=1)


    def interactive_move(self, motor_id=0):
        self.print("Starting interactive move for linear track {}".format(motor_id), thr=1)
        while True:
            dis_str = input("Enter the distance to move in mm, empty if need to break: ")
            if dis_str == '':
                break
            try:
                dis = float(dis_str)
            except:
                self.print("Invalid distance entered", thr=0)
                continue
            self.displace(motor_id=motor_id, dis=dis)


    def read_position(self):
        self.position = [0.0]*self.n_motors
        with open(self.position_file_path,'r') as f:
            for i in range(self.n_motors):
                self.position[i] = float(f.readline())
            # self.position = float(f.readline(4))
        return self.position


    def write_position(self, position):
        with open(self.position_file_path,'w') as f:
            for i in range(self.n_motors):
                f.write(str(position[i]))
                f.write('\n')
            # f.write(str(position))


    def set_direction(self, motor_id=0, direction='forward'):
        if direction=='forward':
            self.direction_out[motor_id].throttle = 0.0
        elif direction=='backward':
            self.direction_out[motor_id].throttle = 1.0


    def move(self, motor_id=0, move_time=0.0):
        self.pulse_pwm[motor_id].throttle = 0.5
        sleep_time = max(move_time-self.overhead_time, 0.0)
        time.sleep(sleep_time)
        self.stop(motor_id=motor_id)

    # def move(self, move_time=0.1):
    #     for i in range(int(move_time/delay)):
    #         kit.stepper1.onestep(style=stepper.DOUBLE)
    #         step_motor.onestep(style=stepper.DOUBLE)
    #         time.sleep(delay)
    

    def dis2time(self, dis=0.0):
        dis = self.dis_coeff * dis
        t = dis * (self.pulse_per_rev) / (self.pulse_freq * self.dis_per_rev)
        return t


    def time2dis(self, t=0.0):
        dis = t * (self.pulse_freq * self.dis_per_rev) / (self.pulse_per_rev)
        dis = dis / self.dis_coeff
        return dis


    def position_check(self, motor_id=0, dis=0.0):
        """
        The position valye is maintained and stored to keep track 
        of where the linear track's gantry plate is positioned and can
        be used to bring the plate back to home position(if needed)
        """
        position = self.position[motor_id] + dis
        if position > self.travel_length or position < 0:
            raise Exception("Gantry plate at linear track {} already at the edge".format(motor_id))
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
            direction = 'forward' if dis>=0 else 'backward'
            self.set_direction(motor_id=motor_id, direction=direction)
            move_time = self.dis2time(abs(dis))
            self.move(motor_id=motor_id, move_time = move_time)

            self.position[motor_id] = position
            self.write_position(self.position)

            success = True
            status = None
        else:
            success = False
            status = "invalid_distance"
        return success, status


    def return2home(self, motor_id=0):
        self.print("Returning to home position on linea track {}".format(motor_id), thr=1)
        dis_from_home = self.position[motor_id]

        success = True
        status = None
        if dis_from_home > 0:
            success, status = self.displace(motor_id=motor_id, dis=-1 * dis_from_home)
        elif dis_from_home == 0:
            print("Gantry plate of linear track {} already at home".format(motor_id))
        else:
            raise Exception("The position status variable is negative. Please check the position file")

        return success, status
    

    def go2end(self, motor_id=0):
        self.print("Going to the end of the line on linear track {}".format(motor_id), thr=1)
        dis_from_end = self.travel_length - self.position[motor_id]

        success = True
        status = None
        if dis_from_end > 0:
            success, status = self.displace(motor_id=motor_id, dis=dis_from_end)
        elif dis_from_end == 0:
            print("Gantry plate on linear track {} already at the end",format(motor_id))
        else:
            raise Exception("The position status variable is negative for gotoend. Please check the position file")

        return success, status
    

    def back_and_forth(self, motor_id=0, distance=100.0, margin=100.0, repeats=8, delay=2.0):
        self.print("Moving linear track {} back and forth".format(motor_id), thr=1)
        direction = 'forward'
        rep_id = 0

        while True:
            time.sleep(delay)
            rep_id += 1

            if direction=='forward':
                dir = 1
            elif direction=='backward':
                dir = -1
            else:
                raise Exception("Invalid direction")
            dist = distance * dir
            success, status = self.displace(motor_id=motor_id, dis=dist)
            if success == False:
                break
            
            if rep_id>=repeats:
                rep_id = 0
                if direction == 'forward':
                    direction = 'backward'
                elif direction == 'backward':
                    direction = 'forward'

            if self.position[motor_id] >= self.travel_length - margin:
                rep_id = 0
                direction = 'backward'
            elif self.position[motor_id] <= margin:
                rep_id = 0
                direction = 'forward'

            if rep_id == 0:
                time.sleep(5*delay)




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




def lintrack_run(config):
    lt = LinearTrack(config)

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

    if config.run_tcp_server:
        lt.run_tcp()





if __name__ == '__main__':
    config = Configs_Class()
    lintrack_run(config)
