import sys
import os
import math
import json
import socket
import errno
import time
import numpy as np
from pathlib import Path as P

sys.path.append("/usr/local/lib")
sys.path.append(str(P(__file__).absolute().parent.parent))
from rollereye import *


np.random.seed(0)

ACCEPTANCE_DISTANCE = 0.005

def my_print(*args):
    args = list(args)
    sys.stdout = sys.__stdout__
    while len(args) < 5:
        args.append(None)
    print(args[0], args[1], args[2], args[3], args[4])
    sys.stdout = open(os.devnull, "w")


class Moorebot:
    def __init__(self, socket):
        self.socket = socket
        self.rid = self.read_msg(blocking=True)["rid"]
        self.current_rid = 0

    def wait_for_start(self):
        server_message = self.read_msg(blocking=True)
        if not server_message["type"] == "start":
            print("Expected start, but not received, exitting.")
            exit()

        self.goal_x = server_message["x"]
        self.goal_y = server_message["y"]
        my_print("x", self.goal_x)
        my_print("y", self.goal_y)
        print("Starting algorthim")

    def recv_all(self, total_bytes):
        data = b""
        while len(data) < total_bytes:
            remaining_bytes = total_bytes - len(data)
            packet = self.socket.recv(min(remaining_bytes, 1024))
            if not packet:
                break
            data += packet
        return data

    def send_msg(self, msg):
        msg["rid"] = self.rid
        data = json.dumps(msg).encode()
        data_length = len(data)
        # my_print("Sending message: ", str(data_length).encode().rjust(10, b"0"), data)
        self.socket.send(str(data_length).encode().rjust(10, b"0") + data)

    def read_msg(self, blocking=False):
        msg_length = int(self.recv_all(10))
        server_message = self.recv_all(msg_length).decode()
        server_message = json.loads(server_message)
        my_print("Received server message: ", server_message)
        if server_message.get("type") == "exit":
            raise Exception("Exit received")
        return server_message

    def move_to(self, dst):
        # TODO: Maybe read from message in between, to make sure you are not exitted
        while True:
            message = {"type": "pose"}
            t = time.time()
            self.send_msg(message)
            response = self.read_msg(blocking=True)
            my_print(
                "Received response from server: {}, with a delay of {}. Dst is {}".format(
                    response, time.time() - t, dst
                )
            )
            xy = response["xy"]
            heading = response["heading"]

            distance = math.sqrt((dst[1] - xy[1]) ** 2 + (dst[0] - xy[0]) ** 2)
            my_print("distance is: ", distance)
            if distance < ACCEPTANCE_DISTANCE:
                rollereye.stop_move()
                break

            direction = (
                math.atan2(dst[1] - xy[1], dst[0] - xy[0]) / math.pi * 180
            )  # -180 < dir < 180
            if direction < 0:
                direction += 360
            direction -= heading
            if direction < 0:
                direction += 360
            direction = 360 - direction
            # my_print("Moving to ", dst)

            Kp = 5
            control_speed = max(min(Kp * distance, 1), 0.15)
            my_print("control is: ", control_speed, direction)
            rollereye.set_translate_4(direction, control_speed)


    def get_location(self):
        self.send_msg({"type": "pose"})
        return self.read_msg(blocking=True)


    def run(self):
        rollereye.start()
        rollereye.timerStart()
        while True:
            my_print("Getting location")
            self.move_to([self.goal_x, self.goal_y])
            time.sleep(1)

    def exit(self):
        # rollereye.handle_exception(e.__class__.__name__ + ": " + e.message)
        rollereye.stop()
        self.socket.close()  # close the connection


def client_program():
    SERVER_IP = "192.168.1.111"
    # SERVER_IP = "0.0.0.0"
    SERVER_PORT = 5000

    client_socket = socket.socket()
    client_socket.connect((SERVER_IP, SERVER_PORT))
    # client_socket.setblocking(False)

    moorebot = Moorebot(client_socket)

    moorebot.wait_for_start()

    try:
        moorebot.run()
    except Exception as e:
        my_print("An error here: ", e)
        pass
    moorebot.exit()


if __name__ == "__main__":
    client_program()
