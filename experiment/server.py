import sys
import numpy as np
from pathlib import Path as P

sys.path.append(str(P(__file__).absolute().parent.parent))
import socket
import threading
import json
import cv2
from time import time
from localization.localize import localize_fast


RID_TO_ARUCOID = {0: 0, 1: 1, 2: 2}

vid = cv2.VideoCapture(0)

width = vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
height = vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
# fps = vid.set(cv2.CAP_PROP_FPS, 10)
# fps = vid.get(cv2.CAP_PROP_FPS)


camera_lock = threading.Lock()


def read():
    while True:
        with camera_lock:
            ret = vid.grab()
        # Note that sleep is much less than frame speed which is about 0.02. The reason is when CPU is doing sth else, frames are qued and we should empty them very fast without much sleep.
        # sleep(0.001)
        if not ret:
            break


reader_thread = threading.Thread(target=read, daemon=True)
reader_thread.start()


class RobotHandler(threading.Thread):
    def __init__(self, socket, rid) -> None:
        super(RobotHandler, self).__init__()
        self.socket = socket
        self.rid = rid
        self.aruco_id = RID_TO_ARUCOID[rid]
        self.exit = False
        self.send_msg({"rid": rid})
        self.next_robots_called = 0

    def run(self):
        while True:
            if self.exit:
                exit()

            message = self.read_msg()
            if not message:
                print("EXITING ROBOT HANDLER.")
                exit()
            message = json.loads(message)

            message_type = message["type"]
            print(f"A request from robot: {message['rid']}")

            response = {}
            to_all = False
            if message_type == "pose":
                print("Getting pose for robot: ", self.rid)
                response["xy"], response["heading"] = self.get_location()
                while response["xy"] is None:
                    response["xy"], response["heading"] = self.get_location()

            self.send_msg(response, to_all)

    def stop(self):
        self.exit = True

    def recv_all(self, total_bytes):
        data = b""
        while len(data) < total_bytes:
            remaining_bytes = total_bytes - len(data)
            packet = self.socket.recv(min(remaining_bytes, 1024))
            if not packet:
                break
            data += packet
        return data

    def read_msg(self):
        msg_length = int(self.recv_all(10))
        # print("read_msg len: ", msg_length)
        msg = self.recv_all(msg_length)
        # print("read msg: ", msg)
        return msg.decode()

    def send_msg(self, msg, to_all=False):
        if to_all:
            for robot_handler in robot_handlers:
                robot_handler.send_msg(msg)
            return

        # print("Sending message to: ", self.rid)
        data = json.dumps(msg).encode()
        data_length = len(data)
        # print("Sending message: ", str(data_length).encode().rjust(10, b"0"), data)
        self.socket.send(str(data_length).encode().rjust(10, b"0") + data)

    def get_neighbors(self, my_id, ids, locations):
        return []

    def scale_and_flip(self, locations, headings):
        new_headings = (360 - np.array(headings)).tolist()
        new_locations = []
        for location in locations:
            new_locations.append([location[0], -1 * location[1]])
        return new_locations, new_headings


    def get_location(self):
        try:
            ret, frame = vid.retrieve()
            cv2.imwrite("a.jpg", frame)
            ids, locations, headings = localize_fast(frame)
            locations, headings = self.scale_and_flip(locations, headings)
            my_location = locations[ids.index(self.aruco_id)]
            return [my_location[0], my_location[1]], headings[ids.index(self.aruco_id)]
        except:
            return None, None

    def close(self):
        self.send_msg({"type": "exit"})
        self.socket.close()

    def send_start(self):
        msg = {"type": "start"}

        msg["x"] = 1.0
        msg["y"] = 1.0

        self.send_msg(msg)
        print("Sent start to: ", self.rid)


robot_handlers = []


def server_program():

    host = "0.0.0.0"
    port = 5000
    print("Starting server on port: ", port, "Waiting for clients to connect...")

    server_socket = socket.socket()
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))

    server_socket.listen(10)
    while True:
        # configure how many client the server can listen simultaneously
        try:
            client_socket, address = server_socket.accept()  # accept new connection
        except:
            print("Not accepting new clients anymore")
            break

        print(f"Connection from: {str(address)}, starting a new thread")
        # TODO: Check if I should somehow make this daemon
        robot_handler = RobotHandler(client_socket, len(robot_handlers))
        robot_handlers.append(robot_handler)
        robot_handler.start()
        # from time import sleep
        # sleep(0.1)
        # break

    while True:
        try:
            command = input("Give command -> ")
        except:
            break
        if command == "start":
            for robot_handler in robot_handlers:
                robot_handler.send_start()
        if command == "exit":
            break

    print("Closing sockets")
    server_socket.close()  # close the connection
    for robot_handler in robot_handlers:
        robot_handler.close()
        robot_handler.stop()
        robot_handler.join()


if __name__ == "__main__":
    server_program()
