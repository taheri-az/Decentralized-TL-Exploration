import sys
import numpy as np
from pathlib import Path as P

sys.path.append(str(P(__file__).absolute().parent.parent))
import socket
import threading
import json
# import cv2
import time
# from localization.localize import localize_fast


RID_TO_ARUCOID = {0: 0, 1: 1, 2: 2}

# vid = cv2.VideoCapture(0)

# width = vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# height = vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
# # fps = vid.set(cv2.CAP_PROP_FPS, 10)
# # fps = vid.get(cv2.CAP_PROP_FPS)


# camera_lock = threading.Lock()


# def read():
#     while True:
#         with camera_lock:
#             ret = vid.grab()
#         # Note that sleep is much less than frame speed which is about 0.02. The reason is when CPU is doing sth else, frames are qued and we should empty them very fast without much sleep.
#         # sleep(0.001)
#         if not ret:
#             break


# reader_thread = threading.Thread(target=read, daemon=True)
# reader_thread.start()


class RobotHandler(threading.Thread):
    def __init__(self, socket, rid) -> None:
        super(RobotHandler, self).__init__()
        self.socket = socket
        self.rid = rid
        self.aruco_id = RID_TO_ARUCOID[rid]
        self.exit = False
        self.send_msg({"rid": rid})
        self.next_robots_called = 0

        self.is_ready = False
        self.is_complete = False
        self.snapshot = None

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
            if message_type == "pose":
                print("Getting pose for robot: ", self.rid)
                response["xy"], response["heading"] = self.get_location()
                while response["xy"] is None:
                    response["xy"], response["heading"] = self.get_location()
            elif message_type== 'snapshot':
                self.snapshot = message['data']
                print(f"Received snapshot from Agent {self.rid}")
            
            elif message_type == 'ready':
                self.is_ready = True
                if not self.is_complete:
                    print(f"   Agent {self.rid} ready")
            
            elif message_type == 'complete':
                was_active = not self.is_complete
                self.is_complete = True
                self.is_ready = True
                if was_active:
                    print(f"Agent {self.rid} mission complete!")


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

    def send_msg(self, msg):
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


    # def get_location(self):
    #     try:
    #         ret, frame = vid.retrieve()
    #         cv2.imwrite("a.jpg", frame)
    #         ids, locations, headings = localize_fast(frame)
    #         locations, headings = self.scale_and_flip(locations, headings)
    #         my_location = locations[ids.index(self.aruco_id)]
    #         return [my_location[0], my_location[1]], headings[ids.index(self.aruco_id)]
    #     except:
    #         return None, None

    def close(self):
        self.send_msg({"type": "shutdown"})
        self.socket.close()


    def get_snapshot(self):
        self.snapshot = None
        self.send_msg({"type": "request_snapshot"})
    
        
    def step(self, iteration):
        message = {
            'type': 'step',
            'iteration': iteration
        }
        self.send_msg(message)

class Server:
    def __init__(self, num_agents=2, max_iterations=1000, grid_size=20, com_range=300):
        self.host = "0.0.0.0"
        self.port = 5000
        self.num_agents = num_agents
        self.max_iterations = max_iterations
        self.grid_size = grid_size
        self.com_range = com_range
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))


        self.robot_handlers = []

    def run(self):
        print("Starting server on port: ", self.port, "Waiting for clients to connect...")

        self.socket.listen(10)
        for i in range(self.num_agents):
            # configure how many client the server can listen simultaneously
            try:
                client_socket, address = self.socket.accept()  # accept new connection
            except:
                print("Not accepting new clients anymore")
                break

            print(f"Connection from: {str(address)}, starting a new thread")
            # TODO: Check if I should somehow make this daemon
            robot_handler = RobotHandler(client_socket, len(self.robot_handlers))
            self.robot_handlers.append(robot_handler)
            robot_handler.start()

        command = input("Give command -> ")
        if command != "start":
            exit()
                

        self.iteration = 0
        while self.iteration < self.max_iterations:
            self.iteration += 1
            print(f"{'='*60}")
            print(f"Iteration {self.iteration}")
            print(f"{'='*60}")
            
            self.get_snapshots()
            self.broadcast_snapshots()
            self.broadcast_step()
            
            status = self.wait_for_ready()
            
            if status == False:
                break
            time.sleep(1)
        
        self.shutdown()

    def shutdown(self):
        print("Closing sockets")
        self.socket.close()  # close the connection
        for robot_handler in self.robot_handlers:
            robot_handler.close()
            robot_handler.stop()
            robot_handler.join()
    
    def get_snapshots(self):
        for handler in self.robot_handlers:
            handler.get_snapshot()
        print("waiting for snapshots...")
        while True:
            if all(not handler.snapshot is None for handler in self.robot_handlers):
                return True
            time.sleep(0.1)
        
    
    def wait_for_ready(self):
        print("waiting for ready...")
        while True:
            if all(handler.is_complete or handler.is_ready for handler in self.robot_handlers):
                if all(handler.is_complete for handler in self.robot_handlers):
                    print("All agents complete!")
                    return False
                return True
            time.sleep(0.1)

    def broadcast_snapshots(self):
        snapshots = {}
        for rid in range(self.num_agents):
            snapshots[rid] = self.robot_handlers[rid].snapshot
        
        print("robot snapshots: ", snapshots.keys())
        for rid, snapshot in snapshots.items():
            if self.robot_handlers[rid].is_complete:
                continue
            agent_pos = snapshot['position']
            r1, c1 = agent_pos // self.grid_size, agent_pos % self.grid_size
            
            agents_in_range = {}
            for neighbor_id, neighbor_snapshot in snapshots.items():
                if neighbor_id == rid:
                    continue
                
                other_pos = neighbor_snapshot['position']
                r2, c2 = other_pos // self.grid_size, other_pos % self.grid_size
                dist = abs(r1 - r2) + abs(c1 - c2)
                
                if dist <= self.com_range:
                    agents_in_range[neighbor_id] = neighbor_snapshot
            
            message = {
                'type': 'snapshots',
                'agents_in_range': agents_in_range
            }
            
            self.robot_handlers[rid].send_msg(message)
    
    def broadcast_step(self):
        for handler in self.robot_handlers:
            if not handler.is_complete:
                handler.step(self.iteration)

if __name__ == "__main__":
    server = Server()
    server.run()