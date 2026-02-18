import socket
import threading
import json
import time
import sys
from pathlib import Path as P
from Helper import *
import cv2
from localization.localize import localize_fast

# Load agent configurations and compute DFA data
try:
    with open('agents_config.json', 'r') as f:
        agents_configs = json.load(f)
    
    initial_positions = {}
    dfa_transitions_dict = {}
    dfa_initial_states = {}
    dfa_trash_states = {}
    commit_states_dict = {}
    
    for config in agents_configs:
        aid = config['id']
        initial_positions[aid] = config['start']
        formula_str = config['mission']
        
        # Compute DFA for this agent's mission
        dfa_transitions_dict[aid], dfa_initial_states[aid], dfa_trash_states[aid] = \
            extract_dfa_transitions_with_trash_expanded(formula_str)
        
        commit_states_dict[aid] = compute_commit_states(
            formula_str, 
            dot_file=f"agent_{aid}_product", 
            fmt="pdf"
        )
        
        print(f"Loaded Agent {aid}:")
        print(f"   Mission: {formula_str}")
        print(f"   Starting position: {initial_positions[aid]}")
        
except FileNotFoundError:
    print("❌ Error: agents_config.json not found!")
    sys.exit(1)


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
        self.aruco_id = RID_TO_ARUCOID.get(rid, rid)
        self.exit = False
        
        # Send RID confirmation
        self.send_msg({"rid": rid})
        
        # Send initialization data with DFA
        self.send_init_data()
        
        self.next_robots_called = 0
        self.is_ready = False
        self.is_complete = False
        self.snapshot = None
    
    def send_init_data(self):
        """Send DFA and initial position to client"""
        init_msg = {
            'type': 'init',
            'initial_position': initial_positions[self.rid],
            'dfa_transitions': dfa_transitions_dict[self.rid],
            'initial_state': dfa_initial_states[self.rid],
            'trash_states_set': list(dfa_trash_states[self.rid]),
            'commit_states': list(commit_states_dict[self.rid])
        }
        self.send_msg(init_msg)
        # print("init", init_msg)
        print(f"✅ Sent init data to Agent {self.rid}")
    
    def run(self):
        while True:
            if self.exit:
                return

            message = self.read_msg()
            if not message:
                print("EXITING ROBOT HANDLER.")
                return
            
            try:
                message = json.loads(message)
            except:
                print(f"Failed to parse message from robot {self.rid}")
                return

            message_type = message["type"]
            # print(f"A request from robot: {message['rid']}")

            response = {}
            if message_type == "get_pose":
                response["type"] = "pose"
                print("Getting pose for robot: ", self.rid)
                response["xy"], response["heading"] = self.get_location()
                while response["xy"] is None:
                    print("try again")
                    response["xy"], response["heading"] = self.get_location()
                self.send_msg(response)
            elif message_type== 'snapshot':
                self.snapshot = message['data']
                print(f"Agnet {self.rid}: Received snapshot")
            
            elif message_type == 'ready':
                self.is_ready = True
                if not self.is_complete:
                    print(f"Agent {self.rid}: Received ready")
            
            elif message_type == 'complete':
                was_active = not self.is_complete
                self.is_complete = True
                self.is_ready = True
                if was_active:
                    print(f"Agent {self.rid}: mission complete!")

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
        try:
            msg_length_bytes = self.recv_all(10)
            if not msg_length_bytes or len(msg_length_bytes) < 10:
                return None
            msg_length = int(msg_length_bytes)
        except (ValueError, Exception) as e:
            print(f"Error reading message length from robot {self.rid}: {e}")
            return None
        
        msg = self.recv_all(msg_length)
        if not msg:
            return None
        return msg.decode()

    def send_msg(self, msg):
        data = json.dumps(msg).encode()
        data_length = len(data)
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
        self.send_msg({"type": "shutdown"})
        self.socket.close()

    def get_snapshot(self):
        print("Handler sending request for snapshot")
        self.snapshot = None
        self.send_msg({"type": "request_snapshot"})
    
    def step(self):
        print("Handler sending request for step")
        message = {
            'type': 'step',
        }
        self.send_msg(message)


class Server:
    def __init__(self, num_agents=1, max_iterations=1000, grid_size=20, com_range=300):
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
            try:
                client_socket, address = self.socket.accept()
            except:
                print("Not accepting new clients anymore")
                break

            print(f"Connection from: {str(address)}, starting a new thread")
            robot_handler = RobotHandler(client_socket, len(self.robot_handlers))
            self.robot_handlers.append(robot_handler)
            robot_handler.start()

        # command = input("Give command -> ")
        # if command != "start":
        #     exit()
        
        # Give agents time to process init messages
        print("⏳ Waiting for agents to initialize...")
        time.sleep(2)

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
            time.sleep(0.01)
        
        self.shutdown()

    def shutdown(self):
        print("Closing sockets")
        
        # First, signal all handlers to stop
        for robot_handler in self.robot_handlers:
            robot_handler.stop()
        
        # Send shutdown messages
        for robot_handler in self.robot_handlers:
            try:
                robot_handler.close()
            except:
                pass
        
        # Give threads time to exit gracefully
        time.sleep(0.5)
        
        # Close main socket
        self.socket.close()
        
        # Wait for threads to finish (with timeout)
        for robot_handler in self.robot_handlers:
            robot_handler.join(timeout=1.0)
    
    def get_snapshots(self):
        print("1. Send req for snapshots")
        for handler in self.robot_handlers:
            handler.get_snapshot()
        print("2. Waiting for snapshots...")
        while True:
            if all(not handler.snapshot is None for handler in self.robot_handlers):
                print("3. Got snapshots")
                return True
            time.sleep(0.1)
    
    def wait_for_ready(self):
        print("waiting for ready...")
        while True:
            if all(handler.is_complete or handler.is_ready for handler in self.robot_handlers):
                print("ready received")
                if all(handler.is_complete for handler in self.robot_handlers):
                    print("All agents complete!")
                    return False
                return True
            time.sleep(0.1)

    def broadcast_snapshots(self):
        print("broadcasting snapshots")
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
            handler.is_ready = False
            if not handler.is_complete:
                handler.step()


if __name__ == "__main__":
    server = Server()
    server.run()