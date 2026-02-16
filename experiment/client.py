import sys
import os
import math
import json
import socket
import errno
import time
import numpy as np
from pathlib import Path as P
from Helper import *
from classes import Environment, Agent

# sys.path.append("/usr/local/lib")
# sys.path.append(str(P(__file__).absolute().parent.parent))
# from rollereye import *


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
        self.is_complete = False
        
        try:
            config = load_config('config.json')
        except FileNotFoundError:
            print("❌ Error: config.json not found!")
            sys.exit(1)
        
        n = config['n']
        m = config['m']
        h = config['h']
        com_range = config['com_range']
        alpha1 = config['alpha1']
        alpha2 = config['alpha2']
        alpha3 = config['alpha3']
        node_labels_t = config['node_labels_t']
        
        env = Environment(n, m, node_labels_t)
        
        with open('agents_config.json', 'r') as f:
            agents_configs = json.load(f)
        
        agent_config = None
        for config in agents_configs:
            if config['id'] == self.rid:
                agent_config = config
                break
        
        if agent_config is None:
            print(f"Error: No configuration found for agent {self.rid}")
            exit()
        
        initial_position = agent_config['start']
        formula_str = agent_config['mission']
        
        print(f"   Mission: {formula_str}")
        print(f"   Starting position: {initial_position}")
            
        
        self.agent = Agent(
            agent_id=self.rid,
            initial_position=initial_position,
            formula_str=formula_str,
            env=env,
            h=h,
            alpha1=alpha1,
            alpha2=alpha2,
            alpha3=alpha3
        )

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
        if server_message.get("type") is None or server_message.get("type") != "snapshots":
            my_print("Received server message: ", server_message)
        if server_message.get("type") == "exit":
            raise Exception("Exit received")
        return server_message

    # def move_to(self, dst):
    #     # TODO: Maybe read from message in between, to make sure you are not exitted
    #     while True:
    #         message = {"type": "pose"}
    #         t = time.time()
    #         self.send_msg(message)
    #         response = self.read_msg(blocking=True)
    #         my_print(
    #             "Received response from server: {}, with a delay of {}. Dst is {}".format(
    #                 response, time.time() - t, dst
    #             )
    #         )
    #         xy = response["xy"]
    #         heading = response["heading"]

    #         distance = math.sqrt((dst[1] - xy[1]) ** 2 + (dst[0] - xy[0]) ** 2)
    #         my_print("distance is: ", distance)
    #         if distance < ACCEPTANCE_DISTANCE:
    #             rollereye.stop_move()
    #             break

    #         direction = (
    #             math.atan2(dst[1] - xy[1], dst[0] - xy[0]) / math.pi * 180
    #         )  # -180 < dir < 180
    #         if direction < 0:
    #             direction += 360
    #         direction -= heading
    #         if direction < 0:
    #             direction += 360
    #         direction = 360 - direction
    #         # my_print("Moving to ", dst)

    #         Kp = 5
    #         control_speed = max(min(Kp * distance, 1), 0.15)
    #         my_print("control is: ", control_speed, direction)
    #         rollereye.set_translate_4(direction, control_speed)


    def get_location(self):
        self.send_msg({"type": "pose"})
        return self.read_msg(blocking=True)


    def run(self):
        # rollereye.start()
        # rollereye.timerStart()
        while True:
            msg = self.read_msg(blocking=True)

            if msg['type'] == 'request_snapshot':
                # Send snapshot to server
                self.send_snapshot()
                
            elif msg['type'] == 'shutdown':
                break
            else:
                raise Exception(f"Was waiting for request_snapshot but got: {msg['type']}")

             
            if self.is_complete:
                my_print(f"Agent {self.rid} already completed mission, ignoring further commands.")
                continue

            other_snapshots = self.receive_snapshots()
            
            if other_snapshots:
                my_print(f"Agent {self.rid} received snapshots from: {list(other_snapshots.keys())}")
                
                updates = self.agent.prepare_communication_updates_decentralized(other_snapshots)
                self.agent.apply_communication_updates_decentralized(updates)
                
                learned_from = updates.get('learned_from', {})
                new_nodes = updates['new_visited']
                if len(new_nodes) > 0:
                    breakdown = ', '.join([f"{count} from Agent {aid}" for aid, count in learned_from.items() if count > 0])
                    my_print(f"Learned {len(new_nodes)} new nodes ({breakdown})")

            step_msg = self.read_msg(blocking=True)
            
            if step_msg['type'] != 'step':
                raise Exception(f"Was waiting for step command but got: {step_msg['type']}")
            
            iteration = step_msg['iteration']
                
            
            my_print(f"{'='*60}")
            my_print(f"Agent {self.rid} - Iteration {iteration}")
            my_print(f"{'='*60}")
            
            self.agent.update_product_automaton()
            
            if self.agent.check_mission_complete():
                my_print(f"Agent {self.rid} mission complete!")
                self.send_complete()
                continue
            
            if not hasattr(self.agent, 'current_plan') or self.agent.current_plan is None or len(self.agent.current_plan) == 0:
                # Build agents_in_range from snapshots for frontier selection
                agents_in_range_positions = {
                    int(aid): snapshot['position'] 
                    for aid, snapshot in other_snapshots.items()
                }
                
                best_frontier, path = self.agent.select_frontier(agents_in_range=agents_in_range_positions)
                
                if best_frontier is None or path is None:
                    my_print(f"Agent {self.rid}: Cannot proceed further.")
                    self.send_complete()
                    continue
                else:
                    my_print(f"Agent {self.rid} planning to move to frontier {best_frontier} ({len(path)} steps)")
                    self.agent.current_plan = path
                    self.agent.current_frontier = best_frontier
            
            if hasattr(self.agent, 'current_plan') and self.agent.current_plan is not None and len(self.agent.current_plan) > 1:
                next_position = self.agent.current_plan[1]
                self.agent.move_one_step(next_position)
                self.agent.current_plan = self.agent.current_plan[1:]
                
                my_print(f"  Agent {self.rid} moved to {self.agent.current_physical_state} ({len(self.agent.current_plan)-1} steps remaining)")
                
                if len(self.agent.current_plan) == 1:
                    my_print(f"    Agent {self.rid} reached frontier {self.agent.current_frontier}")
                    self.agent.current_plan = None
            
            self.send_ready()
        

        my_print(f"{'='*60}")
        my_print(f"Agent {self.rid} - FINAL RESULTS")
        my_print(f"{'='*60}")
        my_print(f"Mission: {self.agent.formula_str}")
        my_print(f"Final position: {self.agent.current_physical_state}")
        my_print(f"Trajectory length: {len(self.agent.full_physical_traj)}")
        my_print(f"Trajectory: {self.agent.full_physical_traj}")

    def exit(self):
        # rollereye.handle_exception(e.__class__.__name__ + ": " + e.message)
        # rollereye.stop()
        self.socket.close()  # close the connection

    def send_snapshot(self):
        snapshot = {
            'position': self.agent.current_physical_state,
            'visited': list(self.agent.visited),  # Convert set to list for JSON
            'node_labels': {k: list(v) for k, v in self.agent.node_labels.items()}  # Convert sets to lists
        }
        
        message = {
            'type': 'snapshot',
            'data': snapshot
        }
        self.send_msg(message)

    def send_complete(self):
        self.is_complete = True
        message = {
            'type': 'complete',
        }
        self.send_msg(message)

    def send_ready(self):
        message = {
            'type': 'ready',
        }
        self.send_msg(message)

    def receive_snapshots(self):
        msg = self.read_msg()
        
        if msg['type'] == 'snapshots':
            # Convert lists back to sets
            agents_in_range = msg['agents_in_range']
            
            for other_id, snapshot in agents_in_range.items():
                snapshot['visited'] = set(snapshot['visited'])
                snapshot['node_labels'] = {
                    int(k): set(v) for k, v in snapshot['node_labels'].items()
                }
            
            return agents_in_range
        else: 
            raise Exception(f"Expected 'snapshots' message but got {msg['type']}")
        

if __name__ == "__main__":
    SERVER_IP = "localhost"
    # SERVER_IP = "192.168.1.111"
    SERVER_PORT = 5000
    
    if len(sys.argv) > 1:
        SERVER_IP = sys.argv[2] 
    if len(sys.argv) > 2:
        SERVER_PORT= int(sys.argv[3])
    

    client_socket = socket.socket()
    client_socket.connect((SERVER_IP, SERVER_PORT))
    # client_socket.setblocking(False)

    moorebot = Moorebot(client_socket)

    try:
        moorebot.run()
    except Exception as e:
        my_print("An error here: ", e)
        pass
    moorebot.exit()
