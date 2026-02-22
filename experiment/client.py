# client.py - Python 2 version
import sys
import os
import math
import json
import socket
import errno
import time
import numpy as np
from pathlib import Path as P
from client_helper import *
from classes_2 import Environment, Agent
np.random.seed(0)

REAL_WORLD = False
ACCEPTANCE_DISTANCE = 0.10
if REAL_WORLD:
    sys.path.append("/usr/local/lib")
    sys.path.append(str(P(__file__).absolute().parent.parent))
    from rollereye import *



def my_print(*args):
    if REAL_WORLD:
        args = list(args)
        sys.stdout = sys.__stdout__
        while len(args) < 5:
            args.append(None)
        print(args[0], args[1], args[2], args[3], args[4])
        sys.stdout = open(os.devnull, "w")
    else:
        print(*args)


class Moorebot:
    def __init__(self, socket, REAL_WORLD):
        self.socket = socket
        self.real_world = REAL_WORLD
        
        # Read RID confirmation
        self.rid = self.read_msg(blocking=True)["rid"]
        
        # Read init message with DFA data
        init_msg = self.read_msg(blocking=True)
        
        if init_msg['type'] != 'init':
            print("[ERROR] Expected init message from server")
            sys.exit(1)
        
        self.is_complete = False
        
        try:
            config = load_config('config.json')
        except IOError:
            print("[ERROR] config.json not found!")
            sys.exit(1)
        
        n = config['n']
        m = config['m']
        h = config['h']
        com_range = config['com_range']
        alpha1 = config['alpha1']
        alpha2 = config['alpha2']
        alpha3 = config['alpha3']
        node_labels_t = config['node_labels_t']
        
        self.env = Environment(n, m, node_labels_t)
        
        # Extract DFA data from init message
        initial_position = init_msg['initial_position']
        dfa_transitions = init_msg['dfa_transitions']
        initial_state = init_msg['initial_state']
        trash_states = set(init_msg['trash_states_set'])
        commit_states = set(init_msg['commit_states'])
        
        print("   Starting position: {}".format(initial_position))
        print("   DFA initial state: {}".format(initial_state))
        
        # Create agent with DFA data instead of formula
        self.agent = Agent(
            agent_id=self.rid,
            initial_position=initial_position,
            dfa_transitions=dfa_transitions,
            initial_state=initial_state,
            trash_states_set=trash_states,
            commit_states=commit_states,
            env=self.env,
            h=h,
            alpha1=alpha1,
            alpha2=alpha2,
            alpha3=alpha3
        )
        self.agent.mission_finishing = False

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
        self.socket.send(str(data_length).encode().rjust(10, b"0") + data)

    def read_msg(self, blocking=False):
        msg_length = int(self.recv_all(10))
        server_message = self.recv_all(msg_length).decode()
        server_message = json.loads(server_message)
        if server_message.get("type") == "pose":
            # my_print("Received server message: ", server_message['xy'], None, None, None)
            pass
        elif server_message.get("type") == "snapshots":
            my_print("Received server message: ", "others snapshot")
        else:
            my_print("Received server message: ", server_message)

        if server_message.get("type") == "exit":
            raise Exception("Exit received")
        return server_message
    

    def move_to(self, dst):
        # TODO: Maybe read from message in between, to make sure you are not exitted
        while True:
            message = {"type": "get_pose"}
            t = time.time()
            self.send_msg(message)
            response = self.read_msg(blocking=True)
            my_print(
                "Loc: {},Dst: {}".format(
                    response,  dst
                )
            )
            xy = response["xy"]
            heading = response["heading"]

            distance = math.sqrt((dst[1] - xy[1]) ** 2 + (dst[0] - xy[0]) ** 2)
            # my_print("distance is: ", distance)
            if distance < ACCEPTANCE_DISTANCE:
                my_print("Dist less than threshold, STOP.")
                rollereye.stop_move() #8
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
            control_speed = min(Kp * distance, 1, 0.15)
            # my_print("control is: ", control_speed, direction)
            rollereye.set_translate_4(direction, control_speed)  #9


    def get_location(self):
        self.send_msg({"type": "pose"})
        return self.read_msg(blocking=True)

    def run(self):
        if self.real_world:
            rollereye.start()  #10
            rollereye.timerStart()   #11
        step = 1
        while True:
            my_print("="*60)
            my_print("Agent {}, step: {}".format(self.rid, step))
            step += 1
            my_print("="*60)
            msg = self.read_msg(blocking=True)

            if msg['type'] == 'request_snapshot':
                # Send snapshot to server
                self.send_snapshot()
                
            elif msg['type'] == 'shutdown':
                break
            else:
                raise Exception("Was waiting for request_snapshot but got: {}".format(msg['type']))

            if self.is_complete:
                my_print("Agent {} already completed mission, ignoring further commands.".format(self.rid))
                        
                continue

            other_snapshots = self.receive_snapshots()
            
            if other_snapshots:
                my_print("Agent {} received snapshots from: {}".format(self.rid, list(other_snapshots.keys())))
                
                updates = self.agent.prepare_communication_updates_decentralized(other_snapshots)
                self.agent.apply_communication_updates_decentralized(updates)
                
                learned_from = updates.get('learned_from', {})
                new_nodes = updates['new_visited']
                if len(new_nodes) > 0:
                    breakdown = ', '.join(["{} from Agent {}".format(count, aid) 
                                          for aid, count in learned_from.items() if count > 0])
                    my_print("Learned {} new nodes ({})".format(len(new_nodes), breakdown))

            step_msg = self.read_msg(blocking=True)
            
            if step_msg['type'] != 'step':
                raise Exception("Was waiting for step command but got: {}".format(step_msg['type']))
            
            
            self.agent.update_product_automaton()
            
            accepting_path = self.agent.check_mission_complete()
            if accepting_path is not None and len(accepting_path) > 1:
                my_print("Agent {}: Executing final accepting path: {}".format(
                    self.rid, accepting_path), None, None, None, None)
                self.agent.current_plan     = accepting_path
                self.agent.current_frontier = accepting_path[-1]
                self.agent.mission_finishing = True
            elif accepting_path is not None and len(accepting_path) <= 1:
                my_print("Agent {} mission complete (already at goal)!".format(self.rid))
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
                    my_print("Agent {}: Cannot proceed further.".format(self.rid))
                    self.send_complete()
                    continue
                else:
                    my_print("Agent {} planning to move to frontier {} ({} steps)".format(
                        self.rid, best_frontier, len(path)))
                    self.agent.current_plan = path
                    self.agent.current_frontier = best_frontier
            
            my_print("Before move")
            next_position = self.agent.current_plan[1]
            self.agent.move_one_step(next_position)
            coord = self.env.get_node_coordinates(next_position)
            my_print("Converted {} to {}".format(next_position, coord))
            if self.real_world:
                self.move_to(coord)
            self.agent.current_plan = self.agent.current_plan[1:]
            
            my_print("  Agent {} moved to {} ({} steps remaining)".format(
                self.rid, self.agent.current_physical_state, len(self.agent.current_plan)-1))
            
            # if len(self.agent.current_plan) == 1:
            #     my_print("    Agent {} reached frontier {}".format(self.rid, self.agent.current_frontier), 
            #             None, None, None, None)
            #     self.agent.current_plan = None
            #added for fully following the path
            if len(self.agent.current_plan) == 1:
                my_print("    Agent {} reached frontier {}".format(
                    self.rid, self.agent.current_frontier))
                self.agent.current_plan = None
                if getattr(self.agent, 'mission_finishing', False):
                    my_print("Agent {} mission complete!".format(self.rid))
                    self.send_complete()
                    self.agent.mission_finishing = False
                    continue

            self.send_ready()
        
        my_print("="*60)
        my_print("Agent {} - FINAL RESULTS".format(self.rid))
        my_print("="*60)
        my_print("Final position: {}".format(self.agent.current_physical_state))
        my_print("Trajectory length: {}".format(len(self.agent.full_physical_traj)))
        my_print("Trajectory: {}".format(self.agent.full_physical_traj))

    def exit(self):
        # rollereye.handle_exception(e.__class__.__name__ + ": " + e.message)  #6
        if self.real_world:
            rollereye.stop()
        self.socket.close()

    def send_snapshot(self):
        snapshot = {
            'position': self.agent.current_physical_state,
            'visited': list(self.agent.visited),
            'node_labels': {k: list(v) for k, v in self.agent.node_labels.items()}
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
        my_print("sending ready")
        message = {
            'type': 'ready',
        }
        self.send_msg(message)
        my_print("Sent ready")

    def receive_snapshots(self):
        msg = self.read_msg()
        
        if msg['type'] == 'snapshots':
            agents_in_range = msg['agents_in_range']
            
            for other_id, snapshot in agents_in_range.items():
                snapshot['visited'] = set(snapshot['visited'])
                snapshot['node_labels'] = {
                    int(k): set(v) for k, v in snapshot['node_labels'].items()
                }
            
            return agents_in_range
        else: 
            raise Exception("Expected 'snapshots' message but got {}".format(msg['type']))


if __name__ == "__main__":
    # SERVER_IP = "localhost"
    SERVER_IP = "192.168.1.58"  #5 
    SERVER_PORT = 5000
    
    if len(sys.argv) > 1:
        SERVER_IP = sys.argv[1] 
    if len(sys.argv) > 2:
        SERVER_PORT = int(sys.argv[2])
    
    client_socket = socket.socket()
    client_socket.connect((SERVER_IP, SERVER_PORT))

    moorebot = Moorebot(client_socket, REAL_WORLD)

    try:
        moorebot.run()
    except Exception as e:
        raise e
        my_print("An error here: ", e)
        pass
    moorebot.exit()