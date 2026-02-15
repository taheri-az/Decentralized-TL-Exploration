# client.py
from Helper import *
from classes import Environment, Agent
import socket
import json
import sys
import time

class AgentClient:
    def __init__(self, agent_id, server_host='localhost', server_port=5555):
        self.agent_id = agent_id
        self.server_host = server_host
        self.server_port = server_port
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((server_host, server_port))
        
        register_msg = json.dumps({
            'type': 'register',
            'agent_id': agent_id
        })
        self.socket.send(register_msg.encode('utf-8'))
        
        response = self.socket.recv(1024).decode('utf-8')
        msg = json.loads(response)
        
        if msg['type'] == 'registered':
            print(f"✅ Agent {agent_id} registered with server")
        
        self.running = True
        self.mission_complete = False
    
    def send_snapshot(self, agent):
        """Send snapshot of this agent's state to server."""
        snapshot = {
            'position': agent.current_physical_state,
            'visited': list(agent.visited),  # Convert set to list for JSON
            'node_labels': {k: list(v) for k, v in agent.node_labels.items()}  # Convert sets to lists
        }
        
        message = json.dumps({
            'type': 'snapshot',
            'agent_id': self.agent_id,
            'data': snapshot
        })
        
        self.socket.send(message.encode('utf-8'))
    
    def receive_snapshots(self):
        """Receive snapshots of agents in communication range from server."""
        data = self.socket.recv(65536).decode('utf-8')
        msg = json.loads(data)
        
        if msg['type'] == 'snapshots':
            # Convert lists back to sets
            agents_in_range = msg['agents_in_range']
            
            for other_id, snapshot in agents_in_range.items():
                snapshot['visited'] = set(snapshot['visited'])
                snapshot['node_labels'] = {
                    int(k): set(v) for k, v in snapshot['node_labels'].items()
                }
            
            return agents_in_range
        
        return {}
    
    def send_ready(self):
        """Signal ready to server."""
        message = json.dumps({
            'type': 'ready',
            'agent_id': self.agent_id
        })
        self.socket.send(message.encode('utf-8'))
    
    def send_complete(self):
        """Signal mission complete to server."""
        message = json.dumps({
            'type': 'complete',
            'agent_id': self.agent_id
        })
        self.socket.send(message.encode('utf-8'))
        self.mission_complete = True
    
    def wait_for_message(self):
        """Wait for message from server."""
        data = self.socket.recv(1024).decode('utf-8')
        msg = json.loads(data)
        return msg
    
    def close(self):
        """Close connection."""
        self.socket.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <agent_id> [server_host] [server_port]")
        sys.exit(1)
    
    agent_id = int(sys.argv[1])
    server_host = sys.argv[2] if len(sys.argv) > 2 else 'localhost'
    server_port = int(sys.argv[3]) if len(sys.argv) > 3 else 5555
    
    print(f"🤖 Starting Agent {agent_id}")
    
    # ============= LOAD CONFIGURATION =============
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
    
    # ============= LOAD AGENT CONFIGURATION =============
    try:
        with open('agents_config.json', 'r') as f:
            agents_configs = json.load(f)
        
        agent_config = None
        for config in agents_configs:
            if config['id'] == agent_id:
                agent_config = config
                break
        
        if agent_config is None:
            print(f"❌ Error: No configuration found for agent {agent_id}")
            sys.exit(1)
        
        initial_position = agent_config['start']
        formula_str = agent_config['mission']
        
        print(f"   Mission: {formula_str}")
        print(f"   Starting position: {initial_position}")
        
    except FileNotFoundError:
        print("❌ Error: agents_config.json not found!")
        sys.exit(1)
    
    # ============= CREATE AGENT =============
    agent = Agent(
        agent_id=agent_id,
        initial_position=initial_position,
        formula_str=formula_str,
        env=env,
        h=h,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3
    )
    
    # ============= CONNECT TO SERVER =============
    client = AgentClient(agent_id, server_host, server_port)
    
    print(f"✅ Agent {agent_id} initialized and waiting for synchronization...")
    
    # ============= MAIN LOOP =============
    while client.running:
        # Wait for snapshot request
        msg = client.wait_for_message()
        
        if msg['type'] == 'request_snapshot':
            # Send snapshot to server
            client.send_snapshot(agent)
            
            # Receive snapshots from server
            other_snapshots = client.receive_snapshots()
            
            if other_snapshots:
                print(f"\n📡 Agent {agent_id} received snapshots from: {list(other_snapshots.keys())}")
                
                # Apply communication updates
                updates = agent.prepare_communication_updates_decentralized(other_snapshots)
                agent.apply_communication_updates_decentralized(updates)
                
                learned_from = updates.get('learned_from', {})
                new_nodes = updates['new_visited']
                if len(new_nodes) > 0:
                    breakdown = ', '.join([f"{count} from Agent {aid}" for aid, count in learned_from.items() if count > 0])
                    print(f"🔄 Learned {len(new_nodes)} new nodes ({breakdown})")
            
            # Now wait for step command
            step_msg = client.wait_for_message()
            
            if step_msg['type'] == 'shutdown':
                client.running = False
                break
            
            if step_msg['type'] != 'step':
                continue
            
            iteration = step_msg['iteration']
            
        elif msg['type'] == 'shutdown':
            client.running = False
            break
        else:
            continue
        
        print(f"\n{'='*60}")
        print(f"Agent {agent_id} - Iteration {iteration}")
        print(f"{'='*60}")
        
        # ===== EXECUTION PHASE =====
        agent.update_product_automaton()
        
        if agent.check_mission_complete():
            print(f"✅ Agent {agent_id} mission complete!")
            client.send_complete()
            continue
        
        # Plan or continue existing plan
        if not hasattr(agent, 'current_plan') or agent.current_plan is None or len(agent.current_plan) == 0:
            # Build agents_in_range from snapshots for frontier selection
            agents_in_range_positions = {
                int(aid): snapshot['position'] 
                for aid, snapshot in other_snapshots.items()
            }
            
            best_frontier, path = agent.select_frontier(agents_in_range=agents_in_range_positions)
            
            if best_frontier is None or path is None:
                print(f"❌ Agent {agent_id}: Cannot proceed further.")
                client.send_complete()
                continue
            else:
                print(f"🚀 Agent {agent_id} planning to move to frontier {best_frontier} ({len(path)} steps)")
                agent.current_plan = path
                agent.current_frontier = best_frontier
        
        # Execute ONE step
        if hasattr(agent, 'current_plan') and agent.current_plan is not None and len(agent.current_plan) > 1:
            next_position = agent.current_plan[1]
            agent.move_one_step(next_position)
            agent.current_plan = agent.current_plan[1:]
            
            print(f"  Agent {agent_id} moved to {agent.current_physical_state} ({len(agent.current_plan)-1} steps remaining)")
            
            if len(agent.current_plan) == 1:
                print(f"    Agent {agent_id} reached frontier {agent.current_frontier}")
                agent.current_plan = None
        
        # Signal ready
        client.send_ready()
    
    # ============= CLEANUP =============
    client.close()
    
    print(f"\n{'='*60}")
    print(f"Agent {agent_id} - FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Mission: {agent.formula_str}")
    print(f"Final position: {agent.current_physical_state}")
    print(f"Trajectory length: {len(agent.full_physical_traj)}")
    print(f"Trajectory: {agent.full_physical_traj}")



if __name__ == "__main__":
    main()