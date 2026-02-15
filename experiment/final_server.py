# server.py
import socket
import threading
import json
import time

class SyncServer:
    def __init__(self, host='0.0.0.0', port=5555, num_agents=4):
        self.host = host
        self.port = port
        self.num_agents = num_agents
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(num_agents)
        
        self.clients = {}  # {agent_id: client_socket}
        self.client_ready = {}  # {agent_id: is_ready}
        self.client_complete = {}  # {agent_id: is_complete}
        self.agent_snapshots = {}  # {agent_id: snapshot_data}
        self.lock = threading.Lock()
        
        self.iteration = 0
        self.running = True
        
        print(f"🖥️  Server started on {self.host}:{self.port}")
        print(f"   Waiting for {num_agents} agents to connect...")
    
    def handle_client(self, client_socket, addr):
        """Handle communication with a single client."""
        agent_id = None
        
        try:
            # Receive agent registration
            data = client_socket.recv(1024).decode('utf-8')
            msg = json.loads(data)
            
            if msg['type'] == 'register':
                agent_id = msg['agent_id']
                
                with self.lock:
                    self.clients[agent_id] = client_socket
                    self.client_ready[agent_id] = False
                    self.client_complete[agent_id] = False
                    self.agent_snapshots[agent_id] = None
                
                # Send confirmation
                response = json.dumps({'type': 'registered', 'agent_id': agent_id})
                client_socket.send(response.encode('utf-8'))
                
                print(f"✅ Agent {agent_id} registered from {addr}")
                print(f"   Connected agents: {len(self.clients)}/{self.num_agents}")
                
                # Keep connection alive
                while self.running:
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"❌ Error with agent {agent_id}: {e}")
        finally:
            if agent_id is not None:
                with self.lock:
                    if agent_id in self.clients:
                        del self.clients[agent_id]
                    if agent_id in self.client_ready:
                        del self.client_ready[agent_id]
                    if agent_id in self.agent_snapshots:
                        del self.agent_snapshots[agent_id]
            client_socket.close()
    
    def wait_for_connections(self):
        """Wait for all agents to connect."""
        threads = []
        
        for i in range(self.num_agents):
            client_socket, addr = self.server_socket.accept()
            thread = threading.Thread(target=self.handle_client, args=(client_socket, addr))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        print(f"\n✅ All {self.num_agents} agents connected!")
        return True
    
    def collect_snapshots(self):
        """Collect snapshots from all active agents."""
        print("📸 Collecting snapshots from agents...")
        
        with self.lock:
            active_agents = [aid for aid, complete in self.client_complete.items() if not complete]
            # Clear old snapshots
            self.agent_snapshots = {aid: None for aid in self.clients.keys()}
        
        # Request snapshots
        request_msg = json.dumps({'type': 'request_snapshot'})
        
        with self.lock:
            for agent_id in active_agents:
                if agent_id in self.clients:
                    try:
                        self.clients[agent_id].send(request_msg.encode('utf-8'))
                    except Exception as e:
                        print(f"❌ Failed to request snapshot from agent {agent_id}: {e}")
        
        # Wait for all active agents to send snapshots
        timeout = 5  # 5 seconds timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                active_agents = [aid for aid, complete in self.client_complete.items() if not complete]
                all_snapshots_received = all(
                    self.agent_snapshots.get(aid) is not None 
                    for aid in active_agents
                )
            
            if all_snapshots_received:
                print(f"✅ Received snapshots from all {len(active_agents)} active agents")
                return True
            
            time.sleep(0.05)
        
        print("⚠️  Timeout waiting for snapshots")
        return False
    
    def broadcast_snapshots(self, com_range, grid_n):
        """Send relevant snapshots to each agent (only agents in comm range)."""
        print("📤 Broadcasting snapshots to agents...")
        
        with self.lock:
            snapshots_copy = dict(self.agent_snapshots)
        
        # For each agent, determine who is in range and send relevant snapshots
        for agent_id, snapshot in snapshots_copy.items():
            # print(snapshot)
            if snapshot is None or self.client_complete.get(agent_id, True):
                continue
            
            # Calculate agents in communication range
            agent_pos = snapshot['position']
            r1, c1 = agent_pos // grid_n, agent_pos % grid_n
            
            agents_in_range = {}
            for other_id, other_snapshot in snapshots_copy.items():
                if other_id == agent_id or other_snapshot is None:
                    continue
                
                other_pos = other_snapshot['position']
                r2, c2 = other_pos // grid_n, other_pos % grid_n
                dist = abs(r1 - r2) + abs(c1 - c2)
                
                if dist <= com_range:
                    agents_in_range[other_id] = other_snapshot
            
            # Send relevant snapshots to this agent
            message = json.dumps({
                'type': 'snapshots',
                'agents_in_range': agents_in_range
            })
            
            with self.lock:
                if agent_id in self.clients:
                    try:
                        self.clients[agent_id].send(message.encode('utf-8'))
                    except Exception as e:
                        print(f"❌ Failed to send snapshots to agent {agent_id}: {e}")
    
    def broadcast_step(self):
        """Broadcast 'step' command to all active agents."""
        self.iteration += 1
        
        message = json.dumps({
            'type': 'step',
            'iteration': self.iteration
        })
        
        with self.lock:
            active_agents = [aid for aid, complete in self.client_complete.items() if not complete]
            
            for agent_id in active_agents:
                if agent_id in self.clients:
                    try:
                        self.clients[agent_id].send(message.encode('utf-8'))
                    except Exception as e:
                        print(f"❌ Failed to send to agent {agent_id}: {e}")
    
    def wait_for_ready(self):
        """Wait for all active agents to signal they're ready."""
        with self.lock:
            active_agents = [aid for aid, complete in self.client_complete.items() if not complete]
            complete_agents = [aid for aid, complete in self.client_complete.items() if complete]
        
        if not active_agents:
            print("✅ All agents completed their missions!")
            return 'all_complete'
        
        print(f"⏳ Waiting for {len(active_agents)} active agents to complete iteration {self.iteration}...")
        if complete_agents:
            print(f"   ({len(complete_agents)} agents already complete: {complete_agents})")
        
        with self.lock:
            for agent_id in active_agents:
                self.client_ready[agent_id] = False
        
        while True:
            with self.lock:
                active_agents = [aid for aid, complete in self.client_complete.items() if not complete]
                
                if not active_agents:
                    print("✅ All agents completed their missions!")
                    return 'all_complete'
                
                all_active_ready = all(
                    self.client_ready.get(aid, False) 
                    for aid in active_agents
                )
            
            if all_active_ready:
                print(f"✅ All {len(active_agents)} active agents ready for iteration {self.iteration + 1}")
                return 'ready'
            
            time.sleep(0.1)
    
    def receive_messages(self):
        """Listen for messages from agents."""
        def listen_for_messages(agent_id, client_socket):
            try:
                while self.running:
                    data = client_socket.recv(65536).decode('utf-8')  # Larger buffer for snapshots
                    if not data:
                        break
                    
                    msg = json.loads(data)
                    
                    if msg['type'] == 'snapshot':
                        with self.lock:
                            self.agent_snapshots[agent_id] = msg['data']
                            print(f"   📸 Received snapshot from Agent {agent_id}")
                    
                    elif msg['type'] == 'ready':
                        with self.lock:
                            self.client_ready[agent_id] = True
                            if not self.client_complete[agent_id]:
                                print(f"   Agent {agent_id} ready")
                    
                    elif msg['type'] == 'complete':
                        with self.lock:
                            was_active = not self.client_complete[agent_id]
                            self.client_complete[agent_id] = True
                            self.client_ready[agent_id] = True
                            
                            if was_active:
                                active_count = sum(1 for c in self.client_complete.values() if not c)
                                print(f"✅ Agent {agent_id} mission complete! ({active_count} agents still active)")
            except Exception as e:
                print(f"Error receiving from agent {agent_id}: {e}")
        
        with self.lock:
            for agent_id, client_socket in self.clients.items():
                thread = threading.Thread(target=listen_for_messages, args=(agent_id, client_socket))
                thread.daemon = True
                thread.start()
    
    def run(self, max_iterations=1000, com_range=50, grid_n=20):
        """Main server loop."""
        if not self.wait_for_connections():
            return
        
        self.receive_messages()
        
        time.sleep(1)
        
        print("\n" + "="*60)
        print("🚀 STARTING SYNCHRONIZED EXECUTION")
        print("="*60)
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}")
            print(f"{'='*60}")
            
            # PHASE 1: Collect snapshots from all agents
            if not self.collect_snapshots():
                print("❌ Failed to collect snapshots, skipping iteration")
                continue
            
            # PHASE 2: Broadcast snapshots (filtered by comm range)
            self.broadcast_snapshots(com_range, grid_n)
            
            # PHASE 3: Send step command
            self.broadcast_step()
            
            # PHASE 4: Wait for all active agents to finish
            status = self.wait_for_ready()
            
            if status == 'all_complete':
                break
        
        print("\n✅ Simulation complete!")
        self.shutdown()
    
    def shutdown(self):
        """Shutdown server."""
        self.running = False
        
        message = json.dumps({'type': 'shutdown'})
        with self.lock:
            for client_socket in self.clients.values():
                try:
                    client_socket.send(message.encode('utf-8'))
                except:
                    pass
        
        self.server_socket.close()
        print("🛑 Server shutdown")


if __name__ == "__main__":
    import sys
    
    num_agents = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    com_range = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    grid_n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    
    server = SyncServer(num_agents=num_agents)
    server.run(max_iterations=max_iterations, com_range=com_range, grid_n=grid_n)