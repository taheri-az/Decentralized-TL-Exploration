from Helper import *
from classes import Environment, Agent
import copy
import numpy as np
import json
import time

start_time = time.time()



# ============= MAIN EXECUTION =============
try:
    config = load_config('config.json')
except FileNotFoundError:
    print("❌ Error: config.json not found!")
    print("Please create config.json in the same directory as this script.")
    exit(1)
except Exception as e:
    print(f"❌ Error loading configuration: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Extract parameters from config
n = config['n']
m = config['m']
h = config['h']
com_range = config['com_range']
alpha1 = config['alpha1']
alpha2 = config['alpha2']
alpha3 = config['alpha3']
node_labels_t = config['node_labels_t']


# Create shared environment
env = Environment(n, m, node_labels_t)

try:
    agents = load_agents_from_json('agents_config.json', env, h, alpha1, alpha2, alpha3, Agent)
except FileNotFoundError:
    print("❌ Error: agents_config.json not found!")
    exit(1)
except Exception as e:
    print(f"❌ Error loading agents: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
# Main loop
# Initialize completion tracking
agents_complete = {agent_id: False for agent_id in agents.keys()}

# ============= MAIN LOOP =============
max_iterations = 1000



for iteration in range(1, max_iterations + 1):
    print(f"\n{'='*60}\nIteration {iteration}\n{'='*60}")
    
    # ===== DECENTRALIZED COMMUNICATION PHASE =====
    print("\n📡 Decentralized Communication Phase:")
    
    # PHASE 1: Take snapshots of ALL agents simultaneously
    snapshots = {}
    for agent_id, agent in agents.items():
        snapshots[agent_id] = {
            'visited': agent.visited.copy(),
            'node_labels': {k: v.copy() for k, v in agent.node_labels.items()}
        }
  

    # PHASE 2: Each agent prepares its updates (using snapshots)
    prepared_updates = {}
    for agent_id, agent in agents.items():
        # Find who this agent can communicate with
        in_range = agents_in_comm_range(env, agent_id, agents, com_range)
        
        if in_range:
            # Get snapshots of agents in range
            relevant_snapshots = {other_id: snapshots[other_id] for other_id in in_range.keys()}
            
            # Prepare updates based on snapshots
            prepared_updates[agent_id] = agent.prepare_communication_updates_decentralized(relevant_snapshots)
            
            print(f"📡 Agent {agent_id} can communicate with agents {list(in_range.keys())}")
        else:
            prepared_updates[agent_id] = {
                'new_visited': set(), 
                'new_labels': {},
                'learned_from': {}
            }
    
    # PHASE 3: Apply all updates simultaneously and print details
    print()  
    for agent_id, agent in agents.items():
        updates = prepared_updates[agent_id]
        new_nodes = updates['new_visited']
        learned_from = updates.get('learned_from', {})
        
        agent.apply_communication_updates_decentralized(updates)
        
        # Print detailed breakdown HERE
        if len(new_nodes) > 0:
            breakdown = ', '.join([f"{count} from Agent {aid}" for aid, count in learned_from.items() if count > 0])
            print(f"🔄 Agent {agent_id} learned {len(new_nodes)} new nodes ({breakdown})")
    
    # ===== EXECUTION PHASE (TWO-PHASE) =====
    print("\n🎯 Execution Phase:")
    
    # PHASE 1: Take snapshot of current agent positions
    agent_positions_snapshot = {aid: agent.current_physical_state for aid, agent in agents.items()}
    
    # PHASE 2: Each agent plans based on SNAPSHOT positions
    planned_actions = {}
    for agent_id, agent in agents.items():
        if not agents_complete[agent_id]:
            # Calculate who is in range based on SNAPSHOT positions
            in_range_snapshot = {}
            agent1_pos = agent_positions_snapshot[agent_id]
            r1, c1 = to_rc(agent1_pos, env.n)
            
            for other_id, other_pos in agent_positions_snapshot.items():
                if other_id == agent_id:
                    continue
                r2, c2 = to_rc(other_pos, env.n)
                dist = abs(r1 - r2) + abs(c1 - c2)
                if dist <= com_range:
                    in_range_snapshot[other_id] = other_pos
            
            # Agent plans using snapshot of other agents' positions
            # Update product automaton
            agent.update_product_automaton()
            
            # Check mission complete
            if agent.check_mission_complete():
                agents_complete[agent_id] = True
                planned_actions[agent_id] = None  # Mission complete, no action
                continue
            
            # Select frontier based on snapshot positions
            best_frontier, path = agent.select_frontier(agents_in_range=in_range_snapshot)
            
            if best_frontier is None or path is None:
                print(f"Agent {agent_id}: Cannot proceed further.")
                agents_complete[agent_id] = True
                planned_actions[agent_id] = None
            else:
                print(f"🚀 Agent {agent_id} planning to move to frontier {best_frontier}")
                planned_actions[agent_id] = path
        else:
            planned_actions[agent_id] = None
    
    # PHASE 3: Execute all planned actions simultaneously
    print("\n🏃 Executing movements:")
    for agent_id, agent in agents.items():
        path = planned_actions[agent_id]
        if path is not None:
            agent.move_along_path(path)
            print(f"  Agent {agent_id} moved to {agent.current_physical_state}")
    
    # Check completion
    if all(agents_complete.values()):
        print("\n✅ All agents completed their missions!")
        break
# ============= RESULTS =============
end_time = time.time()
print("simulation_time", end_time - start_time)
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

for agent_id, agent in agents.items():
    print(f"\nAgent {agent_id}:")
    print(f"  Mission: {agent.formula_str}")
    print(f"  Starting position: {agent.full_physical_traj[0]}")
    print(f"  Final position: {agent.current_physical_state}")
    print(f"  Trajectory length: {len(agent.full_physical_traj)}")
    print(f"  Mission complete: {agents_complete[agent_id]}")


for agent_id, agent in agents.items():
    print(f"\nAgent {agent_id} - Mission: {agent.formula_str}")
    print(f"Final trajectory: {agent.full_physical_traj}")
    print(f"Trajectory length: {len(agent.full_physical_traj)}")    