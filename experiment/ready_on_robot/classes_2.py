# -*- coding: utf-8 -*-
from client_helper import *
import copy
import random
import numpy as np

class Environment:
    """Shared environment for all agents"""
    def __init__(self, n, m, node_labels_t):
        self.n = n
        self.m = m
        self.grid = create_grid(n, m)
        self.nodes, self.edges, self.adj_matrix = create_graph(n, m)
        self.node_labels_t = node_labels_t
        
        # Initialize labels for all nodes
        for node in range(n * m):
            if node not in self.node_labels_t:
                self.node_labels_t[node] = set()

    def get_node_coordinates(self, node_index):
        """Convert node index to (x, y) coordinates for robot waypoints."""
        row = node_index // self.m
        col = node_index % self.m

        # Adjust these values based on your Webots world setup
        cell_size = 0.25  # meters per grid cell
        offset_x = 0.5   # world origin offset
        offset_y = 0.5  # Start from top (Y=10)

        x = col * cell_size + offset_x
        y = offset_y + (row * cell_size)  # SUBTRACT to make Y decrease as row increases
        return (y, x)


class Agent:
    def __init__(self, agent_id, initial_position, dfa_transitions, initial_state, trash_states_set, commit_states, env, h, alpha1, alpha2, alpha3):
        """
        Initialize an agent with its own mission and DFA.
        
        Parameters:
        -----------
        agent_id : int
            Unique identifier for the agent
        initial_position : int
            Starting physical state
        formula_str : str
            LTL formula for this agent's mission
        env : Environment
            Shared environment object
        h : int
            Sensing horizon
        alpha1, alpha2, alpha3 : float
            Weight parameters
        """
        self.id = agent_id
        self.env = env  # Reference to shared environment
        self.h = h
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

        self.dfa_transitions = dfa_transitions
        self.initial_state = initial_state
        self.trash_states_set = trash_states_set
        self.commit_states = commit_states
        
        # Extract atomic props from DFA transitions AFTER dfa_transitions is created
        self.atomic_props = extract_atomic_props_from_dfa(self.dfa_transitions)
        print("Agent {} atomic props: {}".format(agent_id, self.atomic_props))
        
        # Now prune using atomic_props
        self.pruned_dfa_transitions = prune_dfa_transitions_single_ap_only(
            self.dfa_transitions, self.atomic_props
        )
        
        # Extract DFA states
        self.dfa_states = set()
        for s1, _, s2 in self.dfa_transitions:
            self.dfa_states.update([s1, s2])
        
        self.accepting_states = set(['accept_all'])
        self.trash_state = 'Trash'
        
        # Filter commit states
        self.commit_states = [s for s in self.commit_states if s not in self.trash_states_set]
        self.commit_states = [str(s) for s in self.commit_states]
        
        # Compute DFA distances
        self.dfa_distances = compute_dfa_distances_to_accepting(
            self.dfa_states, 
            self.pruned_dfa_transitions, 
            self.accepting_states
        )
        
        # State tracking
        self.current_physical_state = initial_position
        self.current_dfa_state = self.initial_state
        
        self.visited = set()
        self.node_labels = {}  # Agent's known labels
        
        # Trajectory
        self.full_traj = []
        self.full_physical_traj = [self.current_physical_state]
        
        # Initialize node labels (all unknown initially)
        for node in range(env.n * env.m):
            self.node_labels[node] = set()
        
        # Product automaton
        self.product_graph = None
        self.transitions = None
        self.product_nodes = None
        self.node_to_index = None
        self.PR_adj_matrix = None
        
        # Initialize with h-neighborhood
        self._initialize_knowledge()
    
    def _initialize_knowledge(self):
        """Initialize agent's knowledge with h-neighborhood of starting position."""
        h_neighbors = get_states_within_h_distance(self.env.m, self.env.n, 
                                                     self.current_physical_state, self.h)
        
        # Update labels and visited for initial neighborhood
        for node in h_neighbors:
            self.node_labels[node] = self.env.node_labels_t.get(node, set())
            self.visited.add(node)
        
        # Build initial product automaton
        initial_nodes_set = set(h_neighbors)
        initial_edges = self._filter_edges(self.env.edges, initial_nodes_set)
        initial_labels_known = {v: self.node_labels[v] for v in initial_nodes_set}
        initial_nodes_array = np.array(list(initial_nodes_set))
        
        # Pass atomic_props to the function
        self.product_graph, self.transitions, self.product_nodes, self.PR_adj_matrix = \
            generate_product_automaton22(
                initial_nodes_array, initial_edges, self.dfa_states, 
                self.dfa_transitions, initial_labels_known, self.atomic_props
            )
        
        self.node_to_index = {node: idx for idx, node in enumerate(self.product_nodes)}
        self.visited_old = copy.deepcopy(self.visited)
    
    def _filter_edges(self, all_edges, known_nodes_set):
        """Filter edges to only include those between known nodes."""
        new_edges = []
        for u, v in all_edges:
            if u in known_nodes_set and v in known_nodes_set:
                new_edges.append((u, v))
        return new_edges
    
    def update_knowledge(self):
        """Update agent's knowledge based on current position."""
        h_neighbors = get_states_within_h_distance(self.env.m, self.env.n, 
                                                     self.current_physical_state, self.h)
        
        for node in h_neighbors:
            self.node_labels[node] = self.env.node_labels_t.get(node, set())
            self.visited.add(node)
    
    def update_product_automaton(self):
        """Update product automaton with newly discovered nodes."""
        new_nodes = self.visited - self.visited_old
        
        if len(new_nodes) > 0:
            self.product_graph, self.transitions, self.product_nodes, \
            self.node_to_index, self.PR_adj_matrix = \
                update_product_automaton_incremental(
                    self.product_graph,
                    self.transitions,
                    self.product_nodes,
                    self.node_to_index,
                    self.PR_adj_matrix,
                    new_nodes,
                    self.env.edges,
                    self.dfa_states,
                    self.dfa_transitions,
                    self.node_labels,
                    self.atomic_props
                )
    
    # def check_mission_complete(self):
    #     """Check if mission can be completed from current state."""
    #     current_product_state = (self.current_physical_state, self.current_dfa_state)
    #     accepting_path = find_shortest_path_to_accepting(
    #         current_product_state, 
    #         self.accepting_states, 
    #         self.transitions
    #     )
        
    #     if accepting_path:
    #         print("[OK] Agent {}: Accepting path found! Executing...".format(self.id))
    #         for state in accepting_path[1:]:
    #             self.current_physical_state, self.current_dfa_state = state
    #             self.full_traj.append(state)
    #             self.full_physical_traj.append(int(self.current_physical_state))
    #             print("Agent {} -> {}".format(self.id, state))
    #         return True
    #     return False
    # added for fully following the path
    def check_mission_complete(self):
        """
        Check if mission can be completed from current state.
        Returns physical path (list) if found, else None.
        Does NOT modify agent state.
        """
        current_product_state = (self.current_physical_state, self.current_dfa_state)
        accepting_path = find_shortest_path_to_accepting(
            current_product_state,
            self.accepting_states,
            self.transitions
        )

        if accepting_path:
            physical_path = [int(s) for (s, q) in accepting_path]
            print("[OK] Agent {}: Accepting path found: {}".format(self.id, physical_path))
            return physical_path

        return None
    
    def select_frontier(self, agents_in_range=None):
        """Select best frontier to explore."""
        unknown = set(range(self.env.n * self.env.m)) - self.visited
        frontiers = detect_frontiers_e(self.env.n, self.env.m, self.visited, unknown)
        
        if not frontiers:
            print("[ERROR] Agent {}: No frontiers left.".format(self.id))
            return None, None
        
        # Compute information gain for each frontier
        I_x_dict = {}
        for x in frontiers:
            revealed = get_states_within_h_distance(self.env.m, self.env.n, x, self.h)
            base_info_gain = len(set(revealed) - self.visited)
            
            # If in communication range with other agents, apply distance penalty
            if agents_in_range and len(agents_in_range) > 0:
                dist_self_to_frontier = dis_to_frontier(self.env.m, self.env.n, 
                                                        self.current_physical_state, x)
                if dist_self_to_frontier > 0:
                    min_dist = min_dist_to_frontier(
                        self.env.m, self.env.n, 
                        self.current_physical_state, 
                        agents_in_range, 
                        x
                    )
                    dis_penalty = min_dist / float(dist_self_to_frontier)
                    norm_dis_penalty = dis_penalty / (1 + dis_penalty)
                    I_x_dict[x] = base_info_gain * norm_dis_penalty 
                else:
                    I_x_dict[x] = base_info_gain
            else:
                I_x_dict[x] = base_info_gain
        
        # Score frontiers
        weights = {}
        best_paths = {}
        for x in frontiers:
            w, sp = compute_frontier_commit(
                x=x,
                product_graph=self.product_graph,
                start_cell=self.current_physical_state,
                start_dfa_state=self.current_dfa_state,
                accepting_states=self.accepting_states,
                commit_states=self.commit_states,
                trash_state=self.trash_state,
                delta_phi=delta_phi,
                I_x=I_x_dict[x],
                X_size=self.env.n * self.env.m,
                dfa_distance=self.dfa_distances,
                alpha1=self.alpha1,
                alpha2=self.alpha2,
                alpha3=self.alpha3
            )
            weights[x] = w
            best_paths[x] = sp
        
        valid_frontiers = [x for x in frontiers if best_paths[x] is not None]
        if not valid_frontiers:
            print("Agent {}: No reachable frontiers left.".format(self.id))
            return None, None
        
        max_weight = max(weights[x] for x in valid_frontiers)
        best_frontiers = [x for x in valid_frontiers if weights[x] == max_weight]
        print("vets_l {}".format(best_frontiers))
        best_frontier = random.choice(best_frontiers)
        print("vets {}".format(best_frontier))
        path = [s for (s, q) in best_paths[best_frontier]]
        
        return best_frontier, path
    
    def move_along_path(self, path):
        """Move agent along the given path."""
        self.visited_old = copy.deepcopy(self.visited)
        
        for step in path[1:]:
            label = self.node_labels.get(step, set())
            self.current_dfa_state = get_next_dfa_state(
                self.current_dfa_state, label, self.dfa_transitions, self.atomic_props
            )
            self.current_physical_state = step
            self.update_knowledge()
            self.full_traj.append((step, self.current_dfa_state))
            self.full_physical_traj.append(step)
    
    def prepare_communication_updates_decentralized(self, other_agents_snapshots):
        """
        PHASE 1: Prepare updates based on snapshots of other agents.
        Does NOT modify this agent's state.
        
        Parameters:
        -----------
        other_agents_snapshots : dict
            {agent_id: {'visited': set, 'node_labels': dict}}
        
        Returns:
        --------
        dict : {
            'new_visited': set,
            'new_labels': dict,
            'learned_from': dict  # tracks what was learned from each agent
        }
        """
        new_nodes_total = set()
        new_labels_total = {}
        learned_from = {}
        nodes_by_source = {}
        
        # My current knowledge at the start of this timestep
        my_visited = self.visited
        
        for other_id, other_snapshot in other_agents_snapshots.items():
            # What nodes does the other agent know that I don't?
            new_from_other = other_snapshot['visited'] - my_visited
            
            # Store these nodes for this source
            nodes_by_source[other_id] = new_from_other
            
            # Track count for this agent (including overlaps with other agents)
            learned_from[other_id] = len(new_from_other)
            
            # Add to total
            new_nodes_total.update(new_from_other)
            
            # Get labels for those nodes
            for node in new_from_other:
                if node in other_snapshot['node_labels']:
                    if node not in new_labels_total:
                        new_labels_total[node] = set()
                    new_labels_total[node].update(other_snapshot['node_labels'][node])
        
        return {
            'new_visited': new_nodes_total,
            'new_labels': new_labels_total,
            'learned_from': learned_from
        }
    
    def apply_communication_updates_decentralized(self, updates):
        """
        PHASE 2: Apply the prepared updates to this agent.
        
        Parameters:
        -----------
        updates : dict
            Updates from prepare_communication_updates_decentralized
        """
        new_nodes = updates['new_visited']
        new_labels = updates['new_labels']
        
        if len(new_nodes) == 0:
            return
        
        # Update visited
        self.visited.update(new_nodes)
        
        # Update labels
        for node, labels in new_labels.items():
            self.node_labels[node] = labels
        
        # Update product automaton
        self.product_graph, self.transitions, self.product_nodes, \
        self.node_to_index, self.PR_adj_matrix = \
            update_product_automaton_incremental(
                self.product_graph,
                self.transitions,
                self.product_nodes,
                self.node_to_index,
                self.PR_adj_matrix,
                new_nodes,
                self.env.edges,
                self.dfa_states,
                self.dfa_transitions,
                self.node_labels,
                self.atomic_props
            )
    
    def step(self, agents_in_range=None):
        """Execute one step of the agent's exploration."""
        print("\n[AGENT] Agent {} - Current state: ({}, {})".format(
            self.id, self.current_physical_state, self.current_dfa_state))
        
        self.update_product_automaton()
        
        if self.check_mission_complete():
            return True
        
        best_frontier, path = self.select_frontier(agents_in_range)
        
        if best_frontier is None or path is None:
            print("Agent {}: Cannot proceed further.".format(self.id))
            return True
        
        print("[START] Agent {} moving to frontier {}".format(self.id, best_frontier))
        self.move_along_path(path)
        
        return False
    
    def move_one_step(self, next_position):
        """
        Move agent one step to the next position.
        Updates knowledge and trajectory.
        
        Parameters:
        -----------
        next_position : int
            The next physical position to move to
        """
        # Get label at next position
        label = self.node_labels.get(next_position, set())
        
        # Update DFA state
        self.current_dfa_state = get_next_dfa_state(
            self.current_dfa_state, label, self.dfa_transitions, self.atomic_props
        )
        
        # Move to next position
        self.current_physical_state = next_position
        
        # Update knowledge at new position
        self.visited_old = copy.deepcopy(self.visited)
        self.update_knowledge()
        
        # Record in trajectory
        self.full_traj.append((next_position, self.current_dfa_state))
        self.full_physical_traj.append(next_position)