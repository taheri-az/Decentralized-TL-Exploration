
import json
import numpy as np 
import re
from collections import deque  
import networkx as nx  
def load_config(config_file='config.json'):
    """
    Load simulation configuration from JSON file.
    
    Parameters:
    -----------
    config_file : str
        Path to configuration file
    
    Returns:
    --------
    dict : Configuration dictionary with keys:
        - n, m, h, com_range (grid parameters)
        - alpha1, alpha2, alpha3 (weight parameters)
        - node_labels_t (node labels dictionary)
    """
    print("Loading configuration from {}...".format(config_file))
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract grid parameters
    grid = config['grid']
    n = grid['n']
    m = grid['m']
    h = grid['h']
    com_range = grid['com_range']
    
    # Extract weight parameters
    weights = config['weights']
    alpha1 = weights['alpha1']
    alpha2 = weights['alpha2']
    alpha3 = weights['alpha3']
    
    # Convert node labels from JSON format to Python format
    # JSON keys are strings, convert them to integers
    # JSON arrays become Python sets
    node_labels_raw = config['node_labels']
    node_labels_t = {}
    for node_str, labels_list in node_labels_raw.items():
        node_id = int(node_str)
        node_labels_t[node_id] = set(labels_list)
    
    print("[OK] Configuration loaded:")
    print("   Grid: {}x{}, h={}, com_range={}".format(n, m, h, com_range))
    print("   Weights: alpha1={}, alpha2={}, alpha3={}".format(alpha1, alpha2, alpha3))
    print("   Node labels: {} nodes with labels".format(len(node_labels_t)))
    
    return {
        'n': n,
        'm': m,
        'h': h,
        'com_range': com_range,
        'alpha1': alpha1,
        'alpha2': alpha2,
        'alpha3': alpha3,
        'node_labels_t': node_labels_t
    }



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



def create_grid(n, m):
    grid = np.zeros((n, m), dtype=int)
    return grid


def create_graph(m, n):
    num_nodes = n * m
    nodes = np.arange(num_nodes).reshape(n, m)
    edges = []

    # Add horizontal edges
    for i in range(n):
        for j in range(m-1):
            edges.append((nodes[i][j], nodes[i][j+1]))
            edges.append((nodes[i][j+1], nodes[i][j]))

    # Add vertical edges
    for j in range(m):
        for i in range(n-1):
            edges.append((nodes[i][j], nodes[i+1][j]))
            edges.append((nodes[i+1][j], nodes[i][j]))

    # Add stay action
    for i in range(n):
        for j in range(m):
            edges.append((nodes[i][j], nodes[i][j]))

    # Create the adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in edges:
        adj_matrix[edge[0], edge[1]] = 1
        adj_matrix[edge[1], edge[0]] = 1

    return nodes, edges, adj_matrix




def extract_atomic_props_from_dfa(dfa_transitions):
    """
    Extract atomic propositions from DFA transitions in the order they appear.
    Looks at the first transition formula to determine the order.
    """
    
    if not dfa_transitions:
        return []
    
    # Get the first transition formula to determine order
    first_formula = dfa_transitions[0][1][0]  # e.g., '!s && !p && !d'
    
    # Extract all tokens (letters/words)
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', first_formula)
    
    # Common LTL keywords and operators to exclude
    keywords = set([
        'X', 'G', 'F', 'U', 'R', 'W', 'M', 'true', 'false',
        'True', 'False', 'not', 'and', 'or'
    ])
    
    # Keep order as they appear in the formula
    atomic_props = []
    seen = set()
    for tok in tokens:
        if tok not in keywords and tok not in seen:
            atomic_props.append(tok)
            seen.add(tok)
    
    return atomic_props




def prune_dfa_transitions_single_ap_only(dfa_transitions, atomic_props):
    """
    Filters DFA transitions to keep only those where exactly one atomic proposition is True.

    Args:
        dfa_transitions (list of tuples): Each tuple is (current_state, [condition], next_state)
        atomic_props (list of str): List of atomic propositions, e.g., ['s', 'p', 'd']

    Returns:
        list of tuples: Filtered DFA transitions
    """
    pruned_transitions = []

    for (current_state, conditions, next_state) in dfa_transitions:
        for cond in conditions:
            # Count number of atomic props that are True in this condition
            true_count = 0
            for ap in atomic_props:
                # Check if atomic proposition appears positively (without !)
                if '{}'.format(ap) in cond and '!{}'.format(ap) not in cond:
                    true_count += 1
            if true_count == 1:
                pruned_transitions.append((current_state, [cond], next_state))

    return pruned_transitions



def compute_dfa_distances_to_accepting(dfa_states, dfa_transitions, accepting_states):
    """
    Compute the shortest distance (in number of transitions) from each DFA state
    to the nearest accepting state, using a list of transitions.

    Parameters
    ----------
    dfa_states : list
        List of DFA states.
    dfa_transitions : list
        List of tuples (q, [formula], q_next)
    accepting_states : set or list
        Accepting DFA states.

    Returns
    -------
    dfa_distances : dict
        Mapping: state -> distance to nearest accepting state (int or float('inf') if unreachable)
    """

    # Initialize distances: all infinity
    dfa_distances = {q: float('inf') for q in dfa_states}

    # BFS queue starting from all accepting states
    queue = deque()
    for acc in accepting_states:
        if acc in dfa_states:
            dfa_distances[acc] = 0
            queue.append(acc)

    # Build a reverse adjacency map: next_state -> list of previous states
    reverse_adj = {q: [] for q in dfa_states}
    for q, obs_list, q_next in dfa_transitions:
        reverse_adj[q_next].append(q)

    # BFS over the reverse DFA graph
    while queue:
        current = queue.popleft()
        current_dist = dfa_distances[current]

        for prev in reverse_adj[current]:
            if dfa_distances[prev] == float('inf'):
                dfa_distances[prev] = current_dist + 1
                queue.append(prev)

    return dfa_distances


def get_states_within_h_distance(m, n, current_state, h):
    # Function to convert state number to row and column
    def state_to_row_col(state):
        return divmod(state, n)

    # Function to convert row and column to state number
    def row_col_to_state(row, col):
        if 0 <= row < m and 0 <= col < n:
            return row * n + col
        return None

    # Function to get all adjacent states of a given state
    def get_adjacent_states(state):
        row, col = state_to_row_col(state)
        adjacent_states = []
        for r, c in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            adjacent_state = row_col_to_state(r, c)
            if adjacent_state is not None:
                adjacent_states.append(adjacent_state)
        return adjacent_states

    # BFS to find all states within h distance
    visited = set()
    queue = [(current_state, 0)]  # Each element is a tuple (state, distance)
    while queue:
        state, distance = queue.pop(0)
        if distance > h:
            break
        visited.add(state)
        for next_state in get_adjacent_states(state):
            if next_state not in visited:
                queue.append((next_state, distance + 1))

    return list(visited)



def generate_product_automaton22(nodes, edges, dfa_states, dfa_transitions, node_labels, atomic_props):
    # Build DFA dictionary for fast lookup
    dfa_dict = {}
    for q, obs_list, q_next in dfa_transitions:
        for formula in obs_list:  # allow multiple formulas
            dfa_dict[(q, formula)] = q_next
    
    # Helper: convert node label set to DFA observation string
    def make_obs_formula(label_set):
        """Generate observation formula based on which APs are present."""
        if not atomic_props:
            return 'true'  # No atomic propositions
        
        # Create formula: each AP is either present or negated
        # Order must match the order in atomic_props
        parts = []
        for ap in atomic_props:
            if ap in label_set:
                parts.append(ap)
            else:
                parts.append("!{}".format(ap))
        
        return ' && '.join(parts)
    
    # Build adjacency dict from edge list
    adj = {int(v): [] for v in nodes.flatten()}
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))  # if undirected
    
    # Enumerate all product nodes
    product_nodes = []
    node_to_index = {}
    for v in nodes.flatten():
        v = int(v)
        for q in dfa_states:
            idx = len(product_nodes)
            product_nodes.append((v, q))
            node_to_index[(v, q)] = idx
    
    # Build product transitions
    transitions = {pn: set() for pn in product_nodes}  # use set to remove duplicates
    for v in nodes.flatten():
        v = int(v)
        if v not in adj:
            continue
        for v_next in adj[v]:
            obs = make_obs_formula(node_labels.get(v_next, set()))
            for q in dfa_states:
                current_state = (v, q)
                if (q, obs) in dfa_dict:
                    q_next = dfa_dict[(q, obs)]
                    transitions[current_state].add((v_next, q_next))
    
    # Convert sets to lists for output
    for k in transitions:
        transitions[k] = list(transitions[k])
    
    # Build NetworkX graph
    product_graph = nx.DiGraph()
    product_graph.add_nodes_from(product_nodes)
    for src, dst_list in transitions.items():
        for dst in dst_list:
            product_graph.add_edge(src, dst)
    
    # Build adjacency matrix
    n = len(product_nodes)
    PR_adj_matrix = np.zeros((n, n), dtype=int)
    for src, dst_list in transitions.items():
        i = node_to_index[src]
        for dst in dst_list:
            j = node_to_index[dst]
            PR_adj_matrix[i, j] = 1
    
    return product_graph, transitions, product_nodes, PR_adj_matrix



def update_product_automaton_incremental(
    product_graph,
    transitions,
    product_nodes,
    node_to_index,
    PR_adj_matrix,
    new_nodes,
    edges,
    dfa_states,
    dfa_transitions,
    node_labels,
    atomic_props
):
    """
    Incrementally add new physical nodes to an existing product automaton.
    Only creates product transitions where physical edges actually exist.
    
    Parameters:
    -----------
    product_graph : nx.DiGraph
        Existing product graph
    transitions : dict
        Existing transitions dictionary {(v,q): [(v',q'), ...]}
    product_nodes : list
        Existing list of product nodes [(v,q), ...]
    node_to_index : dict
        Mapping from (v,q) to index in product_nodes
    PR_adj_matrix : np.ndarray
        Existing adjacency matrix
    new_nodes : set, list, or array
        New physical nodes to add
    edges : array-like
        Physical edges from create_graph (ALL edges including new ones)
    dfa_states : list
        DFA states
    dfa_transitions : list
        DFA transitions [(q, [obs], q_next), ...]
    node_labels : dict
        Node labels {node: set of labels}
    atomic_props : list
        List of atomic propositions in correct order
    
    Returns:
    --------
    Updated versions of all input structures
    """
    
    # Convert new_nodes to set
    if isinstance(new_nodes, set):
        new_nodes_set = set([int(v) for v in new_nodes])
    elif isinstance(new_nodes, np.ndarray):
        new_nodes_set = set([int(v) for v in new_nodes.flatten()])
    else:
        new_nodes_set = set([int(v) for v in new_nodes])
    
    # Build DFA dictionary
    dfa_dict = {}
    for q, obs_list, q_next in dfa_transitions:
        for formula in obs_list:
            dfa_dict[(q, formula)] = q_next
    
    # Helper: convert node label set to DFA observation string
    def make_obs_formula(label_set):
        if not atomic_props:
            return 'true'
        return ' && '.join([ap if ap in label_set else "!{}".format(ap) for ap in atomic_props])
    
    # Build physical adjacency dict from edges
    physical_adj = {}
    for u, v in edges:
        u, v = int(u), int(v)
        if u not in physical_adj:
            physical_adj[u] = []
        if v not in physical_adj:
            physical_adj[v] = []
        physical_adj[u].append(v)
        physical_adj[v].append(u)  # undirected
    
    # Extract existing physical nodes
    existing_physical_nodes = set([v for v, q in product_nodes])
    
    # Filter truly new nodes
    truly_new_nodes = new_nodes_set - existing_physical_nodes
    
    if not truly_new_nodes:
        return product_graph, transitions, product_nodes, node_to_index, PR_adj_matrix
    
    # Add new product nodes
    old_size = len(product_nodes)
    for v in truly_new_nodes:
        for q in dfa_states:
            idx = len(product_nodes)
            new_node = (v, q)
            product_nodes.append(new_node)
            node_to_index[new_node] = idx
            product_graph.add_node(new_node)
            transitions[new_node] = []
    
    # Expand adjacency matrix
    new_size = len(product_nodes)
    if new_size > old_size:
        new_matrix = np.zeros((new_size, new_size), dtype=int)
        new_matrix[:old_size, :old_size] = PR_adj_matrix
        PR_adj_matrix = new_matrix
    
    # Add transitions ONLY where physical edges exist
    nodes_to_process = truly_new_nodes | existing_physical_nodes
    
    for v in nodes_to_process:
        if v not in physical_adj:
            continue
            
        # Only process if v is in the product automaton
        if not any((v, q) in node_to_index for q in dfa_states):
            continue
        
        # For each physical neighbor
        for v_next in physical_adj[v]:
            # Only create transition if both nodes exist in product automaton
            if not any((v_next, q) in node_to_index for q in dfa_states):
                continue
            
            # Check if this edge involves at least one new node
            if v not in truly_new_nodes and v_next not in truly_new_nodes:
                continue  # Skip edges between existing nodes (already processed)
            
            # Create product transitions based on DFA
            obs = make_obs_formula(node_labels.get(v_next, set()))
            
            for q in dfa_states:
                current_state = (v, q)
                
                if current_state not in node_to_index:
                    continue
                
                if (q, obs) in dfa_dict:
                    q_next = dfa_dict[(q, obs)]
                    next_state = (v_next, q_next)
                    
                    if next_state not in node_to_index:
                        continue
                    
                    # Add transition
                    if next_state not in transitions[current_state]:
                        transitions[current_state].append(next_state)
                        product_graph.add_edge(current_state, next_state)
                        
                        i = node_to_index[current_state]
                        j = node_to_index[next_state]
                        PR_adj_matrix[i, j] = 1
    
    return product_graph, transitions, product_nodes, node_to_index, PR_adj_matrix




def find_shortest_path_to_accepting(current_product_state, accepting_dfa_states, transitions):
    """
    Find the shortest path from current product state to any accepting DFA state.

    Parameters
    ----------
    current_product_state : tuple
        (physical_state, dfa_state)
    accepting_dfa_states : set
        Accepting DFA states (e.g., {'accept_all'})
    transitions : dict
        Product transitions: (v, q) -> list of (v', q')

    Returns
    -------
    path : list or None
        List of product states forming the path, or None if unreachable
    """

    queue = deque()
    queue.append(current_product_state)

    parent = {current_product_state: None}
    visited = set([current_product_state])

    while queue:
        current = queue.popleft()
        _, q = current

        # Check acceptance
        if q in accepting_dfa_states:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]

        for nxt in transitions.get(current, []):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = current
                queue.append(nxt)

    return None




def detect_frontiers_e(m, n, visited, unknown):
    """
    Frontier cells are VISITED cells that are 4-connected
    to at least one UNKNOWN
     cell.
    """

    frontiers = set()

    for v in visited:
        r = v // m
        c = v % m

        neighbors = [
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1)
        ]

        for rr, cc in neighbors:
            if 0 <= rr < n and 0 <= cc < m:
                u = rr * m + cc
                if u in unknown:
                    frontiers.add(v)   # <-- ADD THE VISITED CELL
                    break              # no need to check other neighbors
    print(frontiers)
    return frontiers




def dis_to_frontier(m, n, agent_pos, frontier_x):
    fx, fy = to_rc(frontier_x, n)
    r1, c1 = to_rc(agent_pos, n)

    dist = abs(r1 - fx) + abs(c1 - fy)

    return dist




def to_rc(cell, n):
    return cell // n, cell % n




def agents_in_comm_range(env, agent_id, all_agents, com_range):
    """Find agents within communication range."""
    agent1_pos = all_agents[agent_id].current_physical_state
    r1, c1 = to_rc(agent1_pos, env.n)
    in_range_agents = {}
    
    for aid, agent in all_agents.items():
        if aid == agent_id:
            continue
        
        pos = agent.current_physical_state
        r2, c2 = to_rc(pos, env.n)
        dist = abs(r1 - r2) + abs(c1 - c2)
        
        if dist <= com_range:
            in_range_agents[aid] = pos
    
    return in_range_agents



def min_dist_to_frontier(m, n, agent_pos, agents_in_range, frontier_x):
    """
    m, n: grid size
    agent1_pos: int
    agents_in_range: dict {agent_id: position}
    frontier_x: int (cell index)

    return: minimum hop distance from frontier to any agent
            (including agent 1)
    """

    fx, fy = to_rc(frontier_x, n)

    min_dist = float('inf') 

    # check all other agents
    for pos in agents_in_range.values():
        r, c = to_rc(pos, n)
        dist = abs(r - fx) + abs(c - fy)

        if dist < min_dist:
            min_dist = dist

    return min_dist






def compute_frontier_commit(
    x,
    product_graph,
    start_cell,
    start_dfa_state,
    accepting_states,
    commit_states,
    trash_state,
    delta_phi,
    I_x,
    X_size,
    dfa_distance,
    alpha1,
    alpha2,
    alpha3
):
    """
    Compute frontier value V(x) and trajectory sp.

    Returns:
        Vx: float
        sp: list of product states (or None if unreachable/unsafe)
    """

    # ----------------------------
    # 1. Compute shortest product path to frontier
    # ----------------------------
    sp = shortest_product_path_to_frontier(
        product_graph,
        start_cell,
        start_dfa_state,
        x,
        accepting_states,
        commit_states,
        trash_state
    )

    # ----------------------------
    # 2. Compute task progress metric Omega(sp)
    # ----------------------------
    if sp is None:
        Omega = float('-inf')
    else:
        q0 = sp[0][1]          # initial DFA state
        qf = sp[-1][1]         # final DFA state

        if qf == trash_state:
            Omega = float('-inf')
        elif qf in commit_states:
            Omega = -alpha1 * X_size / alpha2
        else:
            Omega = delta_phi(q0, qf, dfa_distance)

    # ----------------------------
    # 3. Compute trajectory weight Wp(sp)
    # ----------------------------
    Wp = len(sp) - 1 if sp is not None else 1

    # ----------------------------
    # 4. Compute frontier value
    # ----------------------------
    if Omega == float('-inf'):
        Vx = float('-inf')
    else:
        Vx = (alpha1 * I_x + alpha2 * Omega) / (Wp ** alpha3)

    # ----------------------------
    # 5. Return both weight and path
    # ----------------------------
    return Vx, sp



def delta_phi(q_start, q_final, dfa_distances):
    if q_start not in dfa_distances or q_final not in dfa_distances:
        return float('-inf')

    d0 = dfa_distances[q_start]
    df = dfa_distances[q_final]

    if d0 == float('inf') or df == float('inf'):
        return float('-inf')

    return d0 - df



def get_next_dfa_state(current_dfa_state, node_label, dfa_transitions, atomic_props):
    """
    Compute the next DFA state given the current state and node label.
    
    Parameters
    ----------
    current_dfa_state : str
        Current DFA state.
    node_label : set
        Set of atomic propositions active at the node, e.g., {'s', 'd'}.
    dfa_transitions : list of tuples
        DFA transitions of the form (q, [formula], q_next).
    atomic_props : list
        List of atomic propositions in the correct order for this DFA.
    
    Returns
    -------
    next_state : str or None
        Next DFA state if a transition exists, otherwise None.
    """
    # Convert node_label set to formula string for matching
    def make_obs_formula(label_set):
        if not atomic_props:
            return 'true'
        return ' && '.join([ap if ap in label_set else "!{}".format(ap) for ap in atomic_props])
    
    obs_formula = make_obs_formula(node_label)
    
    # Search DFA transitions
    for q, formulas, q_next in dfa_transitions:
        if q != current_dfa_state:
            continue
        # Only single-formula transitions
        if len(formulas) != 1:
            continue
        formula = formulas[0]
        if formula == obs_formula:
            return q_next
    
    # No valid transition found
    return None




def shortest_product_path_to_frontier(
    product_graph,
    start_cell,
    start_dfa_state,
    frontier_cell,
    accepting_states,
    commit_states,
    trash_state
):
    """
    Return shortest path from (start_cell, start_dfa_state) to frontier_cell
    in ANY non-trash DFA state.
    """

    # collect all product nodes corresponding to the frontier cell (skip trash)
    frontier_nodes = [
        (frontier_cell, q[1]) for q in product_graph.nodes if q[0] == frontier_cell and q[1] != trash_state
    ]

    if not frontier_nodes:
        return None

    start_node = (start_cell, start_dfa_state)

    shortest_path = None
    min_len = float('inf')

    for target in frontier_nodes:
        try:
            path = nx.shortest_path(product_graph, source=start_node, target=target)
            if len(path) < min_len:
                shortest_path = path
                min_len = len(path)
        except nx.NetworkXNoPath:
            continue

    return shortest_path


def task_progress_metric(
    sp,
    accepting_states,
    commit_states,
    trash_state,
    delta_phi,
    X_size,
    alpha1,
    alpha2
):
    if sp is None:
        return -math.inf

    _, qf = sp[-1]

    if qf == trash_state:
        return -math.inf

    if qf in commit_states:
        return -alpha1 * X_size / alpha2

    q0 = sp[0][1]
    return delta_phi(q0, qf)





