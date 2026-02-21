import spot
import buddy
from graphviz import Digraph
from itertools import product
import re
def create_grid(n, m):
    grid = np.zeros((n, m), dtype=int)
    return grid

# Function to create the graph representation of the grid
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

# Function to extract observations from DFA content
def extract_observations(dfa_content):
    observations = set(re.findall(r'\((.*?)\)', dfa_content))
    return observations

def compute_commit_states(phi, dot_file=None, fmt="pdf"):
    """
    Given an LTL formula phi, translate it to a deterministic, complete DFA,
    build its full self-product automaton,
    and compute the commit states.

    Commit states q ≠ init_state such that from (init_state, q) there is a path
    to (p_accept, q_non_accept) in the product.

    Args:
      phi (str): LTL formula string.
      dot_file (str or None): If provided, save the product graph visualization to this file.
      fmt (str): Format for the graph output, default 'pdf'.

    Returns:
      List[int]: The list of commit states in the original DFA.
    """

    # 1) Translate formula → deterministic, complete DFA
    dfa = spot.translate(phi, 'deterministic', 'complete')
    n = dfa.num_states()
    bdd_dict = dfa.get_dict()
    init_state = dfa.get_init_state_number()

    # 2) Identify accepting and non-accepting states
    accept_states = {s for s in range(n) if dfa.state_is_accepting(s)}
    non_accept_states = set(range(n)) - accept_states

    # 3) Prepare false BDD constant
    false_bdd = buddy.bddfalse

    # 4) Initialize graphviz Digraph if requested
    dot = None
    if dot_file:
        dot = Digraph(comment=f"Full Self-Product of '{phi}'",
                      filename=dot_file,
                      format=fmt)
        for p in range(n):
            for q in range(n):
                dot.node(f"{p},{q}", label=f"{p},{q}", shape="circle")

    # 5) Build adjacency for the product automaton
    adj = {(p, q): [] for p in range(n) for q in range(n)}
    for p in range(n):
        for tr_p in dfa.out(p):
            for q in range(n):
                for tr_q in dfa.out(q):
                    joint = tr_p.cond & tr_q.cond
                    if joint == false_bdd:
                        continue
                    p2, q2 = tr_p.dst, tr_q.dst
                    adj[(p, q)].append((p2, q2))
                    if dot:
                        lbl = spot.bdd_format_formula(bdd_dict, joint)
                        dot.edge(f"{p},{q}", f"{p2},{q2}", label=lbl)

    # 6) Compute commit states
    commit_states = []
    for q in range(n):
        if q == init_state:
            continue
        start = (init_state, q)
        seen = {start}
        stack = [start]
        found = False
        while stack and not found:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
                    p2, q2 = v
                    if (p2 in accept_states) and (q2 in non_accept_states):
                        found = True
                        break
        if found:
            commit_states.append(q)

 
    return commit_states




import numpy as np
import networkx as nx


import networkx as nx
import numpy as np

#prunning added
def generate_product_automaton22(nodes, edges, dfa_states, dfa_transitions, node_labels, atomic_props):
    import numpy as np
    import networkx as nx

    # --- Build DFA dictionary for fast lookup ---
    dfa_dict = {}
    for q, obs_list, q_next in dfa_transitions:
        for formula in obs_list:
            dfa_dict[(q, formula)] = q_next

    # --- Helper: convert node label set to DFA observation string ---
    def make_obs_formula(label_set):
        if not atomic_props:
            return 'true'
        return ' && '.join([ap if ap in label_set else f"!{ap}" for ap in atomic_props])

    # --- Helper: check if a physical node is an obstacle ---
    def is_obstacle(v):
        return 'obs' in node_labels.get(v, set())

    # --- Build adjacency dict from edge list (skip obstacle nodes) ---
    adj = {}
    for v in nodes.flatten():
        v = int(v)
        if not is_obstacle(v):
            adj[v] = []

    for u, v in edges:
        u, v = int(u), int(v)
        if is_obstacle(u) or is_obstacle(v):
            continue  # skip any edge involving an obstacle
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # --- Enumerate all product nodes (obstacles excluded) ---
    product_nodes = []
    node_to_index = {}
    for v in nodes.flatten():
        v = int(v)
        if is_obstacle(v):
            continue  # never add obstacle nodes to the PA
        for q in dfa_states:
            idx = len(product_nodes)
            product_nodes.append((v, q))
            node_to_index[(v, q)] = idx

    # --- Build product transitions ---
    transitions = {pn: [] for pn in product_nodes}

    for v in adj:
        for v_next in adj[v]:
            if is_obstacle(v_next):
                continue
            obs = make_obs_formula(node_labels.get(v_next, set()))
            for q in dfa_states:
                current_state = (v, q)
                if (q, obs) in dfa_dict:
                    q_next = dfa_dict[(q, obs)]
                    next_state = (v_next, q_next)
                    if next_state in node_to_index and next_state not in transitions[current_state]:
                        transitions[current_state].append(next_state)

    # --- Build NetworkX graph ---
    product_graph = nx.DiGraph()
    product_graph.add_nodes_from(product_nodes)
    for src, dst_list in transitions.items():
        for dst in dst_list:
            product_graph.add_edge(src, dst)

    # --- Build adjacency matrix ---
    n = len(product_nodes)
    PR_adj_matrix = np.zeros((n, n), dtype=int)
    for src, dst_list in transitions.items():
        i = node_to_index[src]
        for dst in dst_list:
            j = node_to_index[dst]
            PR_adj_matrix[i, j] = 1

    return product_graph, transitions, product_nodes, PR_adj_matrix

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
                if f'{ap}' in cond and f'!{ap}' not in cond:
                    true_count += 1
            if true_count == 1:
                pruned_transitions.append((current_state, [cond], next_state))

    return pruned_transitions

from collections import deque

from collections import deque

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

    # print("dfaaaa",dfa_distances)
    return dfa_distances


import spot
import spot


def get_states_within_h_distance_with_diagonals(m, n, current_physical_state, h):
    """
    Get all grid states within Chebyshev distance h (diagonals allowed).

    Parameters
    ----------
    m, n : int
        Grid size (m rows, n columns)
    current_physical_state : int or tuple
        Either linear index or (row, col)
    h : int
        Distance threshold

    Returns
    -------
    neighbors : set
        Set of physical states (same format as input)
    """

    # Convert to (row, col) if needed
    if isinstance(current_physical_state, int):
        r0 = current_physical_state // n
        c0 = current_physical_state % n
        return_as_index = True
    else:
        r0, c0 = current_physical_state
        return_as_index = False

    neighbors = set()

    for r in range(max(0, r0 - h), min(m, r0 + h + 1)):
        for c in range(max(0, c0 - h), min(n, c0 + h + 1)):
            # Chebyshev distance
            if max(abs(r - r0), abs(c - c0)) <= h:
                if return_as_index:
                    neighbors.add(r * n + c)
                else:
                    neighbors.add((r, c))

    return neighbors



def find_new_physical_nodes_edges(visited, product_nodes, adj_matrix, product_graph):
    """
    Find newly discovered product nodes and edges from unvisited states.

    Parameters
    ----------
    visited : set
        Visited product states (v, q)
    product_nodes : list
        All product states
    adj_matrix : np.ndarray
        Product adjacency matrix
    product_graph : nx.DiGraph
        Product graph

    Returns
    -------
    new_nodes : set
        Newly discovered product states
    new_edges : set
        Newly discovered edges (u, v)
    """

    new_nodes = set()
    new_edges = set()

    node_to_idx = {node: i for i, node in enumerate(product_nodes)}

    for u in visited:
        if u not in node_to_idx:
            continue

        i = node_to_idx[u]
        successors = product_graph.successors(u)

        for v in successors:
            if v not in visited:
                new_nodes.add(v)
                new_edges.add((u, v))

    return new_nodes, new_edges




def find_new_physical_nodes_edges_fixed(visited_product, product_graph):
    """
    Incrementally find new product nodes and edges reachable from visited nodes.

    Parameters
    ----------
    visited_product : set
        Set of product nodes (v, q) that have been visited
    product_graph : nx.DiGraph
        The full product graph

    Returns
    -------
    new_nodes : set
        Newly discovered product nodes
    new_edges : set
        Newly discovered edges (u, v)
    """

    new_nodes = set()
    new_edges = set()

    for u in visited_product:
        # iterate over successors in the product graph
        for v in product_graph.successors(u):
            if v not in visited_product:
                new_nodes.add(v)
                new_edges.add((u, v))

    return new_nodes, new_edges




from collections import deque

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
    to at least one UNKNOWN cell.
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

import numpy as np
from collections import deque


def normalize_dfa_transitions(dfa_transitions):
    """
    Ensures DFA states are integers everywhere.
    """
    normalized = {}
    for s, trans in dfa_transitions.items():
        s = int(s)
        normalized[s] = {}
        for ap, nxt in trans.items():
            normalized[s][ap] = int(nxt)
    return normalized


import networkx as nx

def build_product_graph(n, m, dfa_transitions, node_labels):
    """
    Builds the product automaton graph safely.
    """
    G = nx.DiGraph()

    grid_states = range(n * m)

    for x in grid_states:
        aps = node_labels[x]

        for q, trans in dfa_transitions.items():
            q = int(q)
            s = (x, q)
            G.add_node(s)

            for ap in aps:
                if ap in trans:
                    q_next = trans[ap]
                    s_next = (x, q_next)
                    G.add_edge(s, s_next, weight=1)

    return G


import math
from collections import deque


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






def compute_frontier_commit(x, product_graph, start_cell, start_dfa_state,
                             accepting_states, commit_states, trash_state,
                             delta_phi, I_x, X_size, dfa_distance,
                             alpha1, alpha2, alpha3):

    start_node = (start_cell, start_dfa_state)

    # Get all product nodes for this frontier cell (skip trash)
    frontier_nodes = [
        (x, q) for (cell, q) in product_graph.nodes()
        if cell == x and q != trash_state
    ]

    if not frontier_nodes:
        return float('-inf'), None

    best_Vx = float('-inf')
    best_sp = None

    # Evaluate EVERY possible (x, q) target — pick the one with best value
    for target in frontier_nodes:
        try:
            sp = nx.shortest_path(product_graph, source=start_node, target=target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        q0 = sp[0][1]
        qf = sp[-1][1]

        # Compute Omega for this path
        if qf == trash_state:
            continue  # skip trash paths entirely
        elif qf in commit_states:
            Omega = -alpha1 * X_size / alpha2
        else:
            Omega = delta_phi(q0, qf, dfa_distance)

        Wp = len(sp) - 1  # path length in hops
        if Wp == 0:
            Wp = 1

        Vx = (alpha1 * I_x + alpha2 * Omega) / (Wp ** alpha3)

        if Vx > best_Vx:
            best_Vx = Vx
            best_sp = sp

    return best_Vx, best_sp



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
        return ' && '.join([ap if ap in label_set else f"!{ap}" for ap in atomic_props])
    
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




def extract_dfa_transitions_with_trash_expanded(formula):
    # Translate formula into deterministic complete DFA
    dfa = spot.translate(formula, 'deterministic', 'complete')
    
    # Get the BDD dictionary
    bdd_dict = dfa.get_dict()
    num_states = dfa.num_states()

    # Identify Trash states (sink non-accepting states)
    is_sink = [False] * num_states
    trash_states_set = set()  # Store original state indices that are trash
    for s in range(num_states):
        outgoing = list(dfa.out(s))
        if len(outgoing) == 1:
            tr = outgoing[0]
            if tr.dst == s and tr.cond == buddy.bddtrue and not dfa.state_is_accepting(s):
                is_sink[s] = True
                trash_states_set.add(s)

    # Map state index to names
    state_names = {}
    for i in range(num_states):
        if dfa.state_is_accepting(i):
            state_names[i] = "accept_all"
        elif is_sink[i]:
            state_names[i] = "Trash"
        else:
            state_names[i] = str(i)

    # Extract initial state index and name
    initial_state_index = dfa.get_init_state_number()
    initial_state_name = state_names[initial_state_index]
    # print(f"Initial state index: {initial_state_index}")
    # print(f"Initial state name: {initial_state_name}")

    # Get atomic propositions actually used in the automaton
    atomic_props = [str(ap) for ap in dfa.ap()]

    # Generate all possible valuations (full minterms)
    all_valuations = list(product([False, True], repeat=len(atomic_props)))

    # Helper: convert valuation to formula string
    def valuation_to_formula(valuation):
        return ' && '.join([prop if val else f'!{prop}' for prop, val in zip(atomic_props, valuation)])

    # Helper: check if valuation satisfies the transition condition
    def valuation_satisfies(cond_bdd, valuation):
        val_bdd = buddy.bddtrue
        for prop, val in zip(atomic_props, valuation):
            var_num = bdd_dict.varnum(prop)
            var_bdd = buddy.bdd_ithvar(var_num)
            if not val:
                var_bdd = buddy.bdd_not(var_bdd)
            val_bdd = buddy.bdd_and(val_bdd, var_bdd)
        product_bdd = buddy.bdd_and(cond_bdd, val_bdd)
        return product_bdd != buddy.bddfalse

    # Extract and expand transitions
    expanded_transitions = []
    for s in range(num_states):
        for tr in dfa.out(s):
            src = state_names[s]
            dst = state_names[tr.dst]
            cond_bdd = tr.cond

            for valuation in all_valuations:
                if valuation_satisfies(cond_bdd, valuation):
                    cond_str = valuation_to_formula(valuation)
                    expanded_transitions.append((src, [cond_str], dst))

    return expanded_transitions, initial_state_name, trash_states_set


import re


def extract_atomic_props_from_dfa(dfa_transitions):
    """
    Extract atomic propositions from DFA transitions in the order they appear.
    Looks at the first transition formula to determine the order.
    """
    import re
    
    if not dfa_transitions:
        return []
    
    # Get the first transition formula to determine order
    first_formula = dfa_transitions[0][1][0]  # e.g., '!s && !p && !d'
    
    # Extract all tokens (letters/words)
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', first_formula)
    
    # Common LTL keywords and operators to exclude
    keywords = {
        'X', 'G', 'F', 'U', 'R', 'W', 'M', 'true', 'false',
        'True', 'False', 'not', 'and', 'or'
    }
    
    # Keep order as they appear in the formula
    atomic_props = []
    seen = set()
    for tok in tokens:
        if tok not in keywords and tok not in seen:
            atomic_props.append(tok)
            seen.add(tok)
    
    return atomic_props
#prunning added
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
    # Convert new_nodes to set of ints
    if isinstance(new_nodes, np.ndarray):
        new_nodes_set = set(int(v) for v in new_nodes.flatten())
    else:
        new_nodes_set = set(int(v) for v in new_nodes)

    # Build DFA dictionary
    dfa_dict = {}
    for q, obs_list, q_next in dfa_transitions:
        for formula in obs_list:
            dfa_dict[(q, formula)] = q_next

    def make_obs_formula(label_set):
        if not atomic_props:
            return 'true'
        return ' && '.join([ap if ap in label_set else f"!{ap}" for ap in atomic_props])

    def is_obstacle(v):
        return 'obs' in node_labels.get(v, set())

    # Build physical adjacency from edges
    physical_adj = {}
    for u, v in edges:
        u, v = int(u), int(v)
        physical_adj.setdefault(u, []).append(v)
        physical_adj.setdefault(v, []).append(u)

    existing_physical_nodes = set(v for v, q in product_nodes)

    # Filter new nodes — exclude obstacles
    truly_new_nodes = (new_nodes_set - existing_physical_nodes) - {v for v in new_nodes_set if is_obstacle(v)}

    if not truly_new_nodes:
        return product_graph, transitions, product_nodes, node_to_index, PR_adj_matrix

    # ── Add new product nodes ────────────────────────────────────
    for v in truly_new_nodes:
        for q in dfa_states:
            new_node = (v, q)
            idx = len(product_nodes)
            product_nodes.append(new_node)
            node_to_index[new_node] = idx
            product_graph.add_node(new_node)
            transitions[new_node] = []

    # ── Rebuild adjacency matrix from scratch to stay in sync ───
    # This avoids any shape mismatch from prior pruning or removals.
    new_size = len(product_nodes)
    new_matrix = np.zeros((new_size, new_size), dtype=int)

    # Copy existing edges from the graph (source of truth)
    for (u_node, v_node) in product_graph.edges():
        if u_node in node_to_index and v_node in node_to_index:
            i = node_to_index[u_node]
            j = node_to_index[v_node]
            if i < new_size and j < new_size:
                new_matrix[i, j] = 1

    PR_adj_matrix = new_matrix

    # ── Prune any existing obstacle nodes already in the PA ──────
    nodes_to_prune = [(v, q) for (v, q) in list(product_nodes) if is_obstacle(v)]
    for node in nodes_to_prune:
        if product_graph.has_node(node):
            product_graph.remove_node(node)
        transitions.pop(node, None)
        for key in transitions:
            if node in transitions[key]:
                transitions[key].remove(node)
        if node in product_nodes:
            product_nodes.remove(node)
        if node in node_to_index:
            del node_to_index[node]

    # Rebuild index and matrix after pruning
    node_to_index.clear()
    for idx, node in enumerate(product_nodes):
        node_to_index[node] = idx

    final_size = len(product_nodes)
    PR_adj_matrix = np.zeros((final_size, final_size), dtype=int)
    for (u_node, v_node) in product_graph.edges():
        if u_node in node_to_index and v_node in node_to_index:
            PR_adj_matrix[node_to_index[u_node], node_to_index[v_node]] = 1

    # ── Add transitions where physical edges exist ───────────────
    nodes_to_process = truly_new_nodes | existing_physical_nodes

    for v in nodes_to_process:
        if is_obstacle(v) or v not in physical_adj:
            continue
        if not any((v, q) in node_to_index for q in dfa_states):
            continue

        for v_next in physical_adj[v]:
            if is_obstacle(v_next):
                continue
            if not any((v_next, q) in node_to_index for q in dfa_states):
                continue
            if v not in truly_new_nodes and v_next not in truly_new_nodes:
                continue

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
                    if next_state not in transitions[current_state]:
                        transitions[current_state].append(next_state)
                        product_graph.add_edge(current_state, next_state)
                        i = node_to_index[current_state]
                        j = node_to_index[next_state]
                        PR_adj_matrix[i, j] = 1

    return product_graph, transitions, product_nodes, node_to_index, PR_adj_matrix

##Multi

def to_rc(cell,n):
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

    

    fx, fy = to_rc(frontier_x,n)

   

    min_dist = float('inf') 

    # check all other agents
    for pos in agents_in_range.values():
        r, c = to_rc(pos,n)
        dist = abs(r - fx) + abs(c - fy)

        if dist < min_dist:
            min_dist = dist

    return min_dist



def dis_to_frontier(m,n,agent_pos, frontier_x):

    fx, fy = to_rc(frontier_x,n)
    r1, c1 = to_rc(agent_pos,n)

    dist = abs(r1 - fx) + abs(c1 - fy)

    return dist
    





def get_new_nodes_between_agents(agent1_visited, agent2_visited):
    """
    Calculate new nodes for two agents when they communicate.
    Each agent learns about nodes the other has visited that are new to them.
    
    Parameters:
    -----------
    agent1_visited : set
        Current visited nodes for agent 1
    agent2_visited : set
        Current visited nodes for agent 2
    
    Returns:
    --------
    new_nodes_agent1 : set
        New nodes for agent 1 (nodes agent 2 has visited that agent 1 hasn't)
    new_nodes_agent2 : set
        New nodes for agent 2 (nodes agent 1 has visited that agent 2 hasn't)
    """
    # Agent 1 learns from Agent 2
    new_nodes_agent1 = agent2_visited - agent1_visited
    
    # Agent 2 learns from Agent 1
    new_nodes_agent2 = agent1_visited - agent2_visited
    
    return new_nodes_agent1, new_nodes_agent2



def get_new_nodes_and_labels_between_agents(agent1_visited, agent1_node_labels,
                                             agent2_visited, agent2_node_labels):
    """
    Calculate new nodes and their labels for two agents when they communicate.
    Each agent learns about nodes and their labels from the other agent.
    
    Parameters:
    -----------
    agent1_visited : set
        Current visited nodes for agent 1
    agent1_node_labels : dict
        Node labels for agent 1 {node: set of labels}
    agent2_visited : set
        Current visited nodes for agent 2
    agent2_node_labels : dict
        Node labels for agent 2 {node: set of labels}
    
    Returns:
    --------
    new_nodes_agent1 : set
        New nodes for agent 1 (nodes agent 2 has visited that agent 1 hasn't)
    new_labels_agent1 : dict
        New labels for agent 1 {node: set of labels}
    new_nodes_agent2 : set
        New nodes for agent 2 (nodes agent 1 has visited that agent 2 hasn't)
    new_labels_agent2 : dict
        New labels for agent 2 {node: set of labels}
    """
    # Agent 1 learns from Agent 2
    new_nodes_agent1 = agent2_visited - agent1_visited
    new_labels_agent1 = {}
    for node in new_nodes_agent1:
        node = int(node)
        if node in agent2_node_labels:
            new_labels_agent1[node] = agent2_node_labels[node].copy()
        else:
            new_labels_agent1[node] = set()
    
    # Agent 2 learns from Agent 1
    new_nodes_agent2 = agent1_visited - agent2_visited
    new_labels_agent2 = {}
    for node in new_nodes_agent2:
        node = int(node)
        if node in agent1_node_labels:
            new_labels_agent2[node] = agent1_node_labels[node].copy()
        else:
            new_labels_agent2[node] = set()
    
    return new_nodes_agent1, new_labels_agent1, new_nodes_agent2, new_labels_agent2


import json

def load_agents_from_json(config_file, env, h, alpha1, alpha2, alpha3, AgentClass):
    """Load agent configurations from JSON file."""
    print(f"Loading agents from {config_file}...")
    
    with open(config_file, 'r') as f:
        configs = json.load(f)
    
    agents = {}
    for config in configs:
        agent_id = config['id']
        initial_pos = config['start']
        formula = config['mission']
        description = config.get('description', 'No description')
        
        print(f"\nCreating Agent {agent_id}...")
        print(f"  Description: {description}")
        print(f"  Starting at: {initial_pos}")
        print(f"  Mission: {formula}")
        
        agents[agent_id] = AgentClass(
            agent_id=agent_id,
            initial_position=initial_pos,
            formula_str=formula,
            env=env,
            h=h,
            alpha1=alpha1,
            alpha2=alpha2,
            alpha3=alpha3
        )
    
    print(f"\n✅ Successfully loaded {len(agents)} agents")
    return agents




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
    print(f"Loading configuration from {config_file}...")
    
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
    
    print(f"✅ Configuration loaded:")
    print(f"   Grid: {n}x{m}, h={h}, com_range={com_range}")
    print(f"   Weights: α1={alpha1}, α2={alpha2}, α3={alpha3}")
    print(f"   Node labels: {len(node_labels_t)} nodes with labels")
    
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