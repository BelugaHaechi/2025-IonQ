## ==========================================
## TEAM $\langle \bluedevil ^4 \vert \stanfordtree \rangle$
## iQuHACK 2025, Feb. 1–2, MIT, Boston, MA
## ==========================================
key = "b1491918a470768ed8e36b828c3c6c1c6157a07c9163daf81b8d75b18c0a2475"

## ==========================================
# region 0. DEPENDENCIES
## ==========================================

# pip install qiskit qiskit-aer networkx numpy pandas -q

## IonQ, Inc., Copyright (c) 2025,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at iQuHack2025 hosted by MIT and only during the Feb 1-2, 2025 
# duration of such event.

import matplotlib.pyplot as plt
from IPython import display

import networkx as nx
import numpy as np
import pandas as pd
import time

from typing import List
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

# endregion

## ==========================================
# region 1. INPUT (CHALLENGE SETUP)
## ==========================================

## ----------------------------------
# region 1a. GRAPHS (PROBLEMS; TEST CASES; !!need to come up with graph8)
## ----------------------------------

# other graphs candidates to check

import networkx as nx
import matplotlib.pyplot as plt
import random

# region All Prebuilt Graph Types

#-> Cycle Graph C8
def cycle_graph_c8():
    G = nx.cycle_graph(8)
    # plt.figure(figsize=(6, 6))
    # pos = nx.circular_layout(G)
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    # plt.title("Cycle Graph C8")
    # plt.show()
    return G

# Path Graph P16
def path_graph_p16():
    G = nx.path_graph(16)
    # plt.figure(figsize=(12, 2))
    # pos = nx.spring_layout(G, seed=42)
    # nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=300)
    # plt.title("Path Graph P16")
    # plt.show()
    return G

#-> Complete Bipartite Graph K8,8
def complete_bipartite_graph_k88():
    G = nx.complete_bipartite_graph(8, 8)
    # plt.figure(figsize=(8, 6))
    # pos = nx.bipartite_layout(G, nodes=range(8))
    # nx.draw(G, pos, with_labels=True, node_color=['lightcoral'] * 8 + ['lightblue'] * 8,
    #         edge_color='gray', node_size=300)
    # plt.title("Complete Bipartite Graph K8,8")
    # plt.show()
    return G

#-> Complete Bipartite Graph K8,8
def complete_bipartite_graph_k_nn(n):
    G = nx.complete_bipartite_graph(n, n)
    # plt.figure(figsize=(8, 6))
    # pos = nx.bipartite_layout(G, nodes=range(n))
    # nx.draw(G, pos, with_labels=True, node_color=['lightcoral'] * n + ['lightblue'] * n,
    #         edge_color='gray', node_size=300)
    # plt.title("Complete Bipartite Graph K{},{}".format(n,n))
    # # plt.show()
    return G

# Star Graph S16
def star_graph_s16():
    G = nx.star_graph(16)
    # plt.figure(figsize=(8, 8))
    # pos = nx.spring_layout(G, seed=42)
    # nx.draw(G, pos, with_labels=True, node_color='gold', edge_color='gray', node_size=300)
    # plt.title("Star Graph S16")
    # plt.show()
    return G

# Grid Graph 8x4
def grid_graph_8x4():
    G = nx.grid_graph(dim=[8, 4])
    # plt.figure(figsize=(12, 6))
    # pos = {node: node for node in G.nodes()}
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300)
    # plt.title("Grid Graph 8x4")
    # plt.show()
    return G

# Grid Graph 8x4
def grid_graph_nxm(n,m):
    G = nx.grid_graph(dim=[n, m])
    # plt.figure(figsize=(12, 6))
    # pos = {node: node for node in G.nodes()}
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300)
    # plt.title("Grid Graph {}x{}".format(n,m))
    # plt.show()
    return G


#-> 4-Regular Graph with 8 Vertices
def regular_graph_4_8():
    G = nx.random_regular_graph(d=4, n=8, seed=42)
    # plt.figure(figsize=(6, 6))
    # pos = nx.circular_layout(G)
    # nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=500)
    # plt.title("4-Regular Graph with 8 Vertices")
    # # plt.show()
    return G

#-> Cubic (3-Regular) Graph with 16 Vertices
def cubic_graph_3_16():
    G = nx.random_regular_graph(d=3, n=16, seed=42)
    # plt.figure(figsize=(8, 6))
    # pos = nx.spring_layout(G, seed=42)
    # nx.draw(G, pos, with_labels=True, node_color='lightcoral', edge_color='gray', node_size=300)
    # plt.title("Cubic (3-Regular) Graph with 16 Vertices")
    # # plt.show()
    return G

# Disjoint Union of Four C4 Cycles
def disjoint_union_c4():
    cycles = [nx.cycle_graph(4) for _ in range(4)]
    G = nx.disjoint_union_all(cycles)
    # plt.figure(figsize=(12, 6))
    # pos = {}
    # shift_x = 0
    # for component in nx.connected_components(G):
    #     subgraph = G.subgraph(component)
    #     pos_sub = nx.circular_layout(subgraph, scale=1, center=(shift_x, 0))
    #     pos.update(pos_sub)
        # shift_x += 3
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300)
    # plt.title("Disjoint Union of Four C4 Cycles")
    # # plt.show()
    return G

# Complete Bipartite Graph K16,16
def complete_bipartite_graph_k1616():
    G = nx.complete_bipartite_graph(16, 16)
    # plt.figure(figsize=(12, 6))
    # pos = nx.bipartite_layout(G, nodes=range(16))
    # nx.draw(G, pos, with_labels=False, node_color=['lightcoral'] * 16 + ['lightblue'] * 16,
    #         edge_color='gray', node_size=100)
    # plt.title("Complete Bipartite Graph K16,16")
    # # plt.show()
    return G

# 5-Dimensional Hypercube Graph Q5
def hypercube_graph_q5():
    G = nx.hypercube_graph(5)
    # plt.figure(figsize=(10, 8))
    # pos = nx.spring_layout(G, seed=42)
    # nx.draw(G, pos, with_labels=False, node_color='lightgreen', edge_color='gray', node_size=200)
    # plt.title("5-Dimensional Hypercube Graph Q5")
    # plt.show()
    return G

# Tree Graph with 8 Vertices
def tree_graph_8():
    G = nx.balanced_tree(r=2, h=2)
    G.add_edge(6, 7)
    # plt.figure(figsize=(8, 6))
    # pos = nx.spring_layout(G, seed=42)
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300)
    # plt.title("Tree Graph with 8 Vertices")
    # plt.show()
    return G

# Wheel Graph W16
def wheel_graph_w16():
    G = nx.wheel_graph(16)
    # plt.figure(figsize=(8, 8))
    # pos = nx.circular_layout(G)
    # nx.draw(G, pos, with_labels=True, node_color='lightcoral', edge_color='gray', node_size=300)
    # plt.title("Wheel Graph W16")
    # plt.show()
    return G

#-> Random Connected Graph with 16 Vertices
def random_connected_graph_16(p=0.15):
    #n, p = 16, 0.25
    n=16
    while True:
        G = nx.erdos_renyi_graph(n, p, seed=random.randint(1, 10000))
        if nx.is_connected(G):
            break
    # plt.figure(figsize=(10, 8))
    # pos = nx.spring_layout(G, seed=42)
    # nx.draw(G, pos, with_labels=False, node_color='lightgreen', edge_color='gray', node_size=100)
    # plt.title("Random Connected Graph with 16 Vertices")
    # # plt.show()
    return G

# Expander Graph with 32 Vertices
def expander_graph_32():
    G = nx.random_regular_graph(4, 32, seed=42)
    # plt.figure(figsize=(10, 8))
    # pos = nx.spring_layout(G, seed=42)
    # nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=100)
    # plt.title("Expander Graph with 32 Vertices")
    # plt.show()
    return G

#-> Expander Graph with n Vertices
def expander_graph_n(n):
    G = nx.random_regular_graph(4, n, seed=42)
    # plt.figure(figsize=(10, 8))
    # pos = nx.spring_layout(G, seed=42)
    # nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=100)
    # plt.title("Expander Graph with {} Vertices".format(n))
    # plt.show()
    return G

# Planar Connected Graph with 16 Vertices
def planar_connected_graph_16():
    G = nx.grid_graph(dim=[8, 2])
    G = nx.convert_node_labels_to_integers(G)
    additional_edges = [(0, 9), (1, 10), (2, 11), (3, 12), (4, 13), (5, 14), (6, 15),
                        (7, 15), (8, 7)]#, (6, 15), (14, 1), (1, 13), (10, 9), (0, 10), (12, 2), (8, 7)]
    G.add_edges_from([e for e in additional_edges if e[0] < 16 and e[1] < 16])
    assert nx.check_planarity(G)[0], "Graph is not planar."
    # pos = {node: (node // 2, node % 2) for node in G.nodes()}
    # plt.figure(figsize=(16, 8))
    # nx.draw(G, pos, with_labels=False, node_color='lightcoral', edge_color='gray', node_size=100)
    # plt.title("Planar Connected Graph with 16 Vertices")
    # plt.axis('equal')
    # plt.show()
    return G

# endregion

# Choose your favorite graph and build your winning ansatz!

graph1 = cycle_graph_c8() 
graph2 = complete_bipartite_graph_k88() 
graph3 = complete_bipartite_graph_k_nn(5) 
graph4 = regular_graph_4_8() 
graph5 = cubic_graph_3_16() 
graph6 = random_connected_graph_16(p=0.18)
graph7 = expander_graph_n(16) 
#graph8 = -> make your own cool graph

# endregion

graph = graph4

balanced = True
dfs_ansatz = False
n_steps = 50
lr = 0.5
conv_epsilon = 0.0010

## ----------------------------------
# region 1b. ANSATZ (INITIAL VALUE; !!optimize this)
## ----------------------------------

def build_dfs_ansatz(graph: nx.Graph) -> QuantumCircuit:
    """
    Build an optimized QAOA ansatz for the max_cut problem using a DFS-based
    optimization procedure, but define a separate parameter for each edge
    exactly as in the original ParameterVector approach.
    
    A Depth First Search (DFS) is used to obtain a DFS tree of the graph. For
    each DFS tree edge (u, v), we apply:
        U2 = RY(2 * θ_e, v) --> CX(u, v)
    and for each non-tree edge, we apply:
        U1 = CX(u, v) --> RY(2 * θ_e, v) --> CX(u, v).
    
    The order of parameters in the ParameterVector exactly follows the order
    of edges as returned by `graph.edges()`.
    
    Args:
        graph (nx.Graph): The input graph.
    
    Returns:
        QuantumCircuit: A parameterized circuit with one parameter per edge.
    """
    num_qubits = graph.number_of_nodes()
    qc = QuantumCircuit(num_qubits)

    # 1) Hadamard on all qubits to prepare uniform superposition
    qc.h(range(num_qubits))
    
    # 2) Define a ParameterVector with one parameter per edge
    #    The order of edges in `graph.edges()` defines the parameter order.
    edges = list(graph.edges())
    theta = ParameterVector(r"$\theta$", len(edges))
    
    # 3) Perform DFS to identify the DFS tree edges (u, v)
    visited = set()
    dfs_tree_edges = []
    
    def dfs(v):
        visited.add(v)
        for w in sorted(graph.neighbors(v)):  # sort for determinism
            if w not in visited:
                dfs_tree_edges.append((v, w))
                dfs(w)

    # Handle possibly disconnected graphs
    for node in sorted(graph.nodes()):
        if node not in visited:
            dfs(node)
    
    # Create a set of DFS edges in canonical (sorted) form to detect membership
    dfs_tree_set = set(tuple(sorted(e)) for e in dfs_tree_edges)
    
    # 4) Now iterate over edges in the same order as the ParameterVector
    #    For each edge, apply either the optimized cost operator (if DFS edge)
    #    or the full cost operator (if non-tree).
    for (param, edge) in zip(theta, edges):
        # Sort edge for membership check
        u, v = sorted(edge)
        
        # If edge is in the DFS tree set => short operator
        if (u, v) in dfs_tree_set:
            #   RY(2θ) on v -> CX(u,v)
            qc.ry(param, v)
            qc.cx(u, v)
        else:
            # Full operator: CX(u,v) -> RY(2θ) on v -> CX(u,v)
            qc.cx(u, v)
            qc.ry(param, v)
            qc.cx(u, v)

    return qc


def build_ansatz(graph: nx.Graph) -> QuantumCircuit:
    """
    Default ansatz
    """
    ansatz = QuantumCircuit(graph.number_of_nodes())
    # create blank QC template with n_nodes qubits reserved
    ansatz.h(range(graph.number_of_nodes()))
    # Hadamard on {1, 2, ..., n_nodes}
    theta = ParameterVector(r"$\theta$", graph.number_of_edges())
    for t, (u, v) in zip(theta, graph.edges):
        ansatz.cx(u, v) # controlled_not(edge_vertex_u, edge_vertex_v)
        ansatz.ry(t, v) # rotate_y(theta_n, edge_vertex_v)
        ansatz.cx(u, v)

    return ansatz


# endregion

if dfs_ansatz:
    ansatz = build_ansatz(graph)
else:
    ansatz = build_ansatz(graph)
ansatz.draw("mpl", filename="ansatz.svg", fold=-1)  # Save as SVG


## ----------------------------------
# region 1c. HAMILTONIAN (OBJECTIVE OPERATOR; !!optimize this)
## ----------------------------------

def build_maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    """
    Build the MaxCut Hamiltonian for the given graph H = (|E|/2)*I - (1/2)*Σ_{(i,j)∈E}(Z_i Z_j)
    """
    num_qubits = len(graph.nodes)
    edges = list(graph.edges())
    num_edges = len(edges)

    pauli_terms = ["I"*num_qubits] # start with identity
    coeffs = [-num_edges / 2]

    for (u, v) in edges: # for each edge, add -(1/2)*Z_i Z_j
        z_term = ["I"] * num_qubits
        z_term[u] = "Z"
        z_term[v] = "Z"
        pauli_terms.append("".join(z_term))
        coeffs.append(0.5)

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))


def build_balanced_maxcut_hamiltonian(graph: nx.Graph, alpha: float) -> SparsePauliOp:
    """
    Build a 'balanced' MaxCut Hamiltonian for the given graph:

    H_balanced = (|E|/2)*I - (1/2)*Σ_{(i,j)∈E}(Z_i Z_j)
                 + alpha * ( (Σ_j X_j) - n/2 )^2

    In Pauli-Z form, this becomes:

        = (|E|/2 + alpha*n/4)*I
          + Σ_{(i,j) in E} [0.5]*Z_iZ_j
          + Σ_{i<j} [alpha/2]*Z_iZ_j

    We implement it with minimal changes from the original function.
    """
    num_qubits = len(graph.nodes)
    edges = list(graph.edges())
    num_edges = len(edges)

    # 1) Identity offset:
    #    Original: -(num_edges/2)
    #    Now add + (alpha*n/4)
    pauli_terms = ["I" * num_qubits]
    coeffs = [(- num_edges / 2.0) + (alpha * num_qubits / 4.0)]

    # 2) Usual MaxCut part: for each edge, add -(1/2)*Z_iZ_j
    #    (which is represented by +0.5 in the coefficient list,
    #     because the overall sign in front is negative).
    for (u, v) in edges:
        z_term = ["I"] * num_qubits
        z_term[u] = "Z"
        z_term[v] = "Z"
        pauli_terms.append("".join(z_term))
        coeffs.append(0.5)  # same as original code

    # 3) Balanced penalty part: for every pair i<j, add alpha/2
    #    (from expanding (sum_j X_j - n/2)^2 => (alpha/2)*Z_iZ_j)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            z_term = ["I"] * num_qubits
            z_term[i] = "Z"
            z_term[j] = "Z"
            pauli_terms.append("".join(z_term))
            coeffs.append(alpha / 2.0)

    # 4) Build SparsePauliOp from the final list of (Pauli-string, coefficient).
    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))


if balanced:
    alpha = 2
    #ham = build_maxcut_hamiltonian(graph, alpha)
    ham = build_balanced_maxcut_hamiltonian(graph,alpha)
else:
    ham = build_maxcut_hamiltonian(graph)


# endregion


# endregion

# >>>>> EXPAND ABOVE REGION (1) TO WORK ON CHALLENGE <<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>> GOOD >>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<< LUCK <<<<<<<<<<<<<<<<<<<<<<<<<

## ==========================================
# region 2. ENERGY MINIMIZATION VIA Q.I.T.E. (SOLVER)
## ==========================================


## ----------------------------------
# region 2a. Q.I.T. EVOLVER CLASS
## ----------------------------------

class QITEvolver:
    """
    A class to evolve a parametrized quantum state under the action of an Ising
    Hamiltonian according to the variational Quantum Imaginary Time Evolution
    (QITE) principle described in IonQ's latest joint paper with ORNL.
    """
    def __init__(self, hamiltonian: SparsePauliOp, ansatz: QuantumCircuit):
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz

        # Define some constants
        self.backend = AerSimulator()
        self.num_shots = 10000
        self.energies, self.param_vals, self.runtime = list(), list(), list()

    def evolve(self, num_steps: int, lr: float = 0.4, \
               verbose: bool = True, conv_epsilon: float = -1):
        """
        Evolve the variational quantum state encoded by ``self.ansatz`` under
        the action of ``self.hamiltonian`` according to varQITE.
        """
        curr_params = np.zeros(self.ansatz.num_parameters)
        for k in range(num_steps):
            # Get circuits and measure on backend
            iter_qc = self.get_iteration_circuits(curr_params)
            job = self.backend.run(iter_qc, shots=self.num_shots)
            q0 = time.time()
            measurements = job.result().get_counts()
            quantum_exec_time = time.time() - q0

            # Update parameters-- set up defining ODE and step forward
            Gmat, dvec, curr_energy = self.get_defining_ode(measurements)
            dcurr_params = np.linalg.lstsq(Gmat, dvec, rcond=1e-2)[0]
            curr_params += lr * dcurr_params

            # Progress checkpoint!
            if verbose:
                self.print_status(measurements)
            self.energies.append(curr_energy)
            self.param_vals.append(curr_params.copy())
            self.runtime.append(quantum_exec_time)

            # ADD: Convergence success if past 5 change is small enough
            if conv_epsilon > 0 and k > 5:
                if abs(self.energies[-1] - self.energies[-5]) < conv_epsilon:
                    print("Converged before exhausting num_steps, past 5 threshold:", conv_epsilon)
                    break

    def get_defining_ode(self, measurements: List[dict[str, int]]):
        """
        Construct the dynamics matrix and load vector defining the varQITE
        iteration.
        """
        # Load sampled bitstrings and corresponding frequencies into NumPy arrays
        dtype = np.dtype([("states", int, (self.ansatz.num_qubits,)), ("counts", "f")])
        measurements = [np.fromiter(map(lambda kv: (list(kv[0]), kv[1]), res.items()), dtype) for res in measurements]

        # Set up the dynamics matrix by computing the gradient of each Pauli word
        # with respect to each parameter in the ansatz using the parameter-shift rule
        pauli_terms = [SparsePauliOp(op) for op, _ in self.hamiltonian.label_iter() if set(op) != set("I")]
        Gmat = np.zeros((len(pauli_terms), self.ansatz.num_parameters))
        for i, pauli_word in enumerate(pauli_terms):
            for j, jth_pair in enumerate(zip(measurements[1::2], measurements[2::2])):
                for pm, pm_shift in enumerate(jth_pair):
                    Gmat[i, j] += (-1)**pm * expected_energy(pauli_word, pm_shift)
        
        # Set up the load vector
        curr_energy = expected_energy(self.hamiltonian, measurements[0])
        dvec = np.zeros(len(pauli_terms))
        for i, pauli_word in enumerate(pauli_terms):
            rhs_op_energies = get_ising_energies(pauli_word, measurements[0]["states"])
            rhs_op_energies *= get_ising_energies(self.hamiltonian, measurements[0]["states"]) - curr_energy
            dvec[i] = -np.dot(rhs_op_energies, measurements[0]["counts"]) / self.num_shots
        return Gmat, dvec, curr_energy

    def get_iteration_circuits(self, curr_params: np.array):
        """
        Get the bound circuits that need to be evaluated to step forward
        according to QITE.
        """
        # Use this circuit to estimate your Hamiltonian's expected value
        circuits = [self.ansatz.assign_parameters(curr_params)]

        # Use these circuits to compute gradients
        for k in np.arange(curr_params.shape[0]):
            for j in range(2):
                pm_shift = curr_params.copy()
                pm_shift[k] += (-1)**j * np.pi/2
                circuits += [self.ansatz.assign_parameters(pm_shift)]
        
        # Add measurement gates and return
        [qc.measure_all() for qc in circuits]
        return circuits
    
    def plot_convergence(self):
        """
        Plot the convergence of the expected value of ``self.hamiltonian`` with
        respect to the (imaginary) time steps.
        """
        plt.plot(self.energies)
        plt.xlabel("(Imaginary) Time step")
        plt.ylabel("Hamiltonian energy")
        plt.title("Convergence of the expected energy")

    def print_status(self, measurements):
        """
        Print summary statistics describing a QITE run.
        """
        stats = pd.DataFrame({
            "curr_energy": self.energies,
            "num_circuits": [len(measurements)] * len(self.energies),
            "quantum_exec_time": self.runtime 
        })
        stats.index.name = "step"
        display.clear_output(wait=True)
        display.display(stats)

# endregion


## ----------------------------------
# region 2b. UTILITY FUNCTIONS
## ----------------------------------

def compute_cut_size(graph, bitstring):
    """
    Get the cut size of the partition of ``graph`` described by the given
    ``bitstring``.
    """
    cut_sz = 0
    for (u, v) in graph.edges:
        if bitstring[u] != bitstring[v]:
            cut_sz += 1
    return cut_sz


def get_ising_energies(
        operator: SparsePauliOp, 
        states: np.array
    ):
    """
    Get the energies of the given Ising ``operator`` that correspond to the
    given ``states``.
    """
    # Unroll Hamiltonian data into NumPy arrays
    paulis = np.array([list(ops) for ops, _ in operator.label_iter()]) != "I"
    coeffs = operator.coeffs.real
    
    # Vectorized energies computation
    energies = (-1) ** (states @ paulis.T) @ coeffs
    return energies


def expected_energy(
        hamiltonian: SparsePauliOp,
        measurements: np.array
):
    """
    Compute the expected energy of the given ``hamiltonian`` with respect to
    the observed ``measurement``.

    The latter is assumed to by a NumPy records array with fields ``states``
    --describing the observed bit-strings as an integer array-- and ``counts``,
    describing the corresponding observed frequency of each state.
    """
    energies = get_ising_energies(hamiltonian, measurements["states"])
    return np.dot(energies, measurements["counts"]) / measurements["counts"].sum()


def interpret_solution(graph, bitstring):
    """
    Visualize the given ``bitstring`` as a partition of the given ``graph``.
    """
    pos = nx.spring_layout(graph, seed=42)
    set_0 = [i for i, b in enumerate(bitstring) if b == '0']
    set_1 = [i for i, b in enumerate(bitstring) if b == '1']

    plt.figure(figsize=(4, 4))
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=set_0, node_color='blue', node_size=700)
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=set_1, node_color='red', node_size=700)

    cut_edges = []
    non_cut_edges = []
    for (u, v) in graph.edges:
        if bitstring[u] != bitstring[v]:
            cut_edges.append((u, v))
        else:
            non_cut_edges.append((u, v))

    nx.draw_networkx_edges(graph, pos=pos, edgelist=non_cut_edges, edge_color='gray', width=2)
    nx.draw_networkx_edges(graph, pos=pos, edgelist=cut_edges, edge_color='green', width=2, style='dashed')

    nx.draw_networkx_labels(graph, pos=pos, font_color='white', font_weight='bold')
    plt.axis('off')
    plt.show()


# endregion


## ----------------------------------
# region 2c. CALL Q.I.T.E MINIMIZER (!!params to optimize: num_steps, lr)
## ----------------------------------


print("Running: QITEvolver Minimization\n")

# Set up your QITEvolver and evolve!
qit_evolver = QITEvolver(ham, ansatz)
qit_evolver.evolve(num_steps=n_steps, lr=lr, verbose=True, conv_epsilon=conv_epsilon) # lr was 0.4

print("\nCompleted: QITEvolver Minimization\n")

# Visualize your results!
qit_evolver.plot_convergence()

# endregion


# endregion


## ==========================================
# region 3. OUTPUT (SCORING)
## ==========================================

## ----------------------------------
# region 3a. QUANTUM CIRCUIT SAMPLING (COUNTING)
## ----------------------------------

from qiskit_aer import AerSimulator

shots = 100_000

# Sample your optimized quantum state using Aer
backend = AerSimulator()
optimized_state = ansatz.assign_parameters(qit_evolver.param_vals[-1])
optimized_state.measure_all()
counts = backend.run(optimized_state, shots=shots).result().get_counts()

# Find the sampled bitstring with the largest cut value
cut_vals = sorted(((bs, compute_cut_size(graph, bs)) for bs in counts), key=lambda t: t[1])
best_bs = cut_vals[-1][0]

# Now find the most likely MaxCut solution as sampled from your optimized state
# We'll leave this part up to you!!! --Done (Daniel Lai)
most_likely_soln = max(counts, key=counts.get)

print("Quantum Circuit Sample. Size:", shots)
print(counts)
print("Most Likely Solution Found:")
print(most_likely_soln, "Frequency:", max(counts.values()))


# endregion


## ----------------------------------
# region 3b. BRUTE-FORCE CLASSICAL SOLUTION (ANSWER)
## ----------------------------------

# Brute-force approach with conditional checks

verbose = False

G = graph
n = len(G.nodes())
w = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        temp = G.get_edge_data(i, j, default=0)
        if temp != 0:
            w[i, j] = 1.0
if verbose:
    print(w)

best_cost_brute = 0
best_cost_balanced = 0
best_cost_connected = 0

for b in range(2**n):
    x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]

    # Create subgraphs based on the partition
    subgraph0 = G.subgraph([i for i, val in enumerate(x) if val == 0])
    subgraph1 = G.subgraph([i for i, val in enumerate(x) if val == 1])

    bs = "".join(str(i) for i in x)
    
    # Check if subgraphs are not empty
    if len(subgraph0.nodes) > 0 and len(subgraph1.nodes) > 0:
        cost = 0
        for i in range(n):
            for j in range(n):
                cost = cost + w[i, j] * x[i] * (1 - x[j])
        if best_cost_brute < cost:
            best_cost_brute = cost
            xbest_brute = x
            XS_brut = []
        if best_cost_brute == cost:
            XS_brut.append(bs)

        outstr = "case = " + str(x) + " cost = " + str(cost)

        if (len(subgraph1.nodes)-len(subgraph0.nodes))**2 <= 1:
            outstr += " balanced"
            if best_cost_balanced < cost:
                best_cost_balanced = cost
                xbest_balanced = x
                XS_balanced = []
            if best_cost_balanced == cost:
                XS_balanced.append(bs)

        if nx.is_connected(subgraph0) and nx.is_connected(subgraph1):
            outstr += " connected"
            if best_cost_connected < cost:
                best_cost_connected = cost
                xbest_connected = x
                XS_connected = []
            if best_cost_connected == cost:
                XS_connected.append(bs)
        if verbose:
            print(outstr)


# This is classical brute force solver results:

print("\nClassical Brute Force Solver Results")

# interpret_solution(graph, xbest_brute)
print("\nBest solution = " + str(xbest_brute) + " cost = " + str(best_cost_brute))
print(XS_brut)

# interpret_solution(graph, xbest_balanced)
print("\nBest balanced = " + str(xbest_balanced) + " cost = " + str(best_cost_balanced))
print(XS_balanced)

# interpret_solution(graph, xbest_connected)
print("\nBest connected = " + str(xbest_connected) + " cost = " + str(best_cost_connected))
print(XS_connected)

# endregion


## ----------------------------------
# region 3c. FINAL SCORING (AUTO-JUDGE)
## ----------------------------------

def final_score(graph, XS_brut,counts,shots,ansatz,challenge):

    if(challenge=='base'):
        sum_counts = 0
        for bs in counts:
            if bs in XS_brut:
                sum_counts += counts[bs]
        print(f"Pure max-cut: {sum_counts} out of {shots}")
    elif(challenge=='balanced'):
        sum_balanced_counts = 0
        for bs in counts:
            if bs in XS_balanced:
                sum_balanced_counts += counts[bs]
        print(f"Balanced max-cut: {sum_balanced_counts} out of {shots}")
        sum_counts = sum_balanced_counts
    elif(challenge=='connected'):
        sum_connected_counts = 0
        for bs in counts:
            if bs in XS_connected:
                sum_connected_counts += counts[bs]
        print(f"Connected max-cut: {sum_connected_counts} out of {shots}")
        sum_counts = sum_connected_counts

    
    transpiled_ansatz = transpile(ansatz, basis_gates = ['cx','rz','sx','x'])
    cx_count = transpiled_ansatz.count_ops()['cx']
    score = (4*2*graph.number_of_edges())/(4*2*graph.number_of_edges() + cx_count) * sum_counts/shots

    return np.round(score,5)


print("\nSummary of Final Score\n")

print("Base score: " + str(final_score(graph,XS_brut,counts,shots,ansatz,'base')))
print("Balanced score: " + str(final_score(graph,XS_brut,counts,shots,ansatz,'balanced')))
print("Connected score: " + str(final_score(graph,XS_brut,counts,shots,ansatz,'connected')))

print("\nJob Done at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# alpha_candidates = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
# best_alpha = None
# best_score = -1

# for alpha_test in alpha_candidates:
#     ham_test = build_balanced_maxcut_hamiltonian(graph, alpha_test)
#     qit_evolver_test = QITEvolver(ham_test, ansatz)
#     qit_evolver_test.evolve(num_steps=100, lr=0.5, verbose=False)

#     # Sample
#     shots = 50000
#     backend = AerSimulator()
#     optimized_state = ansatz.assign_parameters(qit_evolver_test.param_vals[-1])
#     optimized_state.measure_all()
#     counts_test = backend.run(optimized_state, shots=shots).result().get_counts()

#     # Evaluate your final scoring logic
#     candidate_score = final_score(graph, XS_brut, counts_test, shots, ansatz, 'balanced')

#     print(f"Alpha={alpha_test} => Balanced Score={candidate_score}")
#     if candidate_score > best_score:
#         best_score = candidate_score
#         best_alpha = alpha_test

# print(f"\nBest alpha found so far: {best_alpha} with score {best_score}")



# endregion



# endregion


## ==========================================
## END OF TEST SUITE CODE
## ==========================================