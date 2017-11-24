#!/usr/bin/python
# -*- coding: utf-8 -*-
from scipy.sparse import lil_matrix
import pdb
import random
random.seed(1847859218408232171737)
global EDGES, DEG, DEPTH

def solve_it(input_data):
    global EDGES, DEG, DEPTH
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    EDGES = lil_matrix((node_count, node_count), dtype='bool')
    for e in edges:
        EDGES[e[0],e[1]] = True
    del edges
    DEG = make_degree()
    sol = [-1] * EDGES.shape[0]

    # set max depth of search expansion
    DEPTH = EDGES.shape[0]

    # build a trivial solution
    # every node has its own color
    #solution = range(0, node_count)

    exp_fnc = 'best_node'

    solution = search(exp_fnc=exp_fnc).sol
    obj = len(set(solution))

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def search(exp_fnc='best_node'):
    global EDGES
    best = None
    best_deg = best_node([-1]*EDGES.shape[0])
    #best_deg = random.randint(0, EDGES.shape[0]-1)
    first_sol = [-1]*EDGES.shape[0]
    first_sol[best_deg] = 0
    curr = CPNode(first_sol)
    to_expand = [curr]
    while len(to_expand) > 0:
        node = to_expand.pop()
        if node.is_leaf:
            if best is None or\
                node.value < best.value:
                best = node
        else:
            novels = node.expand(exp_fnc)
            for new in novels[::-1]:
                if best is None or\
                    new.value < best.value:
                    to_expand.append(new)
    #assert(assert_sol(best))
    return best

def make_degree():
    global EDGES
    deg = [0]*EDGES.shape[0]
    for i in range(EDGES.shape[0]):
        deg[i] = (i, EDGES[i,:].nnz + EDGES[:,i].nnz)
    deg.sort(key=lambda x: x[1], reverse=True)
    return deg

def best_node(curr_sol):
    global DEG
    for i in DEG:
        if curr_sol[i[0]] == -1:
            return i[0]
    return None

def max_constr(curr_sol):
    constr = [(i, 0) for i in range(len(curr_sol))]
    for i in range(len(curr_sol)):
        rows, cols = EDGES[i,:].nonzero()
        for c in cols:
            if curr_sol[c] != -1:
                constr[i] = (constr[i][0], constr[i][1] + 1)
        rows, cols = EDGES[:,i].nonzero()
        for r in rows:
            if curr_sol[r] != -1:
                constr[i] = (constr[i][0], constr[i][1] + 1)
    constr.sort(key=lambda x: x[1], reverse=True)
    for i in constr:
        if curr_sol[i[0]] == -1:
            return i[0]
    return None

def has_edge(i, j):
    global EDGES
    return EDGES[i,j] or EDGES[j,i]
    

class CPNode():
    def __init__(self, solution):
        global EDGES
        self.sol = solution
        self.feasible = True
        """
        rows, cols = EDGES.nonzero()
        for r,c in zip(rows, cols):
            if self.sol[r] != -1 and\
                self.sol[r] == self.sol[c]:
                self.feasible = False
                break
        """
        if self.feasible:
            self.is_leaf = (sum(x == -1 for x in self.sol) == 0)
            self.value = max(self.sol)

    def expand(self, exp_fnc):
        fnc = globals()[exp_fnc]
        if self.is_leaf:
            return []
        novel = fnc(self.sol)
        available = assign_value(self.sol, novel)
        to_ret = []
        for c in available:
            new_sol = self.sol.copy()
            new_sol[novel] = c
            to_ret.append(CPNode(new_sol))
        return to_ret


def assign_value(curr_sol, idx):
    global EDGES, DEPTH
    new_sol = curr_sol.copy()
    all_values = set()
    rows, cols = EDGES[idx,:].nonzero()
    for c in cols:
        all_values.add(curr_sol[c])
    rows, cols = EDGES[:,idx].nonzero()
    for r in rows:
        all_values.add(curr_sol[r])
    if not all_values:
        return 0
    available = set(range(max(curr_sol) + 1))
    available = available.difference(all_values)
    if not available:
        return [max(all_values) + 1]
    if len(available) == 1:
        return available
    available = list(sorted(available))
    # decide behavior according to problem size
    if EDGES.shape[0] <= 70:
        return available[:DEPTH]
    elif EDGES.shape[0] == 250:
        best = available[0]
        for a in available:
            if curr_sol.count(best) < curr_sol.count(a):
                best = a
        return [best]
    else:
        return [available[0]]

def assert_sol(best):
    global EDGES
    sol = best.sol
    rows, cols = EDGES.nonzero()
    for r,c in zip(rows, cols):
        if sol[r] == sol[c]:
            return False
    return True

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
