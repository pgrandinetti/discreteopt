#!/usr/bin/python
# -*- coding: utf-8 -*-
from scipy.sparse import lil_matrix
import pdb
global EDGES, DEG

def solve_it(input_data):
    global EDGES, DEG
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

    # build a trivial solution
    # every node has its own color
    #solution = range(0, node_count)

    solution = search().sol
    obj = max(solution) + 1

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def search():
    global EDGES
    best = None
    best_deg = best_node([-1]*EDGES.shape[0])
    first_sol = [-1]*EDGES.shape[0]
    first_sol[best_deg] = 0
    curr = CPNode(first_sol)
    to_expand = [curr]
    while len(to_expand) > 0:
        node = to_expand[-1]
        if node.is_leaf:
            if best is None or\
                node.value < best.value:
                best = node
            to_expand.pop()
        elif node.can_expand():
            novel = node.expand()
            if best is None or\
                novel.value < best.value:
                to_expand.append(novel)
        else:
            to_expand.pop()
    assert(assert_sol(best))
    return best

def make_degree():
    global EDGES
    deg = [0]*EDGES.shape[0]
    for i in range(EDGES.shape[0]):
        deg[i] = EDGES[i,:].nnz + EDGES[:,i].nnz
    return deg

def best_node(curr_sol):
    global DEG
    degrees = sorted(DEG, reverse=True)
    for i in range(len(degrees)):
        if curr_sol[degrees[i]] == -1:
            return DEG.index(degrees[i])
    return None

def has_edge(i, j):
    global EDGES
    return EDGES[i,j] or EDGES[j,i]
    

class CPNode():
    def __init__(self, solution):
        global EDGES
        self.sol = solution
        self.next_child = 0
        self.feasible = True
        rows, cols = EDGES.nonzero()
        for r,c in zip(rows, cols):
            if self.sol[r] != -1 and\
                self.sol[r] == self.sol[c]:
                self.feasible = False
                break
        if self.feasible:
            self.is_leaf = (sum(x == -1 for x in self.sol) == 0)
            self.value = max(self.sol)

    def can_expand(self):
        return self.next_child < sum(x == -1 for x in self.sol)

    def expand(self):
        all_sol = self.list_to_assign()
        novel = all_sol[self.next_child]
        self.next_child += len(self.sol)+1
        new_sol = assign_value(self.sol, novel)
        return CPNode(new_sol)

    def list_to_assign(self):
        global DEG
        to_assign = []
        for idx, c in enumerate(self.sol):
            if c == -1:
                to_assign.append(idx)
        to_assign.sort(key=lambda x: DEG[x], reverse=True)
        return to_assign

def assign_value(curr_sol, idx):
    global EDGES
    new_sol = curr_sol.copy()
    all_values = set()
    rows, cols = EDGES.nonzero()
    for r,c in zip(rows, cols):
        if r == idx:
            all_values.add(curr_sol[c])
        elif c == idx:
            all_values.add(curr_sol[r])
    for i in range(len(curr_sol)):
        if not i in all_values:
            new_sol[idx] = i
            return new_sol
    raise Exception('Smtg is dead wrong')

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
