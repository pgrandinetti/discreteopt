#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import random
random.seed(1847859218408232171737)
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))

    # default solution
    """
    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight
    """
    if len(items) <= 200:
        # dynamic progr
        value, taken, tab = dynamic_prog(capacity, items)
        # DF-search
        #value, taken, visited = DFSearch(capacity, items)
    elif len(items) <= 400:
        value, taken, tab = dynamic_prog_2(capacity, items, eps=0.01)
    elif len(items) <= 1000:
        value, taken, tab = dynamic_prog(capacity, items)
    else:
        value, taken = greedy(capacity, items, 'value')

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def greedy(K, items, ordering='ratio'):
    elems = _order_elem(items, k=ordering)
    sol = [0] * len(items)
    v = 0
    w = 0
    for r in elems:
        if w + items[r[1]].weight > K:
            break
        v += items[r[1]].value
        w += items[r[1]].weight
        sol[r[1]] = 1
    assert_sol(K, items, v, sol)
    return v, sol


def _order_elem(items, k='ratio'):
    elem = [(0, 0)] * len(items)
    if k == 'ratio':
        for idx in range(len(items)):
            elem[idx] = (items[idx].value / items[idx].weight, idx)
        elem.sort(key=lambda x: x[0])
    elif k == 'value':
        for idx in range(len(items)):
            elem[idx] = (items[idx].value, idx)
        elem.sort(key=lambda x: x[0], reverse=True)
    elif k == 'size':
        for idx in range(len(items)):
            elem[idx] = (items[idx].weight, idx)
        elem.sort(key=lambda x: x[0])
    return elem


def dynamic_prog(K, items):
    N = len(items)
    tab = [[0 for i in range(N + 1)] for j in range(K + 1)]
    """
    # compute first the indexes that will be needed in the table
    to_compute = set()
    visited = set([(K, N)])
    done = set()
    while len(visited) > 0:
        last = visited.pop()
        to_compute.add(last)
        done.add(last)
        i = last[0]
        j = last[1]
        if i > 1 and j > 1:
            if (i, j-1) not in done:
                visited.add((i, j-1))
            if i-items[j-1].weight > 1 and (i-items[j-1].weight, j-1) not in done:
                visited.add((i-items[j-1].weight, j-1))
    to_compute = list(to_compute)
    print("To visit: {}".format(len(to_compute)))
    to_compute.sort(key=lambda x:(x[0],x[1]))
    for pair in to_compute:
        i = pair[0]
        j = pair[1]
        if items[j-1].weight > i:
            tab[i][j] = tab[i][j-1]
        else:
            tab[i][j] = max(tab[i][j-1], tab[i-items[j-1].weight][j-1]+items[j-1].value)        
    """
    for i in range(1, K + 1):
        for j in range(1, N + 1):
            if items[j - 1].weight > i:
                tab[i][j] = tab[i][j - 1]
            else:
                tab[i][j] = max(
                    tab[i][j - 1], tab[i - items[j - 1].weight][j - 1] + items[j - 1].value)

    opt = tab[K][N]
    taken = [0] * N
    i, j2 = (K, N)
    while j2 >= 1:
        if tab[i][j2] == tab[i][j2 - 1]:
            taken[j2 - 1] = 0
        else:
            taken[j2 - 1] = 1
            i -= items[j2 - 1].weight
        j2 -= 1
    assert_sol(K, items, opt, taken)
    return opt, taken, tab


def dynamic_prog_2(K, items2, eps=0.2):
    N = len(items2)
    Lbound = max([i.value for i in items2])
    items = []
    for i in items2:
        items.append(Item(i.index, math.ceil(
            i.value / ((eps / N) * Lbound)), i.weight))
    P = sum([i.value for i in items])
    #print(N, P, K)
    tab = [[max(P, K + 1) for p in range(P + 1)] for i in range(N + 1)]
    tab[0][0] = 0
    for i in range(1, N + 1):
        tab[i][0] = 0
    for i in range(1, N + 1):
        for j in range(1, P + 1):
            if items[i - 1].value <= j:
                tab[i][j] = min(tab[i - 1][j], items[i - 1].weight +
                                tab[i - 1][j - items[i - 1].value])
            else:
                tab[i][j] = tab[i - 1][j]
    opt = -1
    for p in range(P):
        if tab[N][p] <= K:
            opt = max(opt, p)
    sol = [0] * len(items)
    p = opt
    for i in range(N, 0, -1):
        if items[i - 1].value <= p:
            if items[i - 1].weight + tab[i - 1][p - items[i - 1].value] < tab[i - 1][p]:
                sol[i - 1] = 1
                p -= items[i - 1].value
    opt = 0
    for i in range(N):
        if sol[i] == 1:
            opt += items2[i].value
    assert_sol(K, items2, opt, sol)
    return opt, sol, tab


def DFSearch(K, items):
    n1 = Node(K, items, [])
    solution, visited = search(K, items, [n1])
    assert_sol(K, items, solution.value, solution.sol)
    return solution.value, solution.sol, visited


def search(K, items, node_list):
    curr_best = None
    visited = 0
    while len(node_list) > 0:
        n = node_list.pop()
        if n.feasible:
            if not n.is_leaf:
                if curr_best == None or n.estimate > curr_best.value:
                    visit(n, node_list, K, items)
                    visited += 1
            else:
                if curr_best == None or n.value > curr_best.value:
                    curr_best = n
    return curr_best, visited


def visit(node, node_list, K, items):
    sol1 = node.sol.copy()
    sol_l = sol1 + [1]
    sol_r = sol1 + [0]
    left = Node(K, items, sol_l)
    right = Node(K, items, sol_r)
    #depth_first(node_list, left, right)
    #best_first(node_list, left, right)
    rand_first(node_list, left, right)


def depth_first(node_list, left, right):
    if right.feasible:
        node_list.append(right)
    if left.feasible:
        node_list.append(left)


def best_first(node_list, left, right):
    if right.feasible:
        node_list.append(right)
    if left.feasible:
        node_list.append(left)
    if left.feasible or right.feasible:
        node_list.sort(key=lambda x: x.value)


def rand_first(node_list, left, right):
    r = random.random()
    if r > 0.5:
        l = [left, right]
    else:
        l = [right, left]
    if l[0].feasible:
        node_list.append(l[0])
    if l[1].feasible:
        node_list.append(l[1])


class Node():
    def __init__(self, K, items, sol):
        w, v = (0, 0)
        self.is_leaf = (len(sol) == len(items))
        for n in range(len(sol)):
            if sol[n]:
                w += items[n].weight
                v += items[n].value
        if w > K:
            self.feasible = False
        else:
            self.value = v
            self.feasible = True
            self.sol = sol
        if self.feasible:
            if not self.is_leaf:
                self.estimate = relax(K, items, sol, self.value)
            else:
                self.estimate = self.value


def relax(K, items, curr_sol, curr_value):
    ratio = [(0, 0)] * (len(items) - len(curr_sol))
    for idx in range(len(curr_sol), len(items)):
        ratio[len(curr_sol) - idx] = (items[idx].value /
                                      items[idx].weight, idx)
    ratio.sort(key=lambda x: x[0])
    v = curr_value
    for r in ratio:
        v += items[r[1]].value
        if v > K:
            return K
    return v


def assert_sol(K, items, opt_value, sol):
    v = 0
    w = 0
    for i in range(len(sol)):
        if sol[i] == 1:
            v += items[i].value
            w += items[i].weight
    assert(v == opt_value)
    assert(w <= K)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
