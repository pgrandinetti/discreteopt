#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import time
import random
import pdb
from datetime import datetime
from collections import namedtuple
from plot_tour import plotTSP

Point = namedtuple("Point", ['x', 'y'])

global POINTS

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data):
    global POINTS

    # Modify this code to run your optimization algorithm
    random.seed(1847859218408232171737)
    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    POINTS = points
    print('Problem instance with N = {}'.format(len(POINTS)))
    # build a trivial solution
    #guess = [x for x in range(0, nodeCount)]
    guess = make_initial_guess(nodeCount) # nearest neigh
    init_value = state_value(guess)
    print('Initial guess computed with value {}'.format(init_value))
    #print(guess)
    if len(guess) > 500:
        time_limit = 600
    else:
        time_limit = 180
    if len(guess) < 30000:
        print('Starts at {}'.format(datetime.now().time()))
        solution = local_search(POINTS, guess, init_value, time_limit=time_limit)
        #solution = random_search(guess)
    else:
        solution = guess

    # calculate the length of the tour
    obj = state_value(solution)

    #plotTSP([solution], points)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def make_initial_guess(node_count):
    global POINTS
    s = set(range(node_count))
    guess = [s.pop()]
    while len(s) > 1:
        min_dist = -1
        best = None
        for elem in s:
            if min_dist == -1 or\
               length(POINTS[guess[-1]], POINTS[elem]) < min_dist:
                min_dist = length(POINTS[guess[-1]], POINTS[elem])
                best = elem
        guess.append(best)
        s.remove(best)
    guess.append(s.pop())
    return guess

def state_value(solution):
    global POINTS
    obj = length(POINTS[solution[-1]], POINTS[solution[0]])
    for index in range(0, len(POINTS)-1):
        obj += length(POINTS[solution[index]], POINTS[solution[index+1]])
    return obj

def random_search(guess):
    time_limit = 300
    start = time.time()
    diff = time.time() - start
    i = 0
    best = guess.copy()
    while diff < time_limit:
        novel = best.copy()
        random.shuffle(novel)
        if state_value(novel) < state_value(best):
            best = novel
        i += 1
        diff = time.time() - start
    print('Returning solution with {} visited permutations'.format(i))
    return best

def accept(current, novel, temperature):
    old_val = state_value(current)
    new_val = state_value(novel)
    if new_val <= old_val:
        return True
    if math.exp(-abs(new_val - old_val) / temperature) > random.random():
        return True
    #if random.uniform(temperature, 1) < random.random():
       #return True
    return False

def local_search(points, guess, guess_val, time_limit=120):
    tabu = [] # keep last visited states
    tabu_size = 5000
    best = guess.copy()
    current = guess
    lost = 0
    counter = 0
    max_iter = 50000
    T = math.sqrt(len(points))
    alpha = 0.999
    min_T = 1e-8
    start = time.time()
    diff = time.time() - start
    while diff < time_limit:
        if T <= min_T:
            T = math.sqrt(len(points))
        tabu.append(current)
        neigh = find_neigh(current, tabu, counter)
        if neigh is not None:
            tabu.append(neigh)
            if len(tabu) == tabu_size + 1:
                tabu = tabu[1:]
            if accept(current, neigh, T):
                current = neigh
                if state_value(current) < state_value(best):
                    best = current
        else:
            lost += 1
        counter += 1
        T *= alpha
        diff = time.time() - start
    assert(assert_sol(best, len(points)))
    print('Returning solution after {} iteration and {} lost iterations at {}'.format(counter, lost, datetime.now().time()))
    return best

def find_neigh(current, tabu, counter):
    global POINTS
    if counter == 0:
        rand = len(current) - 1
    else:
        rand = None
    # for k-opt algorithm
    '''
    all_sol.sort(key=lambda x: state_value(x))
    for s in all_sol:
        if not is_permutation(current, s):
            if not s in tabu:
                return s
    return None
    '''
    # for 2-opt
    neigh = two_opt(current)
    if not neigh in tabu:
        return neigh
    return None

def two_opt(curr_sol):
    l = random.randint(2, len(curr_sol) - 1)
    i = random.randint(0, len(curr_sol) - l)
    novel = curr_sol.copy()
    novel[i:(i+l)] = reversed(novel[i:(i+l)])
    return novel

def is_permutation(sol1, sol2):
    j1 = 0
    j2 = sol2.index(sol1[j1])
    while j1 < len(sol1):
        if sol1[j1] != sol2[j2 % len(sol2)]:
            return False
        j1 += 1
        j2 += 1
    return True

def k_opt(curr_sol, rand=None):
    global POINTS
    all_sol = []
    if rand is None:
        rand = random.randint(0, len(curr_sol)-1)
    x1 = (curr_sol[rand-1], rand-1)
    x2 = (curr_sol[rand], rand)
    opt = 5
    current = curr_sol
    while opt > 0:
        length1 = length(POINTS[x1[0]], POINTS[x2[0]])
        x3 = None
        to_avoid = {x1[1], x1[1] % len(POINTS),
                    x2[1], x2[1] % len(POINTS),
                    x2[1]+1, (x2[1]+1) % len(POINTS)}
        for i in range(len(current)):
            if i not in to_avoid and\
                length(POINTS[x2[0]], POINTS[i]) < length1:
                x3 = (current[i], i)
                break
        if x3 is None:
            return all_sol
        x4 = (current[x3[1]-1], x3[1]-1)
        a_sol = [x2[0], x3[0]]
        i = x3[1] + 1
        while current[i % len(POINTS)] != x1[0]:
            a_sol.append(current[i % len(POINTS)])
            i += 1
        a_sol += [x1[0], x4[0]]
        i = x4[1] - 1
        while current[i % len(POINTS)] != x2[0]:
            a_sol.append(current[i % len(POINTS)])
            i -= 1
        if len(set(a_sol)) != len(POINTS):
            raise ValueError('Stmg is wrong')
        all_sol.append(a_sol.copy())
        opt -= 1
        x1 = (x1[0], a_sol.index(x1[0]))
        x2 = (x4[0], a_sol.index(x4[0]))
        current = a_sol.copy()
    return all_sol
    

def assert_sol(solution, tot):
    return len(set(solution)) == tot


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

