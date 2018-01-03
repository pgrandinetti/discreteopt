#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import random
import time
from datetime import datetime
from copy import deepcopy
import pdb

global DEPOT

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def solve_it(input_data):
    global DEPOT
    random.seed(1659163401)
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    #the depot is always the first customer in the input
    depot = customers[0] 
    DEPOT = depot

    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = trivial_sol(customers, depot, vehicle_count, vehicle_capacity)

    # use local search
    limit, neigh_fnc = param_map(customer_count, vehicle_count)
    vehicle_tours = local_search(customers, vehicle_tours, vehicle_capacity, time_limit=limit, fnc=neigh_fnc)

    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = state_value(vehicle_tours)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += str(depot.index) + ' ' + ' '.join([str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'

    return outputData

def param_map(customer_count, vehicle_count):
    # give different time limits to problem instances
    # times are in seconds
    if customer_count < 50:
        return 300, find_neigh_2
    elif customer_count < 90:
        return 300, find_neigh_2 # instance 3
    elif customer_count < 400:
        return 300, find_neigh
    else:
        return 600, find_neigh_2 # instance 6

def trivial_sol(customers, depot, vehicle_count, vehicle_capacity):
    vehicle_tours = []
    
    remaining_customers = set(customers)
    remaining_customers.remove(depot)
    
    for v in range(0, vehicle_count):
        # print "Start Vehicle: ",v
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
            used = set()
            order = sorted(remaining_customers, key=lambda customer: -customer.demand)
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(customer)
                    # print '   add', ci, capacity_remaining
                    used.add(customer)
            remaining_customers -= used
    return vehicle_tours

def state_value(veh_tours):
    global DEPOT
    obj = 0
    for v in range(len(veh_tours)):
        vehicle_tour = veh_tours[v]
        if len(vehicle_tour) > 0:
            obj += length(DEPOT, vehicle_tour[0])
            for i in range(0, len(vehicle_tour)-1):
                obj += length(vehicle_tour[i], vehicle_tour[i+1])
            obj += length(vehicle_tour[-1], DEPOT)    
    return obj

def accept(current, novel, temperature):
    old_val = state_value(current)
    new_val = state_value(novel)
    if new_val <= old_val:
        return True
    if math.exp(-abs(new_val - old_val) / temperature) > random.random():
        return True
    return False

def find_neigh(curr_sol, customers, vehicle_capacity):
    neigh = deepcopy(curr_sol)
    r1 = random.randint(1, len(customers)-1) # what customer to move...
    d = customers[r1].demand
    available = []
    for veh_tour in neigh:
        if customers[r1] in veh_tour:
            veh_tour.remove(customers[r1])
        tot = vehicle_capacity
        for cus in veh_tour:
            tot -= cus.demand
        available.append(tot)
    s = [(i,x) for (i,x) in enumerate(available) if x>=d]
    r2 = s[random.randint(0, len(s)-1)][0] # ... to what vehicle...
    r3 = random.randint(0, len(curr_sol[r2])) # ... in what position
    neigh[r2].insert(r3, customers[r1])
    return neigh

def find_neigh_2(curr_sol, customers, vehicle_capacity):
    neigh = deepcopy(curr_sol)
    v1 = random.randint(0, len(curr_sol)-1) # from what vehicle
    while len(curr_sol[v1]) == 0:
        v1 = random.randint(0, len(curr_sol)-1)
    c1 = random.randint(0, len(curr_sol[v1])-1) # what customer
    tmp = neigh[v1][c1]
    cap1, cap2 = vehicle_capacity+1, 0
    while cap1 > vehicle_capacity or cap2 < tmp.demand:
        v2 = random.randint(0, len(curr_sol)-1) # to what vehicle
        c2 = random.randint(0, len(curr_sol[v2])) # in what position
        cap1 = sum(curr_sol[v1][x].demand for x in range(len(curr_sol[v1])) if x!=c1)
        if c2 < len(curr_sol[v2]):
             cap1 += curr_sol[v2][c2].demand
        cap2 = vehicle_capacity - sum(curr_sol[v2][x].demand for x in range(len(curr_sol[v2])) if x!=c2)
    if c2 < len(curr_sol[v2]):
        neigh[v1][c1] = neigh[v2][c2]
        neigh[v2][c2] = tmp
    else:
        neigh[v1].remove(tmp)
        neigh[v2].insert(c2, tmp)
    return neigh

def local_search(customers, guess, vehicle_capacity, time_limit=120, fnc=find_neigh):
    best = deepcopy(guess)
    current = guess
    restart = 0
    counter = 0
    T = len(customers)
    alpha = 0.999
    min_T = 1e-8
    start = time.time()
    diff = time.time() - start
    print('Local search starts at {}'.format(datetime.now().time()))
    while diff < time_limit:
        if T <= min_T:
            T = len(customers)
            restart += 1
        neigh = fnc(current, customers, vehicle_capacity)
        if neigh is not None:
            if accept(current, neigh, T):
                current = neigh
                if state_value(current) < state_value(best):
                    best = deepcopy(current)
        counter += 1
        T *= alpha
        diff = time.time() - start
    print('Returning solution after {} iteration and {} restarts at {}'.format(counter, restart, datetime.now().time()))
    return best


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

