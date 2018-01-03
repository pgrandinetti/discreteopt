import math
import networkx
import itertools
from pyscipopt import Model, Conshdlr, quicksum, SCIP_RESULT, multidict

def pairs(vertices):
    return itertools.combinations(vertices, 2)

class TSPconshdlr(Conshdlr):

    def __init__(self, variables, Q, q):
        self.variables = variables
        self.EPS = 1e-6
        self.Q = Q
        self.q = q

    # find subtours in the graph induced by the edges {i,j} for which x[i,j] is positive
    # at the given solution; when solution is None, then the LP solution is used
    def find_subtours(self, solution=None):
        edges = []
        x = self.variables
        for (i,j) in x:
            if self.model.getSolVal(solution, x[i,j]) > self.EPS and\
                (i!=0 and j!=0):
                edges.append((i,j))
        G = networkx.Graph()
        G.add_edges_from(edges)
        components = list(networkx.connected_components(G))
        return components

    def conscheck(self, constraints, solution, check_integrality,
                        check_lp_rows, print_reason, completely, **results):
        subsets = self.find_subtours()
        feasible = True
        for aSet in subsets:
            if sum(self.q[i] for i in aSet) > self.Q:
                #add = self.model.addCons(quicksum(x[i,j] for i in aSet for j in aSet if j > i) <= S_card-NS)
                feasible = False
        if not feasible:
            return {"result": SCIP_RESULT.INFEASIBLE}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}

    # enforces LP solution
    def consenfolp(self, constraints, n_useful_conss, sol_infeasible):
        subsets = self.find_subtours()
        cut = False
        x = self.variables
        for subset in subsets:
            if sum(self.q[i] for i in subset) > self.Q:
                self.model.addCons(quicksum(x[i,j] for (i,j) in pairs(subset) if j>i) <= len(subset) - 2)
                cut = True
        if cut:
            return {"result": SCIP_RESULT.CONSADDED}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}
        '''
        subtours = self.find_subtours()
        if subtours:
            x = self.variables
            for subset in subtours:
                self.model.addCons(quicksum(x[i,j] for (i,j) in pairs(subset) if j>i) <= len(subset) - 1)
                self.model.addCons(quicksum(x[i,j]*self.q[j] for (i,j) in pairs(subset) if j>i) <= self.Q)
            return {"result": SCIP_RESULT.CONSADDED}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}
        '''

    def conslock(self, constraint, nlockspos, nlocksneg):
        pass

def scip_vrp(V, c, m, q, Q, limit=30):
    """ - V: set of nodes in the graph, with V[0]=depot
        - c[i,j]: cost edge (i,j)
        - m: number of vehicles available
        - q[i]: demand for customer i
        - Q: vehicle capacity
        - limit: maximum time limit allowed, in seconds
    """

    model = Model("vrp")
    x = {}
    for i in V:
        for j in V:
            if j > i:
                x[i,j] = model.addVar(vtype='B', name="x(%s,%s)"%(i,j))
    
    model.addCons(quicksum(x[V[0],j] for j in V[1:]) == 2*m, "DegreeDepot")
    for i in V[1:]:
        model.addCons(quicksum(x[j,i] for j in V if j < i) +
                      quicksum(x[i,j] for j in V if j > i) == 2, "Degree(%s)"%i)

    model.setObjective(quicksum(c[i,j]*x[i,j] for i in V for j in V if j>i))
    conshdlr = TSPconshdlr(x, Q, q)
    model.includeConshdlr(conshdlr, "TSP", "TSP subtour eliminator", chckpriority = -10, needscons=False)
    model.setMinimize()
    model.setRealParam('limits/time', limit)
    model.optimize()
    val = model.getObjVal()
    sol = [(i,j) for (i,j) in x if model.getVal(x[i,j]) > 0.5]
    return val, sol

def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def make_data(n):
    V = range(1,n+1)
    x = dict([(i,random.random()) for i in V])
    y = dict([(i,random.random()) for i in V])
    c,q = {},{}
    Q = 100
    for i in V:
        q[i] = random.randint(10,20)
        for j in V:
            if j > i:
                c[i,j] = distance(x[i],y[i],x[j],y[j])
    return V,c,q,Q

if __name__ == "__main__":
    import random
    n = 19
    m = 3
    seed = 1
    random.seed(seed)
    V,c,q,Q = make_data(n)
    z,edges = scip_vrp(V,c,m,q,Q)
    print("Optimal solution:",z)
    print("Edges in the solution:")
    print(sorted(edges))
