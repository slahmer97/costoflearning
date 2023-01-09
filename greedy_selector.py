import numpy as np
from network.globsim import SimGlobals as G


class GreedyBalancer:
    def __init__(self, g_eps=0, g_eps_decay=0, g_eps_min=0):
        self.g_eps = g_eps
        self.g_eps_decay = g_eps_decay
        self.g_eps_min = g_eps_min

        self.step = 0

        self.greedy = 0
        self.non_greedy = 0

    def reset_gepsilon(self, val=0.1, step=190000):
        self.g_eps = val
        self.step = step

    def act_int_pg(self, CP, UP, DP, E):
        self.step += 1
        if self.step % 1000 == 0:
            # self.g_eps = max(self.g_eps_min, self.g_eps * self.g_eps_decay)
            self.g_eps = -7.96e-7 * self.step + 3.0 / 15.0
            self.g_eps = max(self.g_eps, self.g_eps_min)
        if np.random.random(1)[0] > self.g_eps or self.step == 1:
            self.non_greedy += 1
            return False, (None, None, None)
        self.greedy += 1
        from pulp import LpProblem, LpMinimize, LpVariable, value
        prob = LpProblem("ResourceOptimization", LpMinimize)

        X0 = LpVariable("X0", lowBound=0, upBound=15, cat="Integer")
        X1 = LpVariable("X1", lowBound=0, upBound=15, cat="Integer")
        X2 = LpVariable("X2", lowBound=0, upBound=15, cat="Integer")

        Z0 = LpVariable("Z0", lowBound=0, cat="Integer")
        Z1 = LpVariable("Z1", lowBound=0, cat="Integer")
        Z2 = LpVariable("Z2", cat="Integer")

        prob += + Z0 + Z1 + Z2 + X0 - X0 + X1 - X1 + X2 - X2, "obj"

        prob += X0 + X1 + X2 <= 15, "c0"

        prob += X0 <= min(DP[0] + E[0], 15), "c1"

        prob += X1 <= min(DP[1] + E[1], 15), "c2"

        prob += Z0 >= min(DP[0] + E[0] - 1400, 15) - X0, "c3"

        prob += Z1 >= min(UP[1], 15) - X1, "c4"

        # prob += X2 <= CP, "c5"
        prob += X2 <= min(CP, 15), "c5"

        prob += Z2 == min(CP, 15) - X2, "c6"

        # prob.writeLP("resource_optimization.lp")
        from pulp.apis.glpk_api import GLPK_CMD
        prob.solve(GLPK_CMD(msg=0))
        ret = ()
        sum = 0
        for v in prob.variables():
            # print(v.name, "=", v.varValue)
            if v.name[0] == "X":
                ret += (int(v.varValue),)
                sum += int(v.varValue)
        # Print the value of the objective
        # print("objective=", value(prob.objective))
        if sum < 15:
            tmp = list(ret)

            tmp[1] += 15 - sum
            ret = tuple(tmp)
            # print(sum, ret)
        return True, ret

    def act(self, control_plane_packets, urgent_packets, data_plane_packets, expected_arrivals):

        self.step += 1
        if self.step % 100 == 0:
            self.g_eps = max(self.g_eps_min, self.g_eps * self.g_eps_decay)

        if np.random.random(1)[0] > self.g_eps or self.step == 1:
            return False, (None, None, None)

        def filter_val(cost):
            if cost > 0:
                pkts_all = G.BANDWIDTH_PER_RESOURCE * G.RESOURCES_COUNT * G.NET_TIMESLOT_DURATION_S
                return min(cost, pkts_all)
            return cost

        def eval_candidate_solution(s0, s1, l):

            pkts_s0 = int(G.BANDWIDTH_PER_RESOURCE * s0 * G.NET_TIMESLOT_DURATION_S)
            pkts_s1 = int(G.BANDWIDTH_PER_RESOURCE * s1 * G.NET_TIMESLOT_DURATION_S)
            pkts_l = int(G.BANDWIDTH_PER_RESOURCE * l * G.NET_TIMESLOT_DURATION_S)

            """
            if dead1 == 0
                then it is optimal
            if dead1 > 0
                the current solution cannot satisfy all urgent packets in slice 1
            if dead1 < 0
                the current solution is wasting some resources
            """
            dead1 = urgent_packets[1] - pkts_s1

            """
            if dropped1 == 0
                then it is optimal
            if dropped1 > 0
                the current solution will lead to some dropped packets
            if dropped1 < 0
                the current solution is wasting some resources
            """
            dropped0 = data_plane_packets[0] + expected_arrivals[0] - pkts_s0

            not_fowarded = control_plane_packets - pkts_l

            dead1 = filter_val(dead1)
            dropped0 = filter_val(dropped0)
            not_fowarded = filter_val(not_fowarded)
            return abs(- abs(dead1) - abs(dropped0) - abs(not_fowarded))

        def validate_combination(a, b, c):
            return a >= 0 and b >= 0 and c >= 0 and a + b + c == G.RESOURCES_COUNT

        min_val = float("inf")
        min_triplet = (None, None, None)
        for i in range(G.RESOURCES_COUNT + 1):
            for j in range(G.RESOURCES_COUNT + 1):
                for k in range(G.RESOURCES_COUNT + 1):
                    if validate_combination(i, j, k):
                        value = eval_candidate_solution(i, j, k)
                        # print("Candidate Solution({},{},{}) = {}".format(i, j, k, value))
                        if value < min_val:
                            min_triplet = (i, j, k)
                            min_val = value
                            # print("   Exchanged with {}={}".format(min_triplet, min_val))
        return True, min_triplet
