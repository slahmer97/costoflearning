from pulp import *

from greedy_selector import GreedyBalancer

a = GreedyBalancer()
control_plane = 1500
urgent_packets = (0, 8)
data_plane = (1500, 450)
expected = (6, 2)
ret = a.act_int_pg(control_plane, urgent_packets, data_plane, expected)
print(ret)
"""
# A new LP problem
prob = LpProblem("test1", LpMinimize)

# Variables
# 0 <= x <= 4
x = LpVariable("x", 0, 4, cat="Integer")
# -1 <= y <= 1
y = LpVariable("y", -1, 1, cat="Integer")
# 0 <= z
z = LpVariable("z", 0, cat="Integer")

prob += x + 4 * y + 9 * z, "obj"

# Constraints
prob += x + y <= 5, "c1"
prob += x + z >= 10, "c2"
prob += -y + z == 7, "c3"

prob.writeLP("test1.lp")

prob.solve()

print("Status:", LpStatus[prob.status])

# Print the value of the variables at the optimum
for v in prob.variables():
    print(v.name, "=", v.varValue)

# Print the value of the objective
print("objective=", value(prob.objective))
"""