from pydtmc import MarkovChain, plot_graph
from matplotlib import pyplot as plt
p = [[0.7, 0.3, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]
mc = MarkovChain(p, ['A', 'B', 'N'])
print(mc.steady_states)
walk = ["A"]
for i in range(1, 10):
    current_state = walk[-1]
    next_state = mc.next_state(current_state, seed=32)
    print(f'{i:02} {current_state} -> {next_state}')
    walk.append(next_state)

plot_graph(mc)
plt.show()
