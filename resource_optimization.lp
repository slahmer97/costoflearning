\* ResourceOptimization *\
Minimize
obj: 0 X0 + 0 X1 + 0 X2 + Z0 + Z1 + Z2
Subject To
c0: X0 + X1 + X2 <= 15
c1: X0 <= 1506
c2: X1 <= 452
c3: X0 + Z0 >= 6
c4: X1 + Z1 >= 8
c5: X2 <= 1500
c6: X2 + Z2 = 1500
Bounds
 0 <= X0 <= 15
 0 <= X1 <= 15
 0 <= X2 <= 15
 0 <= Z0
 0 <= Z1
 Z2 free
Generals
X0
X1
X2
Z0
Z1
Z2
End
