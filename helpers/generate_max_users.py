k = []
for u0 in range(29, 5, -1):
    for u1 in range(1, 60):
        if u1 < 13:
            continue
        term1 = u0 * 0.35
        term2 = u1 * 0.5

        load = term1 + term2
        if 14 >= load:
            continue
        if load >= 16:
            continue
        k.append((u0,u1))

print(k)