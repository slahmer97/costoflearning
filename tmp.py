a = [1, 2, 3, 4, 5, 6]
indices_tobe_deleted = [0, 2, 4]
for i in sorted(indices_tobe_deleted, reverse=True):
    del a[i]



print(a)