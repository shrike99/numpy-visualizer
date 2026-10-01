# Sorting and shuffling move elements around - follow the colours.
rng = np.random.default_rng(1)
a = rng.integers(0, 50, size=(4, 5))

rows = np.sort(a, axis=1)        # sort inside each row
cols = np.sort(a, axis=0)        # sort inside each column
flat = np.sort(a, axis=None)     # sort everything -> 1D
b = a.copy()
rng.shuffle(b)                   # shuffle the rows in place
