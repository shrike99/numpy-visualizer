# What's new: loops, your own functions, in-place writes, unpacking, containers, try / with.
# The last column of the step list says which pass (or function) a step came from.
a = np.arange(12).reshape(3, 4)

# 1. loops: every pass gets its own steps
for row in a:                 # row is a[0], then a[1], then a[2]
    doubled = row * 2

total = np.zeros(4, dtype=int)
i = 0
while i < 3:
    total = total + a[i]      # a running sum, one step per pass
    i += 1

# 2. in-place writes: only the elements that change pop out and back in (SET)
b = a.copy()
b[1, 1:3] = -1                # 2 of the 12 change
b[b > 8] *= 10                # just the big ones
for j in range(4):
    if j % 2:
        continue              # odd columns are skipped
    b[:, j] = 0

# 3. unpacking: each result is linked to where its elements came from
left, right = np.hsplit(a, 2)
q, r = divmod(a, 5)

# 4. your own functions: the steps inside them show up too (untick "functions" to hide them)
def flip_rows(m):
    m = m.copy()
    m[[0, -1]] = m[[-1, 0]]   # swap the first and last rows
    return m

f = flip_rows(a)

# 5. arrays inside dicts, lists and your own objects are followed as well
parts = {"even": a[:, ::2], "odd": a[:, 1::2]}
parts["sum"] = parts["even"] + parts["odd"]
views = [a.T, a.ravel()]

class Box:
    pass

box = Box()
box.grid = np.zeros((3, 4), dtype=int)
box.grid[1] = a[1]            # copy one row of a in

# 6. try / except / with run step by step as well
try:
    last = a[5]               # there is no row 5...
except IndexError:
    last = a[-1]              # ...so take the last row instead
with np.errstate(divide="ignore"):
    inv = 1 / a               # 1/0 gives inf, without a warning
