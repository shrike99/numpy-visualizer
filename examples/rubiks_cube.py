# A Rubik's cube: 27 little cubes, a[page, row, column].
# Page 0 = front (nearest you), row 0 = top, column 0 = left.
# Each number is one little cube, so you can follow where a turn moves it.
cube = np.arange(27).reshape(3, 3, 3)

def turn(c, axis, layer, k):
    c = c.copy()
    s = [slice(None)] * 3
    s[axis] = layer
    c[tuple(s)] = np.rot90(c[tuple(s)], k)
    return c

# k=1 clockwise (looking at that face), k=-1 anticlockwise (prime), k=2 half turn
def F(c, k=1): return turn(c, 0, 0, -k)   # front
def B(c, k=1): return turn(c, 0, 2, k)    # back
def U(c, k=1): return turn(c, 1, 0, k)    # up
def D(c, k=1): return turn(c, 1, 2, -k)   # down
def L(c, k=1): return turn(c, 2, 0, -k)   # left
def R(c, k=1): return turn(c, 2, 2, k)    # right
def M(c, k=1): return turn(c, 2, 1, -k)   # middle slice, turns like L
def E(c, k=1): return turn(c, 1, 1, -k)   # equator slice, turns like D
def S(c, k=1): return turn(c, 0, 1, -k)   # standing slice, turns like F
def x(c, k=1): return np.rot90(c, k, axes=(0, 1))    # whole cube, like R
def y(c, k=1): return np.rot90(c, k, axes=(0, 2))    # whole cube, like U
def z(c, k=1): return np.rot90(c, -k, axes=(1, 2))   # whole cube, like F

# every single move from the solved cube
f = F(cube)
b = B(cube)
u = U(cube)
d = D(cube)
l = L(cube)
r = R(cube)
r_prime = R(cube, -1)
r2 = R(cube, 2)
m = M(cube)
e = E(cube)
s = S(cube)
cx = x(cube)
cy = y(cube)
cz = z(cube)

# the "sexy move" R U R' U', step by step
s1 = R(cube)
s2 = U(s1)
s3 = R(s2, -1)
s4 = U(s3, -1)

# do it 6 times and the cube is solved again (every turn of every pass is a step)
back = cube
for _ in range(6):
    back = R(back)
    back = U(back)
    back = R(back, -1)
    back = U(back, -1)
