# Copying and shifting elements.
a = np.arange(6).reshape(2, 3)

rep = np.repeat(a, 2, axis=1)    # every column twice   -> (2, 6)
til = np.tile(a, (2, 2))         # the whole block 2x2  -> (4, 6)
rol = np.roll(a, 1, axis=1)      # shift right, wrap around
flp = np.flip(a)                 # reverse both axes
pad = np.pad(a, 1)               # a border of zeros    -> (4, 5)
