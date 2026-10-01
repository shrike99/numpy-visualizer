# A tiny 4x6 RGB "image": (height, width, channels).
# Tip: switch "layout" to "image order" to see it the way you'd picture an image.
img = np.arange(4 * 6 * 3).reshape(4, 6, 3)

chw = img.transpose(2, 0, 1)      # channels first (PyTorch style) -> (3, 4, 6)
red = img[..., 0]                 # one channel -> (4, 6)
flipped = img[:, ::-1]            # mirror left/right
small = img[::2, ::2]             # downsample: every other pixel
batch = np.stack([img, flipped])  # a batch of 2 images -> (2, 4, 6, 3)
grey = img.mean(axis=-1)          # average the channels -> (4, 6)
