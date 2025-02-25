import string
from PIL import Image

IMG_SIZE = 200


def scatter_pixels(img_file):
    w = IMG_SIZE
    img = Image.open(img_file).resize((w, w)).convert("L")
    pels = img.load()
    black_pels = [(x, y) for x in range(w) for y in range(w)
                  if pels[x, y] <= 50]
    return [t[0] for t in black_pels], [w - t[1] for t in black_pels]


def pack_data(x, y):
    """
    pack 2d data to 1d vector
    """
    one_d_data = []
    for i in range(len(x)):
        one_d_data.append(x[i])
        one_d_data.append(y[i])

    return one_d_data


def unpack_1d_data(one_d_data):
    """
    unpack 1d data to 2d vector
    """
    x = []
    y = []
    for i in range(len(one_d_data)):
        if i % 2 == 0:
            x.append(one_d_data[i])
        else:
            y.append(one_d_data[i])
    return x, y