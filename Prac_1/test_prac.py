import matplotlib.image as mpimg


a = mpimg.imread('4001.jpg')
print a.reshape(a.shape[0], a.shape[1], 1)
