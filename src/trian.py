import math
import numpy as np
import pylab as pl

h = 5.0 # Height of the camera
d = 1.0 # Distance between walls
n = 10 # Number of walls
s = 3  # Starting wall

degrees = []
for i in range(s, n):
    degrees.append(math.atan(i*d/h) - math.atan((i-1)*d/h))

m = max(degrees)
degrees = [x/m for x in degrees]

pl.plot(range(s, n), degrees, 'b')
pl.plot(range(s, n), [1.0/x for x in degrees], 'r')
pl.show()
