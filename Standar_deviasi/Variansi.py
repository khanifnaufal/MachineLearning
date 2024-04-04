#Variance is another number that indicates how spread out the values are.
##In fact, if you take the square root of the variance, you get the standard deviation!
##Or the other way around, if you multiply the standard deviation by itself, you get the variance!

import numpy

speed = [32,111,138,28,59,77,97]

x = numpy.var(speed)

print(x)
