import numpy
import matplotlib.pyplot as plt

#Create an array with 100000 random numbers, and display them using a histogram with 100 bars:
x = numpy.random.normal(0.0, 5.0, 100000)

plt.hist(x, 100)
plt.show()