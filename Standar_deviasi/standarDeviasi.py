import numpy

##Standard deviation is a number that describes how spread out the values are.
##A low standard deviation means that most of the numbers are close to the mean (average) value.
##A high standard deviation means that the values are spread out over a wider range.
#Example: This time we have registered the speed of 7 cars:
speed = [86,87,88,86,87,85,86]

x = numpy.std(speed)

#The standard deviation is:0.9
print("Standar Deviasi: ",x)

#Let us do the same with a selection of numbers with a wider range:
speed1 = [32,111,138,28,59,77,97]

y = numpy.std(speed1)

#The standard deviation is:37,85
print("Standar Deviasi: ",y)
#Meaning that most of the values are within the range of 37.85 from the mean value, which is 77.4.
#As you can see, a higher standard deviation indicates that the values are spread out over a wider range.