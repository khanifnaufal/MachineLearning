#program menggunakan numpy
import numpy

dt = [12,89,87,55,22]#array of data

x = numpy.median(dt)#menaruh median di variabel

print("List data: ", dt)
print("Median: ", x)

#program median tanpa numpy
def median(numbers):
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        mid = n // 2
        return (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2
    else:
        return sorted_numbers[n // 2]

data = [23, 45, 12, 88, 56]
median_result = median(data)
print("List data: ", data)
print("Median (tanpa NumPy):", median_result)