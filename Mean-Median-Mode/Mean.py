import numpy

datanumpy = [23,45,12,88,56]

x = numpy.mean(datanumpy)
 
print("list data: ",datanumpy)
print("rata-rata: ",x)

##program mean tanpa numpy
def mean(numbers):
    if len(numbers) == 0:
        return 0  # Mengembalikan 0 jika daftar kosong
    total = 0
    for number in numbers:
        total += number
    return total / len(numbers)

data = [5, 2, 7, 11, 9]
average = mean(data)
print ("Program Mean tanpa numpy")
print ("list data: ", data)
print("Rata-rata:", average) 