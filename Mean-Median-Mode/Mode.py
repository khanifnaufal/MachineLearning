#program mode dengan menggunakan scipy
from scipy import stats

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = stats.mode(speed)
print ("list data: ", speed)
print("hasil mode:",x)

#program mode tanpa library
def mode(numbers):
    frequency = {}
    
    # Menghitung frekuensi kemunculan setiap angka dalam list
    for num in numbers:
        if num in frequency:
            frequency[num] += 1
        else:
            frequency[num] = 1
    
    # Mencari angka dengan frekuensi kemunculan tertinggi (modus)
    max_frequency = max(frequency.values())
    modes = [num for num, freq in frequency.items() if freq == max_frequency]
    
    return modes

# Contoh penggunaan:
data = [2, 3, 4, 5, 5, 5, 6, 6, 7, 8, 8]
result = mode(data)
print("list data: ", data)
print("Modus: ", result)