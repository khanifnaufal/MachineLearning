#Penting untuk mengetahui bagaimana hubungan antara nilai-nilai sumbu x dan nilai-nilai sumbu y, jika tidak ada hubungan, regresi linier tidak dapat digunakan untuk memprediksi apa pun.
#Hubungan ini - koefisien korelasi - disebut r.
#Nilai r berkisar dari -1 hingga 1, di mana 0 berarti tidak ada hubungan, dan 1 (dan -1) berarti 100% berhubungan.
#Python dan modul Scipy akan menghitung nilai ini untuk Anda, yang perlu Anda lakukan adalah memberikan nilai x dan y.
import matplotlib.pyplot as plt
from scipy import stats

# Data titik-titik
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# Menghitung regresi linier
slope, intercept, r, p, std_err = stats.linregress(x, y)

# Membuat scatter plot dari data
plt.scatter(x, y)

# Membuat garis regresi
plt.plot(x, slope * (stats.zscore(x)) + intercept, color='red')

# Menampilkan plot
plt.show()

print(r)
#hasil dari r adalah -0.76 itu menunjukan bahwa ada relationship antara kedua data, tetapi tidak bagus.