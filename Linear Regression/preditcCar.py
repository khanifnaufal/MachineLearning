from scipy import stats  # Mengimpor modul stats dari SciPy

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]  # Data untuk sumbu x
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]  # Data untuk sumbu y

# Menghitung regresi linier menggunakan data x dan y
slope, intercept, r, p, std_err = stats.linregress(x, y)

# Mendifinisikan fungsi linier (garis regresi)
def myfunc(x):
  return slope * x + intercept

# Menggunakan fungsi linier untuk memprediksi nilai y berdasarkan nilai x=10
speed = myfunc(10)

# Mencetak nilai prediksi
print(speed)
