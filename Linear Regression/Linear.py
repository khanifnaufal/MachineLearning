import matplotlib.pyplot as plt  # Mengimpor modul untuk membuat plot
from scipy import stats  # Mengimpor modul statistik dari SciPy

# Data titik-titik
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# Menghitung regresi linier
slope, intercept, r, p, std_err = stats.linregress(x, y)

# Mendifinisikan fungsi linier (garis regresi)
def myfunc(x):
  return slope * x + intercept

# Menghasilkan model dari fungsi linier pada setiap titik x
mymodel = list(map(myfunc, x))

# Membuat scatter plot dari data
plt.scatter(x, y)

# Membuat plot garis regresi
plt.plot(x, mymodel)

# Menampilkan plot
plt.show()
