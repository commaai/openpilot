import matplotlib.pyplot as plt

DERIV = [0.65, 0.72, 0.76, 0.91, 1.15, 1.32, 1.33, 1.43, 1.56, 1.38, 1.62, 1.19, 1.26, 1.41, 1.2, 1.05, 1.35, 1.49, 1.14, 1.17, 1.31, 1.01, 1.18, 1.48, 1.42, 2.15, 2.02, 1.57, 1.32, 1.58, 1.87]
STOCK = [0.38, 0.35, 0.42, 0.6, 0.77, 1.16, 1.27, 1.23, 1.37, 1.37, 1.32, 1.74, 1.42, 1.63, 1.13, 1.29, 1.39, 2.0, 3.51, 3.39, 3.25, 3.09, 1.85, 1.41, 1.33, 2.81, 1.31, 1.63, 2.67, 3.91]

plt.title('stock vs PID+F')
plt.plot(DERIV, label='derivative with new PI tune')
plt.plot(STOCK, label='all stock (PI+F)')
plt.legend()
plt.xlabel('angles')
plt.ylabel('errors')
plt.show()