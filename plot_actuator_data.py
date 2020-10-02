import matplotlib.pyplot as plt

ONE_FOUR = [0.37, 0.39, 0.44, 0.79, 0.82, 0.9, 0.99, 1.09, 1.69, 1.38, 1.37, 2.18, 1.72, 1.55, 1.59, 1.89, 3.38, 3.01, 3.61, 2.57, 1.75, 1.12, 1.04, 0.51, 0.64, 0.43]
ONE_TWO = [0.38, 0.35, 0.42, 0.6, 0.77, 1.16, 1.27, 1.23, 1.37, 1.37, 1.32, 1.74, 1.42, 1.63, 1.13, 1.29, 1.39, 2.0, 3.51, 3.39, 3.25, 3.09, 1.85, 1.41, 1.33, 2.81, 1.31, 1.63, 2.67, 3.91]

plt.title('steer actuator delay')
plt.plot(ONE_FOUR, label='0.14')
plt.plot(ONE_TWO, label='0.12')
plt.legend()
plt.xlabel('angles')
plt.ylabel('errors')
plt.show()
