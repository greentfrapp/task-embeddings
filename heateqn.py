import numpy as np
import matplotlib.pyplot as plt

def u(x, t, B_n=1., n=1., k=1., L=1.):
	return B_n * np.sin(n * np.pi * x / L) * np.exp(-k * (n * np.pi / L)**2 * t)

def f(x, L):
	return (x == L).astype(np.float32)

def point_source(L, point, t, k=50., T_1=0., T_2=1.):
	step = 1e-2
	if point != 0.:
		L_1 = point
		x_1 = np.arange(0, L_1 + step * 2, step)
		t_1 = np.ones_like(x_1) * t
		components = []
		for n in np.arange(1, 10):
			B_n = (2. / L_1) * np.trapz(f(x_1, L_1) - (T_1 + ((T_2 - T_1) / L_1) * x_1) * np.sin(n * np.pi * x_1 / L_1), x_1)
			components.append(u(x_1, t_1, B_n=B_n, n=n, k=k, L=L_1))
		y_1 = np.sum(components, axis=0) + T_1 + ((T_2 - T_1) / L_1) * x_1

	if point != 5.:
		L_2 = L - point
		x_2 = np.arange(0, L_2 + step * 2, step)
		t_2 = np.ones_like(x_2) * t
		components = []
		for n in np.arange(1, 10):
			B_n = (2. / L_2) * np.trapz(f(x_2, L_2) - (T_1 + ((T_2 - T_1) / L_2) * x_2) * np.sin(n * np.pi * x_2 / L_2), x_2)
			components.append(u(x_2, t_2, B_n=B_n, n=n, k=k, L=L_2))
		y_2 = np.sum(components, axis=0) + T_1 + ((T_2 - T_1) / L_2) * x_2

	if point == 5.:
		return x_1, y_1
	if point == 0.:
		return x_2 + point, y_2[::-1]
	return np.concatenate([x_1, x_2 + point]), np.concatenate([y_1, y_2[::-1]])

fig, ax = plt.subplots()
color = '#2c3e50'
# for point in np.arange(0, 5.1, 1.):
for point in [1]:
# for t in [0.0001, 0.001,0.01, 0.1, 1.]:
	x, y = point_source(5., point, 0.01)
	test_x = [1.1]
	test_y = np.interp(test_x, x, y)
	ax.plot(x, y)
	# ax.scatter(test_x, test_y)
# color = '#e74c3c'
# ax.plot(-x + L + point, y, color=color)
# ax.legend()
ax.set_ylabel('Temperature/K')
ax.set_xlabel('Displacement/m')
plt.show(block=False)
input()