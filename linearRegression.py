import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

n = 5000
sig = 10
x = np.array([[np.random.randint(0, 100), np.random.randint(0, 100)] for i in range(n)])
epsilon = sig*np.random.randn(n, 1)
y = x@[[2], [3]] + epsilon

beta_h = np.linalg.inv(x.T@x)@x.T@y

x1 = [x[i][0] for i in range(n)]
x2 = [x[i][1] for i in range(n)]
ys = [y[i][0] for i in range(n)]

x1a = np.linspace(0, 100, 100)
x2a = np.linspace(0 ,100, 100)
X1a, X2a = np.meshgrid(x1a, x2a)
Y_h = X1a*beta_h[0][0]+X2a*beta_h[1][0]

y_h = x@beta_h
y_hs = [y_h[i][0] for i in range(n)]
deltas = [y_hs[i]-ys[i] for i in range(n)]


fig = plt.figure()
deltal = np.linspace(-3*sig, 3*sig, 100)
frequency = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.square(deltal)/(2*np.square(sig)))*n
sigma = np.sqrt(np.mean(np.square(deltas)))
plt.hist(deltas, 6*sig)
plt.plot(deltal, frequency, c='c')
plt.xlabel(sigma)

#ax = plt.axes(projection = '3d')
#ax.scatter3D(x1, x2, ys, c='c')
#ax.contour3D(X1a, X2a, Y_h, 50)
#ax.set_xlabel('x1')
#ax.set_ylabel('x2')
#ax.set_zlabel('y_head')
plt.show()