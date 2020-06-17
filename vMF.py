import numpy as np
import random
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv

class vMF(object):
    def __init__(self):
        self.mu = self.uniSampOnSphere(1)[0]
        # self.mu = np.array([[0],[0],[1]])
        self.kappa = 1
        # print('mu', self.mu)

    def rand_uniform_hypersphere(self, N, p):
        v = np.random.normal(0, 1, (N, p))
        v = np.divide(v, np.linalg.norm(v, axis=1, keepdims=True))
        return v

    def uniSampOnSphere(self, N):
        # return a list of np array [3, 1]
        point_list = []
        for i in range(N):
            theta = 2 * 3.141592 * random.random()
            phi = math.acos(1 - 2 * random.random())
            x = math.sin(phi) * math.cos(theta)
            y = math.sin(phi) * math.sin(theta)
            z = math.cos(phi)
            point_list.append(np.array([[x], [y], [z]]))
        return point_list

    def density(self, x):
        a1 = self.mu.transpose().dot(x)
        a2 = math.exp(self.kappa * a1[0, 0])
        z = self.kappa / (4 * 3.141592653 * math.sinh(self.kappa))
        a3 = a2*z
        return a3

    def sample_one(self):
        v = self.rand_uniform_hypersphere(1, 2)     # sample from unit circle
        yup = random.random()
        u = 1 + math.log(yup+(1-yup)*math.exp(-2*self.kappa))/self.kappa
        n = np.hstack((math.sqrt(1-u*u)*v, np.array([[u]])))
        omega = np.cross(n, self.mu.transpose())
        omega_mat = np.array([[0,-omega[0, 2],omega[0, 1]], [omega[0, 2], 0, -omega[0, 0]], [-omega[0, 1], omega[0, 0], 0]])
        theta = math.cosh(n.dot(self.mu))
        R = np.identity(3) + math.sin(theta)/theta*omega_mat + (1-math.cos(theta))/(theta*theta)*omega_mat.dot(omega_mat)
        return R.dot(n.transpose())

    def sample(self, N):
        point_list = []
        for i in range(N):
            point_list.append(self.sample_one())
        return point_list

    def mod(self, x):
        return math.sqrt(x[0, 0]*x[0, 0]+x[1, 0]*x[1, 0]+x[2, 0]*x[2, 0])

if __name__ == '__main__':
    vmf = vMF()
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    N = 1000
    rand_list = vmf.uniSampOnSphere(N)
    x_list = []
    y_list = []
    z_list = []
    density_list = []
    for i in range(N):
        x_list.append(rand_list[i][0][0])
        y_list.append(rand_list[i][1][0])
        z_list.append(rand_list[i][2][0])
        density_list.append(vmf.density(rand_list[i]))
    ax1.scatter(x_list, y_list, z_list, marker='o', c=density_list)
    ax1.set_title('vMF distribution')

    ax2 = fig.add_subplot(122, projection='3d')
    x_list = []
    y_list = []
    z_list = []
    point_list = vmf.sample(N)
    for i in range(N):
        x_list.append(point_list[i][0][0])
        y_list.append(point_list[i][1][0])
        z_list.append(point_list[i][2][0])
    ax2.scatter(x_list, y_list, z_list, marker='o')
    ax2.set_title('vMF distribution')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    plt.show()