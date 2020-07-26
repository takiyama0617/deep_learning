import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2*h)


###########
def function_1(x):
    return 0.01*x**2 + 0.1*x

###########


def function_2(x):
    return x[0]**2 + x[1]**2


if __name__ == "__main__":
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.show()

    x = np.arange(0.0, 20.0, 0.1)
    y = np.arange(0.0, 20.0, 0.1)
    x, y = np.meshgrid(x, y)
    z = function_2((x, y))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("f(x)")
    ax.plot_wireframe(x, y, z)
    plt.show()
