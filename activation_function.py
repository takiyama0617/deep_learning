import numpy as np
import matplotlib.pylab as plt

def step_function(x):
  # Numpyに対して、不等号演算を行うと、配列の各要素に対して、不等号の演算が行われ、
  # Booleanの配列が生成される
  y = x > 0
  return y.astype(np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
