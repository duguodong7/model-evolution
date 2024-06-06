from matplotlib import pyplot as plt
import numpy as np
import math
 
np.random.seed(0)
popsize, idx=5, 0
a = np.random.choice(list(set(range(0, popsize)) - {idx}), 1, replace=False)[0]
print('a:', a)

# def sigmoid_function(z):
#     fz = []
#     for num in z:
#         fz.append(1/(1 + math.exp(-num))+0.05)
#     return fz

 
if __name__ == '__main__':
    # z = np.arange(-5, 0, 0.01)
    # fz = sigmoid_function(z)
    # plt.title('Sigmoid Function')
    # plt.xlabel('z')
    # plt.ylabel('sigmoid(z)')
    # plt.plot(z, fz)
    # plt.show()
    # plt.savefig('sigmoid[-15~0]+0.05.png')


