import os
import numpy as np
pair_task_lst: [[3, 5], [2, 3], [5, 4], [1, 6], [6, 3], [0, 2], [4, 0], [7, 6]]
array = np.random.randn(5,6)

import os
import numpy as np

cross = [0.6818, 0.7866, 0.70275, 0.7931, 0.842, 0.61145, 0.63185, 0.66535]
AVG = [0.41848, 0.87134, 0.79092, 0.86922, 0.59297, 0.77612, 0.76438, 0.61281]

diff = [18, 16, 12]  # 20, 17, 14
fisher = np.array(cross) - diff[0] + np.random.randn(8)
regmean = np.array(cross) - diff[0] + np.random.randn(8)