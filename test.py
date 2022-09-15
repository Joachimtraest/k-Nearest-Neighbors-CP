import numpy as np
from kNN_reg import ICP_kNN as knn

boston = np.load('boston.npy')
prices = np.load('prices.npy')

print(knn.K_fold_validation_ICP_knn(boston, prices, 10, 4, 198, epsilon = 0.10))
