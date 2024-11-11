import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

# Set parameters
device = torch.device('cpu')

# set seed
torch.manual_seed(0)
np.random.seed(1)

# define dataset parameters
N    = 20
dim  = 2
data = np.random.randn(N,dim)

# save data
with open('training_data.npy', 'wb') as f:
    np.save(f, data)

# plot data
plt.figure(figsize=(2,2))
fig = voronoi_plot_2d(Voronoi(data))
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.show()