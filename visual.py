# Import necessary packages
from scipy.io import loadmat
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import seaborn as sns

# Load desired dataset (make sure it is in .mat format)
# To use Salinas dataset, uncomment the line
data = loadmat('Dataset/PaviaU.mat')['paviaU']
gt = loadmat('Dataset/PaviaU_gt.mat')['paviaU_gt']
# data = loadmat('Dataset/Salinas_corrected.mat')['salinas_corrected']
# gt = loadmat('Dataset/Salinas_gt.mat')['salinas_gt']

print(f"Dataset Shape: {data.shape}")
print(f"Ground Truth Shape: {gt.shape}")

"""
To plot a few random image bands
"""

def plot_ran_band(n, x):
  
  for i in range(n):
    
    band = random.randint(0, x.shape[2] - 1)
    img = x[:, :, band]
    plt.figure(figsize = (10, 8))
    plt.imshow(img, cmap = 'jet')
    plt.colorbar(label = 'Reflectance')
    plt.title(f'Band id {band}')
    plt.savefig(f'Figures/PA/{band}id', bbox_inches = 'tight')
    # plt.savefig(f'Figures/SA/{band}id', bbox_inches = 'tight')
    plt.show()
  return 0

"""
To plot the Ground-Truth
"""

def plot_gt(y):
  unique = np.unique(y)
  num = len(unique)
  print(f'Number of unique classes: {len(unique)}')
  print(f'Unique classes: {unique}')
  
  colors = ['white'] + plt.cm.jet(np.linspace(0, 1, num - 1)).tolist()
  custom = ListedColormap(colors)
  
  plt.figure(figsize = (10, 10))
  plt.imshow(y, cmap = custom)
  plt.colorbar(ticks = range(num), label = 'Class Labels')
  plt.title('Pavia University Ground Truth')
  plt.savefig('Figures/PA/gt', bbox_inches = 'tight')
  # plt.savefig(f'Figures/SA/gt', bbox_inches = 'tight')
  plt.show()
  return 0

"""
To plot the correlation between bands
"""
def plot_corr(x):
  x = x.reshape(-1, x.shape[2])
  corr = np.corrcoef(x, rowvar = False)
  plt.figure(figsize = (20, 20))
  sns.heatmap(corr, cmap = 'coolwarm', annot = False, fmt = '.2f')
  plt.title('Correlation')
  plt.xlabel('Bands')
  plt.ylabel('Bands')
  plt.savefig('Figures/PA/corr', bbox_inches = 'tight')
  # plt.savefig('Figures/SA/corr', bbox_inches = 'tight')
  plt.show()
  return 0
  
num = 5 # Can change to a desired number
plot = plot_ran_band(num, data)
ground = plot_gt(gt)
cc = plot_corr(data)
