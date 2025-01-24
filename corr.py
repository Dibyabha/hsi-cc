# Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

# Load dataset
data = loadmat('Dataset/PaviaU.mat')['paviaU']
gt = loadmat('Dataset/PaviaU_gt.mat')['paviaU_gt']


# To use Salinas dataset, uncomment the line
# data = loadmat('Dataset/Salinas_corrected.mat')['salinas_corrected']
# gt = loadmat('Dataset/Salinas_gt.mat')['salinas_gt']

"""
Calculate Correlation Coefficient and store as a matrix
"""

mask = gt>0
data = data[mask]
scaler = StandardScaler()
data = scaler.fit_transform(data)

corr = np.corrcoef(data, rowvar = False)
df = pd.DataFrame(corr)
df.to_csv('PA Correlation Data.csv', index = False)
# df.to_csv('SA Correlation Data.csv', index = False)

"""
Calculate Average Band Correlation (ABC) of each band
"""

x = pd.read_csv('PA Correlation Data.csv', header = None).values
# x = pd.read_csv('SA Correlation Data.csv', header = None).values
x = x[1:, :]

def scores(matrix):
    absol = np.abs(matrix)
    mask = np.eye(absol.shape[0], dtype = bool)
    score = np.mean(absol - mask, axis = 0)
    return score
    

def plot(score):
    plt.figure(figsize = (25, 25))
    plt.bar(range(len(score)), score)
    plt.xlabel('Bands')
    plt.ylabel('Correlation Strength')
    plt.title('Correlation Strength of Bands')
    plt.tight_layout()
    plt.savefig('Figures/PA/corr_strength', bbox_inches = 'tight')
    # plt.savefig('Figures/SA/corr_strength', bbox_inches = 'tight')
    plt.show()

score = scores(x)
plot(score)
df = pd.DataFrame(np.round(score, decimals = 3), columns = ['Correlation Strength'])
df.to_csv('PA Scores.csv', index = False)
# df.to_csv('SA Scores.csv', index = False)

"""
Use threshold = 0.65 to select the bands
"""

x = pd.read_csv('PA Scores.csv', header = None).values
# x = pd.read_csv('SA Scores.csv', header = None).values
x = x[1:, :]

bands = x.shape[0]
t = 0.65
list_band = []
for i in range(bands):
    row = x[i, :].astype(float)
    
    if row <= t:
        list_band.append(i)

print("Selected Bands: ", list_band)
print("Number of Bands selected: ", len(list_band))




