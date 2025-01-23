# Import necessary packages
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from corr import list_band

# Load data and ground truth
data = loadmat('Dataset/PaviaU.mat')['paviaU']
gt = loadmat('Dataset/PaviaU_gt.mat')['paviaU_gt']

# To use Salinas dataset, uncomment the line
# data = loadmat('Dataset/Salinas_corrected.mat')['salinas_corrected']
# gt = loadmat('Dataset/Salinas_gt.mat')['salinas_gt']

num = len(np.unique(gt)) # get unique classes
x = data[:, :, list_band] # only extract selected bands from dataset
y = gt.ravel()
x = x.reshape(-1, len(list_band))

# remove the background class
mask = y>0
x = x[mask]
y = y[mask]

# Standardize dataset before analysis
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split the dataset into train and test pixels, in our paper we kept test size at 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

"""
SVM Classifier
"""

svm = SVC(kernel = 'rbf')
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

"""
Results
"""

conf_test = confusion_matrix(y_test, y_pred)
print(f'Kappa Coeff: {cohen_kappa_score(y_test, y_pred):.2f}')
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print(classification_report(y_test, y_pred))
print(conf_test)

"""
Plotting the classification map
"""

y_full = svm.predict(x)
img = np.zeros_like(gt)
img[mask.reshape(gt.shape)] = y_full
colors = ['white'] + plt.cm.jet(np.linspace(0, 1, num - 1)).tolist()
custom = ListedColormap(colors)
plt.figure(figsize = (10, 10))
plt.imshow(img, cmap = custom)
plt.colorbar(ticks = range(num))
plt.title(f'Classification Map using Proposed method')
plt.savefig(f'Figures/PA/map_ccsvm_{len(list_band)}_bands', bbox_inches = 'tight')
# plt.savefig(f'Figures/SA/map_ccsvm_{len(list_band)}_bands', bbox_inches = 'tight')
plt.show()
