import os

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

x, y = [], []
for name in os.listdir('classes'):
        img = cv2.imread('classes/' + name, 0)
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        x.append(np.asarray(img, dtype=np.float64))
        y.append(name.split('_')[0])

x, y = np.array(x), np.array(y)
x_data = x.reshape((len(x), 150, 150, 1))
x_data = list(x_data)
for i in range(len(x_data)):
    x_data[i] = x_data[i].flatten()
x_data = np.array(x_data)

scaler = StandardScaler()
scaler.fit(x_data)
x_train = scaler.transform(x_data)

model = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=0)
model.fit(x_train, y)


# tsne = TSNE()
# x = tsne.fit_transform(x_train)
# import matplotlib.pyplot as plt
# cmap = {'fist': 'blue', 'down': 'black',
#              'up': 'grey', 'right': 'red', 'horizontal': 'green',
#              'left': 'brown', 'vertical': 'orange'}
# for x1, x2, c in zip(x[:, 0], x[:, 1], y):
#     plt.scatter(x1, x2, c=cmap[c], label=c, alpha=0.3, cmap='seabornqqqqqqqqqqq')
# import matplotlib.patches as mpatches
# plt.legend(handles=[mpatches.Patch(color=color, label=gest) for gest, color in cmap.items()])
# plt.grid()
# plt.show()


def predict_gesture(img):
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    x_test = np.array([np.asarray(img, dtype=np.float64)]).reshape((1, 150, 150, 1))
    x_test = x_test / 255
    x_test = list(x_test)
    for i in range(len(x_test)):
        x_test[i] = x_test[i].flatten()
    x_test = np.array(x_test)
    r = model.predict(x_test)
    return r[0]
