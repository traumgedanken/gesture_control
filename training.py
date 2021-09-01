import os

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier as Classifier
# from sklearn.neighbors import KNeighborsClassifier as Classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import classification_report


x, y = [], []
for name in tqdm(os.listdir("classes")):
    img = cv2.imread("classes/" + name, 0)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    x.append(np.asarray(img, dtype=np.float64))
    y.append(name.split("_")[0])

x, y = np.array(x), np.array(y)
x_data = x.reshape((len(x), 150, 150, 1))
x_data = list(x_data)
for i in range(len(x_data)):
    x_data[i] = x_data[i].flatten()
x_data = np.array(x_data)

scaler = StandardScaler()
scaler.fit(x_data)
x_train = scaler.transform(x_data)

model = Classifier(n_estimators=100, max_depth=30, random_state=0)
model.fit(x_train, y)


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


if __name__ == "__main__":
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x_train, y, test_size=0.5)

    model.fit(x_train, y_train)
    print(classification_report(y_test, model.predict(x_test)))

