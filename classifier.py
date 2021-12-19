import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']

y = pd.read_csv('labels.csv')['labels']

print(pd.Series(y).value_counts())

classes = ['A', 'B' 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z']

nclasses = len(classes)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)

model = LogisticRegression(solver='saga', multi_class='multinomial').fit(x_train, y_train)

def get_prediction(image):
    img = Image.open(image)
    img_bw = img.convert('L')
    image_resized = img_bw.resize((22,30), Image.ANTIALIAS)
    pixels = 20
    min_pixel = np.percentile(image_resized, pixels)
    image_inverted = np.clip(image_resized - min_pixel, 0,255)
    max_pixel = np.max(image_resized)
    image_inverted = np.asarray(image_inverted)/max_pixel
    testsample = np.array(image_inverted).reshape(1,660)
    testprediction = model.predict(testsample)

    return testprediction[0]