import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from keras.preprocessing.image import load_img, img_to_array

# Loading data
def load_images(directory, images, labels, label_to_add):
    for sub_directory in os.listdir(directory):
        image_path = os.path.join(directory, sub_directory)
        image = load_img(image_path, target_size=(224, 224))
        image_gray = image.convert('L')
        image_array = img_to_array(image_gray)
        image_array = image_array.reshape(image_array.shape[0], -1)

        # Appending the images as X_train and labels as Y_train
        images.append(image_array)
        labels.append(label_to_add)
    
    return images, labels

# Setting path of images
dataset_happy_person_path = "Happy person"
dataset_sad_person_path = "Sad person"

# Setting data of X_train and Y_train
X = []
Y = []
X, Y = load_images(dataset_happy_person_path, X, Y, 1)
X, Y = load_images(dataset_sad_person_path, X, Y, 0)

X = np.array(X)
X = X.reshape(X.shape[0], -1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Initialization of the model
model = LogisticRegression(solver='liblinear')

# Training with data
model.fit(X_train, Y_train)

# Testing the accuracy
accuracy_score = accuracy_score(Y_test, model.predict(X_test))
print(f"The accuracy of the model is {accuracy_score}")

# Testing the model
image_testing_path = "happy_person.jpg"
image = load_img(image_testing_path, target_size=(224, 224))
image_gray = image.convert('L')
image_array = img_to_array(image_gray)
image_array = image_array.reshape(1, -1)
output = model.predict(image_array)

# Printing output
if output[0] == 0:
    print(f"The person is sad.")
else:
    print("The person is happy.")

