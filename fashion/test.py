import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

feature_arrays = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = np.array(pickle.load(open('filenames.pkl', 'rb')))

print(feature_arrays.shape)

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
        model,
        GlobalMaxPooling2D()
])

# Image processing
img = image.load_img("test/shirt.jpg", target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)

# Prediction and normalization
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_arrays)

distances, indices = neighbors.kneighbors([normalized_result])

print(indices)

for file in indices[0]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)