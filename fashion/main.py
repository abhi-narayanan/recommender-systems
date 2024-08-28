import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

# Read features list and filenames
feature_arrays = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = np.array(pickle.load(open('filenames.pkl', 'rb')))

# Create ResNet model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
        model,
        GlobalMaxPooling2D()
])

def feature_extraction(img_path, model):
    # Image processing
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    # Prediction and normalization
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(feature, feature_array):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_array)

    _, indices = neighbors.kneighbors([feature])

    return indices

st.title("Fashion Recommender System")

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display image
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # Extract features
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        # Recommendation
        indices = recommend(features, feature_arrays)
        # Show recommendations
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
            
    else:
        st.header("Some error occured in file upload..")