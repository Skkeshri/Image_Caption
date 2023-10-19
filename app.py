import streamlit as st
import tensorflow as tf
#from tensorflow import keras
import random
from PIL import Image, ImageOps
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
import pickle
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Image Caption Generation",
    page_icon = ":mango:",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


with st.sidebar:
        st.image('image.png')
        st.title("Interactive Image Captioning")
        st.subheader("Translate visuals into words seamlessly with our Image Captioning tool. Upload an image and watch as our advanced AI narrates the story behind it.")

             
        
def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            
            return key
        

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model2():
    model=tf.keras.models.load_model('mango_model.h5')
    return model

with st.spinner('Model is being loaded..'):
    # Load the VGG16 model for feature extraction
    base_model = VGG16(weights='imagenet')
    model_vgg16 = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    # Load the trained captioning model
    caption_model_path = 'caption_model.h5'
    caption_model = load_model(caption_model_path)


st.write("""
         # Transforming Pixels into Poetry: Interactive Image Captioning
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
max_length = 38 

def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction


def generate_caption(model, tokenizer, image, max_length):
    in_text = '<start>'
    image = image.reshape((1, -1))  # Reshape the image features to fit the model's expected input shape
    
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='pre')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, "?")
        if word == '<end>':
            break
        in_text += ' ' + word
    
    # Remove <start> and <end> tokens for the final output
    final_caption = in_text.replace('<start> ', '').replace(' <end>', '')
    return final_caption

def preprocess_image_uploaded(file, target_size=(224, 224)):
    # Use PIL to open the image from the uploaded file object
    img = Image.open(file).resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def extract_image_features(model, image_array):
    features = model.predict(image_array)
    return features[0]

def caption_uploaded_image(file, tokenizer, max_length):
    # Preprocess the uploaded image
    image_array = preprocess_image_uploaded(file)
    # Extract features using VGG16
    image_features = extract_image_features(model_vgg16, image_array)
    # Generate the caption using the trained caption model
    caption = generate_caption(caption_model, tokenizer, image_features, max_length)
    return caption

tokenizer_path = 'tokenizer.pkl'
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    # predictions = import_and_predict(image, model)
    image_path = '/content/dog_standing.jpg'
    final_caption = caption_uploaded_image(file, tokenizer, max_length)
    st.balloons()
    st.sidebar.success(final_caption)
