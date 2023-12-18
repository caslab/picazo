from keras.preprocessing.image import load_img, img_to_array
from tempfile import NamedTemporaryFile
from PIL import Image

import numpy as np
import streamlit as st
import tensorflow as tf

###############################################################################
# CONSTANTS
###############################################################################

APP_LOGO = 'img/picazo_logo.png'
MAX_IMGS = 5
ALLOWED_IMG_EXTENSIONS = ['.jpg', '.jpeg',
                          '.png', '.gif', '.tiff', '.bmp', '.webp']
IMG_SIZE = (32, 32)

MODEL = {
    'LeNet-5': 'picazo_lenet_5.keras',
    'EfficientNet': 'picazo_efficientnet.keras',
}

LABELS_MAP = {
    0: 'Real',
    1: 'Fake'
}

TEMP_IMGS_DIRECTORY = './streamlit_uploader'

###############################################################################
# SESSION VARIABLES
###############################################################################

if 'model_loaded' not in st.session_state:
    print('Adding model_loaded to session_state')
    st.session_state.model_loaded = ''
if 'classifier' not in st.session_state:
    print('Adding classifier to session_state')
    st.session_state.model_loaded = None

###############################################################################
# HELPER FUNCTIONS
###############################################################################

# Stores file data in a temporary file and returns its file path.
def fetch_tmp_path(file_data):
    with NamedTemporaryFile(delete=False) as temp:
        temp.write(file_data.read())
        temp.flush()
        return temp.name
    
# Returns (width, heigh, channels).
def image_shape(image):
    size_tuple = tuple(image.size)
    channels_tuple = (len(image.getbands()),)

    return size_tuple + channels_tuple

# Fits image to model's input requirements.
def prepare_image(image_path, image_size):
    # Read temporary image from disk
    img = load_img(path=image_path,
                   color_mode='rgb',
                   target_size=image_size,
                   keep_aspect_ratio=True)
    
    # Convert image to array
    img = img_to_array(img)

    # Add an extra dimension to match the model's input shape
    img = img.reshape(1,
                      img.shape[0], img.shape[1], img.shape[2])
    
    return img

# Imports and image and invokes the classifier.
def import_and_predict(image, model):
    img = prepare_image(image, IMG_SIZE)
    prediction = model.predict(img)
    return prediction

# Spinner (model load)
def load_model(option):
    model = tf.keras.models.load_model(MODEL[option])
    return model

###############################################################################
# UI
###############################################################################

# Site metadata
icon = Image.open(APP_LOGO)
st.set_page_config(
    page_title="PiCazo - GenAI Image Detector",
    page_icon=icon,
    layout="centered",
    initial_sidebar_state="auto",
)

# Logo and header
left_co, right_co = st.columns(2)
with left_co:
    st.image(APP_LOGO)
with right_co:
    """
    # PiCazo
    ## GenAI Image Detector
    [![](https://img.shields.io/github/stars/caslab/picazo?style=social)](https://github.com/caslab/picazo) &nbsp; [![](https://img.shields.io/twitter/follow/victor_caslab?style=social)](https://twitter.com/victor_caslab)
    """

# Subheader
st.write(
    """
    👋 Welcome to PiCazo! This app detects Real vs GenAI/Fake images using a Convolutional Neural Network.
    
    Powered by [Artifact](https://github.com/awsaf49/artifact), a large-scale dataset with artificial and factual images for synthetic image detection.    
    """
)

st.divider()

# Body
menu1, menu2 = st.columns([1, 3])

with menu1:
    option = st.radio(
        'Step 1. Select classifier',
        key='disabled',
        options=MODEL.keys())

    if (st.session_state.model_loaded != option):
        # Load classifier from a saved backup
        classifier = load_model(option)
        st.session_state.model_loaded = option
        st.session_state.classifier = classifier

    if (st.session_state.model_loaded == option):
        st.write(':rainbow[Model loaded.]')

with menu2:
    uploaded_files = st.file_uploader(
        label='Step 2. Choose up to five images.',
        type=ALLOWED_IMG_EXTENSIONS,
        accept_multiple_files=True)

st.divider()

# Response panel
st.columns(1)
if st.session_state.classifier is not None:
    if uploaded_files is not None:
        if (len(uploaded_files) > 0):
            # Read uploaded images
            image_paths = [fetch_tmp_path(i) for i in uploaded_files[:min(
                len(uploaded_files), MAX_IMGS)]]
            images = [Image.open(i) for i in uploaded_files[:min(
                len(uploaded_files), MAX_IMGS)]]
            
            # Render content
            img_cols = st.columns(len(images))
            for i in range(len(images)):
                # Run the model
                print('Evaluating image: ', uploaded_files[i].name)
                predictions = import_and_predict(
                    image_paths[i], st.session_state.classifier)
                score_value = np.max(predictions)
                label = np.argmax(predictions)
                label_name = LABELS_MAP.get(label.item(), 'Unknown')
                print('Predictions: ', predictions)
                print('Score max: ', score_value)
                print('Argmax: ', label)
                print('Label: ', label_name)

                # Visualize predictions
                with img_cols[i]:
                    st.image(images[i], use_column_width=True)
                    st.caption(uploaded_files[i].name)
                    message = '''
                    {label}  
                    Conf: {score:.2%}'''.format(
                        label=label_name, score=score_value)
                    if label == 0:
                        st.success(message)
                    else:
                        st.error(message)
