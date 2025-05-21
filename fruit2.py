import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="🍏Fruit Classification🍐", layout="centered")

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="fruit_classifier_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Get input and output details for the interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Apples', 'Grapes', 'Pineapple', 'Orange', 'Strawberry']

st.title("🍌Fruit Image Classifier")
st.write("Upload an image to classify the fruit.")

file = st.file_uploader("📸 Upload a fruit image", type=["jpg", "jpeg", "png", "bmp"])

def import_and_predict(image_data):
    size = input_details[0]['shape'][1:3]
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img_array = np.asarray(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction

if file is None:
    st.info("🖼️ Please upload a fruit image to proceed.")
else:
    image = Image.open(file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Classifying Fruit..."):
        prediction = import_and_predict(image)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = round(100 * np.max(prediction), 2)

    st.markdown(f"### 🌈 Predicted Fruit: **{predicted_class}** ({confidence}%)")

    if predicted_class == 'Apples':
        st.warning("🍎 Apples are a good source of fiber and low in calories, making them a heart-healthy snack.")
    elif predicted_class == 'Grapes':
        st.success("🍇 Grapes are rich in antioxidants, particularly resveratrol, which supports heart and brain health.")
    elif predicted_class == 'Pineapple':
        st.info("🍍 Pineapples are packed with vitamin C and bromelain, aiding immunity and digestion.")
    elif predicted_class == 'Orange':
        st.info("🍊 Oranges are loaded with vitamin C, supporting immune function and skin health.")
    elif predicted_class == 'Strawberry':
        st.info("🍓 Strawberries are low in calories but high in vitamin C, fiber, and powerful antioxidants.")
