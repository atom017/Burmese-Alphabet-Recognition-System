import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from playsound import playsound
from streamlit_drawable_canvas import st_canvas

# Load the model and character mapping
model = tf.keras.models.load_model('model5.h5')
character_dict = {0: 'က', 1: 'ခ', 2: 'ဂ', 3: 'ဃ', 4: 'င', 5: 'စ',
                  6: 'ဆ', 7: 'ဇ', 8: 'ဈ', 9: 'ည', 10: 'ဋ',
                  11: 'ဌ', 12: 'ဍ', 13: 'ဎ', 14: 'ဏ', 15: 'တ',
                  16: 'ထ', 17: 'ဒ', 18: 'ဓ', 19: 'န', 20: 'ပ',
                  21: 'ဖ', 22: 'ဗ', 23: 'ဘ', 24: 'မ', 25: 'ယ',
                  26: 'ရ', 27: 'လ', 28: 'ဝ', 29: 'သ', 30: 'ဟ', 31: 'ဠ', 32: 'အ'}

# Set up Streamlit UI
st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
st.title('Canvas')

# Canvas parameters
stroke_width = st.sidebar.slider("Stroke width: ", 5, 15, 5)
stroke_color = st.sidebar.color_picker("Stroke color hex:")
canvas_result = st_canvas(
    fill_color="#eee",
    stroke_width=stroke_width,
    update_streamlit=True,
    height=200,
    width=200,
    drawing_mode="freedraw",
    stroke_color=stroke_color,
    background_color="#fff",
    key="canvas",
)

def play_audio(index):
    """Play audio corresponding to the predicted character."""
    audio_path = os.path.join(os.getcwd(), 'audio_mp3', f"{index:02d}.mp3")
    playsound(audio_path)

def play_audio_st(index): 
    """Stream audio corresponding to the predicted character."""
    audio_path = os.path.join(os.getcwd(), 'audio_mp3', f"0{index}.mp3")
    st.audio(audio_path, format="audio/mp3")

def crop_image(img):
    """Crop the image to the bounding box of the drawn content."""
    pixels = np.asarray(img)
    coords = np.column_stack(np.where(pixels < 180))
    if coords.size == 0:  # No drawn content
        return img
    minx, miny = np.min(coords, axis=0)
    maxx, maxy = np.max(coords, axis=0)
    return img[minx - 2:maxx + 2, miny - 2:maxy + 2]

def reshape_dims(img):
    """Reshape the image for model input."""
    img_input = np.expand_dims(np.array(img, dtype="uint8"), axis=2)
    return np.expand_dims(img_input, axis=0)

def convert_RGBA_to_RGB(input_image):
    """Convert an RGBA image to RGB."""
    input_image_normed = input_image / 255
    alpha = input_image_normed[..., -1:]
    input_image_normed_rgb = input_image_normed[..., :-1]
    bg_normed = np.ones_like(input_image_normed_rgb)
    composite_normed = (1 - alpha) * bg_normed + alpha * input_image_normed_rgb
    return (composite_normed * 255).round().astype(np.uint8)

def predict_img(img):
    """Predict the character from the image."""
    x_input = reshape_dims(img)
    y_input_pred = model.predict(x_input)
    return np.argmax(y_input_pred, axis=1)[0]

def resize_image(img):
    """Resize the image to the expected input size."""
    return cv2.resize(img, (50, 50))

if canvas_result.image_data is not None:
    if st.button('Submit'):
        img = convert_RGBA_to_RGB(canvas_result.image_data)

        # Convert to grayscale and threshold
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh, blackAndWhiteImage = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

        # Define the kernel and erode the image
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(blackAndWhiteImage, kernel, iterations=1)

        img_cropped = crop_image(erosion)
        img_resized = resize_image(img_cropped)

        result = predict_img(img_resized)
        text_output = character_dict[result]
        st.text_input('Output', value=text_output)
        play_audio_st(result)  # Play audio for the result

