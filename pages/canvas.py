import streamlit as st
from io import StringIO
import cv2
import numpy as np

import tensorflow as tf
from playsound import playsound
import os
from os import path



st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
model = tf.keras.models.load_model('model5.h5')

character_dict = {0:'က', 1:'ခ', 2:'ဂ', 3:'ဃ', 4:'င', 5:'စ',
                  6:'ဆ', 7:'ဇ', 8:'ဈ', 9:'ည', 10:'ဋ',
                  11:'ဌ', 12:'ဍ', 13:'ဎ', 14:'ဏ', 15:'တ',
                  16:'ထ', 17:'ဒ', 18:'ဓ', 19:'န', 20:'ပ',
                  21:'ဖ', 22:'ဗ', 23:'ဘ', 24:'မ', 25:'ယ',
                  26:'ရ', 27:'လ', 28:'ဝ', 29:'သ', 30:'ဟ', 31:'ဠ', 32:'အ'}


from streamlit_drawable_canvas import st_canvas

st.title('Canvas')
# Specify canvas parameters in application


stroke_width = st.sidebar.slider("Stroke width: ", 5, 15, 5)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")

# Create a canvas component
canvas_result = st_canvas(
    fill_color= "#eee", #rgba(255, 165, 0,0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    update_streamlit=True,
    height=200,
	width=200,
    drawing_mode="freedraw",
	stroke_color=stroke_color,
	background_color="fff",
    key="canvas",
)

def play_audio(index):
	my_path = os.getcwd()
	audio_path = my_path+'/audio_mp3/'
	if index > 9:
		audio_file = audio_path+'0'+str(index)+'.mp3'
	else:
		audio_file = audio_path + '00' + str(index) + '.mp3'


	
	playsound(audio_file)
	


def crop_image(img):
	pixels = np.asarray(img)
	coords = np.column_stack(np.where(pixels < 180))

	minx = min(coords[:, 0])
	miny = min(coords[:, 1])
	maxx = max(coords[:, 0])
	maxy = max(coords[:, 1])
	crop_img = img[minx - 2:maxx + 2, miny - 2:maxy + 2]
	return crop_img

def reshape_dims(img):
	img_input = np.array(img,dtype="uint8")
	img_input = np.expand_dims(img_input, axis=2)
	x_input = np.expand_dims(img_input, axis=0)
	return x_input


def convert_RGBA_to_RGB(input_image):
	# Takes an RGBA image as input

	# Based on the following chat with user Andras Deak
	## https://chat.stackoverflow.com/transcript/message/55060299#55060299

	input_image_normed = input_image / 255  # shape (nx, ny, 4), dtype float
	alpha = input_image_normed[..., -1:]  # shape (nx, ny, 1) for broadcasting
	input_image_normed_rgb = input_image_normed[..., :-1]  # shape (nx, ny, 3)
	# bg_normed = np.zeros_like(red_normed_rgb)  # shape (nx, ny, 3) <-- black background
	bg_normed = np.ones_like(input_image_normed_rgb)  # shape (nx, ny, 3) <-- white background
	composite_normed = (1 - alpha) * bg_normed + alpha * input_image_normed_rgb
	composite = (composite_normed * 255).round().astype(np.uint8)
	return composite

def predict_img(img):
	x_input = reshape_dims(img)
	y_input_pred = model.predict([x_input])
	predict_result = np.argmax(y_input_pred, axis=1)
	return predict_result[0]

def prediction_to_char(pred_result):
	return character_dict[pred_result]

def resize_image(img):
	try:
		if img.shape == (0, 0):
			print('No image data : ', )
		else:

			g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			resize_img = cv2.resize(g_img, dsize=(50, 50))
			return resize_img
	except:
		print('error resizing image: ',)



canvas_img = None
if canvas_result.image_data is not None:
    canvas_img = canvas_result.image_data

if st.button('Submit'):
	img = convert_RGBA_to_RGB(canvas_img)
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	(thresh, blackAndWhiteImage) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
	# define the kernel
	kernel = np.ones((5, 5), np.uint8)

	# erode the image
	erosion = cv2.erode(blackAndWhiteImage, kernel,
						iterations=1)
	img_cropped = crop_image(erosion)
	img_resized = resize_image(img_cropped)

	text=''
	result=predict_img(img_resized)
	text=character_dict[result]
	st.text_input('output', value=text)
	play_audio(result+1)


