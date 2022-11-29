import streamlit as st
import cv2
import numpy as np
import pickle
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

st.title('Word Canvas')

stroke_width = st.sidebar.slider("Stroke width: ", 5, 15, 5)
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

# Create a canvas component
canvas_result = st_canvas(
    fill_color= "#eee", #rgba(255, 165, 0,0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    update_streamlit=True,
    height=200,
	width=600,
	background_color="#fff",
    drawing_mode="freedraw",
    key="canvas",
)

def play_audio(index):
	my_path = os.getcwd()
	audio_path = my_path+'/audio_mp3/'
	if index > 9:
		audio_file = audio_path + '0' + str(index) + '.mp3'
	else:
		audio_file = audio_path + '00' + str(index) + '.mp3'

	playsound(audio_file)

def get_contours(thresh_img):
	contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# contours = contours[0] if len(contours) == 2 else contours[1]
	sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

	alphabets = []
	print(len(contours))
	min_area = 100
	for cntr in sorted_contours:
		area = cv2.contourArea(cntr)
		# if area > min_area:
		x, y, w, h = cv2.boundingRect(cntr)
		final_image = canvas_img[y:y + h, x:x + w]
		alphabets.append(final_image)
	return alphabets

def get_thresh_img(gray_img):

	ret, imgt = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY_INV)

	return imgt

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
			#g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			resize_img = cv2.resize(img, dsize=(50, 50))
			return resize_img
	except:
		print('error resizing image: ',)


canvas_img = None
if canvas_result.image_data is not None:
    canvas_img = canvas_result.image_data

if st.button('Submit'):
	img = convert_RGBA_to_RGB(canvas_img)
	g_img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	thresh_img = get_thresh_img(g_img)
	contour_imgs = get_contours(thresh_img)
	text = ''
	result_arr = []
	for i in range(len(contour_imgs)):
		img_3channel = convert_RGBA_to_RGB(contour_imgs[i])
		gray_img = cv2.cvtColor(img_3channel, cv2.COLOR_BGR2GRAY)
		(thresh, blackAndWhiteImage) = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
		img_resized = resize_image(blackAndWhiteImage)
		result=predict_img(img_resized)
		result_arr.append(result)
		text=text+character_dict[result]
	st.text_input('output', value=text)
	for i in range(len(result_arr)):
		play_audio(result_arr[i]+1)



