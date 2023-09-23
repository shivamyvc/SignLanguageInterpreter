from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login,logout
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.image import image_utils
import nltk
import base64
import joblib
from django.http import JsonResponse
import cv2
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from keras.models import load_model
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from django.contrib.staticfiles import finders
from django.contrib.auth.decorators import login_required
import mediapipe as mp
import threading
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from .preprocess import rec_coords,to_edge,ExtractHandRegion



# Filter the list of possible words to only include valid English words
nltk.download('words')


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# hands=mp_hands.Hands(min_detection_confidence=0.8,
#                      min_tracking_confidence=0.5) 
model = load_model("C:\\Users\\Shubham Yadav\\Desktop\\Shivam\\Project\\ensemble\\cnn-rf\\cnn-rf.h5")
knn_model=joblib.load("C:\\Users\\Shubham Yadav\\Desktop\\Shivam\\Project\\ensemble\\cnn-rf\\rf_model.pkl")
# # lables= {0: '1', 1: '2', 2: '3', 3: '4',4: '5', 5: '6', 6: '7', 7: '8',8: '9', 9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F', 15: 'G',16: 'H', 17: 'I', 18: 'J', 19: 'K',20: 'L', 21: 'M', 22: 'N', 23: 'O',24: 'P', 25: 'Q', 26: 'R', 27: 'S',28: 'T', 29: 'U', 30: 'V', 31: 'W',32: 'X', 33: 'Y', 34: 'Z',}
lables= {
 0: 'A',
 1: 'B',
 2: 'C',
 3: 'D',
 4: 'E',
 5: 'F',
 6: 'G',
 7: 'H',
 8: 'I',
 9: 'J',
 10: 'K',
 11: 'L',
 12: 'M',
 13: 'N',
 14: 'O',
 15: 'P',
 16: 'Q',
 17: 'R',
 18: 'S',
 19: 'T',
 20: 'U',
 21: 'V',
 22: 'W',
 23: 'X',
 24: 'Y',
 25: 'Z'}
def home_view(request):
	return redirect('animation')

def about_view(request):
	return render(request,'about.html')

def video(request):
	return render(request,'video.html')

def contact_view(request):
	return render(request,'contact.html')

# @login_required(login_url="login")
def animation_view(request):
	if request.method == 'POST':
		text = request.POST.get('sen')
		#tokenizing the sentence
		text.lower()
		#tokenizing the sentence
		words = word_tokenize(text)

		tagged = nltk.pos_tag(words)
		tense = {}
		tense["future"] = len([word for word in tagged if word[1] == "MD"])
		tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
		tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]])
		tense["present_continuous"] = len([word for word in tagged if word[1] in ["VBG"]])



		#stopwords that will be removed
		stop_words = set(["mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn', 'do', "you've",'off', 'for', "didn't", 'm', 'ain', 'haven', "weren't", 'are', "she's", "wasn't", 'its', "haven't", "wouldn't", 'don', 'weren', 's', "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll", "should've", 'a', 'then', 'the', 'mustn', 'i', 'nor', 'as', "it's", "needn't", 'd', 'am', 'have',  'hasn', 'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn', "doesn't", 'll', 'an', 'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn', 'needn', "shan't", 'isn', 'been', 'such', 'shan', "shouldn't", 'aren', 'being', 'were', 'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"])



		#removing stopwords and applying lemmatizing nlp process to words
		lr = WordNetLemmatizer()
		filtered_text = []
		for w,p in zip(words,tagged):
			if w not in stop_words:
				if p[1]=='VBG' or p[1]=='VBD' or p[1]=='VBZ' or p[1]=='VBN' or p[1]=='NN':
					filtered_text.append(lr.lemmatize(w,pos='v'))
				elif p[1]=='JJ' or p[1]=='JJR' or p[1]=='JJS'or p[1]=='RBR' or p[1]=='RBS':
					filtered_text.append(lr.lemmatize(w,pos='a'))

				else:
					filtered_text.append(lr.lemmatize(w))


		#adding the specific word to specify tense
		words = filtered_text
		temp=[]
		for w in words:
			if w=='I':
				temp.append('Me')
			else:
				temp.append(w)
		words = temp
		probable_tense = max(tense,key=tense.get)

		if probable_tense == "past" and tense["past"]>=1:
			temp = ["Before"]
			temp = temp + words
			words = temp
		elif probable_tense == "future" and tense["future"]>=1:
			if "Will" not in words:
					temp = ["Will"]
					temp = temp + words
					words = temp
			else:
				pass
		elif probable_tense == "present":
			if tense["present_continuous"]>=1:
				temp = ["Now"]
				temp = temp + words
				words = temp


		filtered_text = []
		for w in words:
			path = w + ".mp4"
			f = finders.find(path)
			#splitting the word if its animation is not present in database
			if not f:
				for c in w:
					filtered_text.append(c)
			#otherwise animation of word
			else:
				filtered_text.append(w)
		words = filtered_text


		return render(request,'animation.html',{'words':words,'text':text})
	else:
		return render(request,'animation.html')




def preprocessPredict(image):
	# image[y_min:y_max,x_min:x_max]
	img=cv2.resize(image, (150,150))
	Gray_Img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	_,threshold_Img = cv2.threshold(Gray_Img,240,255,cv2.THRESH_BINARY_INV)
	canny_Img = cv2.Canny(threshold_Img,90,100)
	cv2.imwrite("Img.jpg",canny_Img)
	img_array = np.array(canny_Img)
	img_array = np.expand_dims(img_array, axis=0)
	cnn_features = model.predict(img_array)
	predicted_label = knn_model.predict(cnn_features)
	print('Predicted Sign Label:', lables[predicted_label[0]])
	
	
def two_way(request):
	return render(request, 'mp_detect_hand.html')
@csrf_exempt
def get_frames(request):
	if request.method == "POST":
		# Get the base64-encoded image data from the request
		image_data = request.POST.get("image_data")
		# frame_cn=request.POST.get("frame_data")
		# print(len(frame_cn))
		# Decode the base64-encoded image data and convert it into a NumPy array
		image_bytes = base64.b64decode(image_data.split(",")[1])
		
		image_array = np.frombuffer(image_bytes, dtype=np.uint8)
		image = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)
		preprocessPredict(image)
		return HttpResponse(status=200)


def classify(img):
        class_lbl=None
        # img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img=cv2.resize(img, (150,150))
        im = image_utils.img_to_array(img)
        img_array = np.array(im)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        cnn_features = model.predict(img_array)
        predicted_label = knn_model.predict(cnn_features)
        return lables[predicted_label[0]]
# Function that returns a generator object that yields frames of the video
def gen_frames():
	cap = cv2.VideoCapture(0)
	with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
		while True:
			success, image = cap.read()
			if not success:
				break
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			# Flip on horizontal
			image = cv2.flip(image, 1)
			
			# Set flag
			image.flags.writeable = False
			
			# Detections
			results = hands.process(image)
			
			# Set flag to true
			image.flags.writeable = True
			
			# RGB 2 BGR
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			
			# Detections
			# print(results)
			h,w,_=image.shape
			# Rendering results

			if results.multi_hand_landmarks:
				mult_coord=[]
				for num, hand in enumerate(results.multi_hand_landmarks):
					mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
					mp_drawing.DrawingSpec(color=(255,255,255), thickness=3,circle_radius=0),
					mp_drawing.DrawingSpec(color=(255,255,255), thickness=3,circle_radius=0) )
					if len(results.multi_handedness)==2:
						mult_coord.append(rec_coords(hand,w,h))
					else:
						x_min, y_min, x_max, y_max=rec_coords(hand,w,h)
						cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
						both_hands=image[y_min:y_max,x_min:x_max]
						tr_img=cv2.resize(to_edge(both_hands), (250,250))
					
						class_lbl=classify(tr_img)
						cv2.putText(image,class_lbl, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
				if len(mult_coord)!=0:
					x_min=min(mult_coord[0][0],mult_coord[1][0])
					y_min=min(mult_coord[0][1],mult_coord[1][1])
					x_max=max(mult_coord[0][2],mult_coord[1][2])
					y_max=max(mult_coord[0][3],mult_coord[1][3])
					cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
					single_hand=image[y_min:y_max,x_min:x_max]
					tr_img=cv2.resize(to_edge(single_hand), (250,250))
					class_lbl=classify(tr_img)
					cv2.putText(image,class_lbl, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

			ret, buffer = cv2.imencode('.jpg', image)
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
			b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
	cap.release()

# Decorator to compress the video stream using gzip
@gzip.gzip_page
def live_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')