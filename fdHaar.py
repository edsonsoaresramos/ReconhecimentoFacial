import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import os

subjects = ["", "Edson Soares", "Sophia Soares", "Arthur Soares", "Julia Pereira"]

def detect_faces(f_cascade, colored_img):
	img_copy = colored_img.copy()

	gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

	faces = f_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

	return faces


def detect_face(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

	if (len(faces) == 0):
		return None, None

	(x, y, w, h) = faces[0]

	return gray[y:y + w, x:x + h], faces[0]

def prepare_training_data(data_folder_path):
	dirs = os.listdir(data_folder_path)

	faces = []
	labels = []

	for dir_name in dirs:
		if not dir_name.startswith("s"):
			continue

		label = int(dir_name.replace("s", ""))

		subject_dir_path = data_folder_path + "/" + dir_name

		subject_images_names = os.listdir(subject_dir_path)

		for image_name in subject_images_names:
			if (image_name.startswith(".")):
				continue;

			image_path = subject_dir_path + "/" + image_name

			image = cv2.imread(image_path)

			cv2.imshow("Imagem em aprendizado: ", cv2.resize(image, (400, 500)))
			cv2.waitKey(100)

			face, rect = detect_face(image)

			if face is not None:
				faces.append(face)
				labels.append(label)

	cv2.destroyAllWindows()
	cv2.waitKey(1)
	cv2.destroyAllWindows()

	return faces, labels

print('Preparando os dados para reconhecimento facial...')

faces, labels = prepare_training_data("training-data")

print("Dados preparados")

print('Total de rostros encontrados: ' + str(len(faces)))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
	retorno = []

	img = test_img.copy()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
	faces = detect_faces(face_cascade, img)

	if (len(faces) == 0):
		return None, None

	for face in faces:
		(x, y, w, h) = face

		face_, rect = gray[y:y + w, x:x + h], face

		label, confidence = face_recognizer.predict(face_)

		label_text = subjects[label]

		draw_rectangle(img, rect)
		draw_text(img, label_text, rect[0], rect[1]-5)

	return img

print("Identificando imagens...")

#test_img1 = cv2.imread("fotos-testes/test1.jpg")
test_img1 = cv2.imread("fotos-testes/Arraial.jpeg")

predicted_img1 = predict(test_img1)
print("Previsao Finalizada...")

#cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow('Cabo Frio', predicted_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()