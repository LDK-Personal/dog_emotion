import cv2
import dlib
from imutils import face_utils, rotate
import numpy as np
import math
from keras.models import load_model

# ---------------------------------------------------------
# Configuration
img_width, img_height = 100, 100
picSize = 200
rotation = True
pathDet = 'C:\\Users\\owner\\Desktop\\Dog-Emotions-master\\faceDetectors\\dogHeadDetector.dat'
pathPred = 'C:\\Users\\owner\\Desktop\\Dog-Emotions-master\\faceDetectors\\landmarkDetector.dat'

# Load detectors and model
detector = dlib.cnn_face_detection_model_v1(pathDet)
predictor = dlib.shape_predictor(pathPred)
model = load_model('C:\\Users\\owner\\Desktop\\Dog-Emotions-master\\models\\classifierRotatedOn100Ratio90Epochs100.h5')

# Helper function for preprocessing
def preprocess(orig):
    imageList, faces = [], []
    if orig is not None:
        height, width, _ = orig.shape
        ratio = picSize / height
        resized = cv2.resize(orig, None, fx=ratio, fy=ratio)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)

        for i, d in enumerate(dets):
            x1, y1 = int(d.rect.left() / ratio), int(d.rect.top() / ratio)
            x2, y2 = int(d.rect.right() / ratio), int(d.rect.bottom() / ratio)
            shape = face_utils.shape_to_np(predictor(gray, d.rect))
            points = np.array([shape[i-1] for i in [3, 4, 6]]) / ratio

            if rotation:
                dy, dx = points[2, 1] - points[0, 1], points[2, 0] - points[0, 0]
                angle = math.degrees(math.atan2(dy, dx))
                rotated = rotate(orig, -angle)
            else:
                rotated = orig.copy()

            cv2.polylines(orig, [points.astype(np.int32)], True, (0, 255, 0), 1)
            cv2.rectangle(orig, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cropped = rotated[y1:y2, x1:x2]
            resized_face = cv2.resize(cropped, (img_width, img_height))
            gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            processed_face = np.expand_dims(np.expand_dims(gray_face, -1), 0)
            imageList.append(processed_face)
            faces.append((x1, y1))

    return orig, imageList, faces

# Analyze facial emotions function
def analyze_facial_emotions(image):
    marked_image, processed_faces, face_locations = preprocess(image)
    if processed_faces:
        predictions = [model.predict(face)[0] for face in processed_faces]
        emotions = [np.argmax(pred) for pred in predictions]  # Assuming your model returns category indices
        probabilities = [np.max(pred) for pred in predictions]
        return marked_image, list(zip(emotions, face_locations, probabilities))
    return marked_image, []

# ---------------------------------------------------------
# Main function to start the webcam and analyze faces
def face_recognition():
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if ret:
            marked_frame, results = analyze_facial_emotions(frame)
            for emotion, (x, y), prob in results:
                cv2.putText(marked_frame, f"{emotion}: {prob:.2f}%", (x, y), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Dog Emotion Recognition', marked_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

face_recognition()
