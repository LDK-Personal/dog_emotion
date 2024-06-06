import argparse
import cv2
import dlib
import math
import os
import json
import numpy as np
import imutils
from imutils import face_utils
from keras.models import load_model
from helper import No_Preprocessing
import matplotlib.pyplot as plt

# Constants
img_width = 100
img_height = 100
picSize = 200
rotation = True

# Paths to face and landmark detectors
pathDet = 'C:\\Users\\owner\\Desktop\\Dog-Emotions-master\\faceDetectors\\dogHeadDetector.dat'
pathPred = 'C:\\Users\\owner\\Desktop\\Dog-Emotions-master\\faceDetectors\\landmarkDetector.dat'

# Initialize face and landmark detectors
detector = dlib.cnn_face_detection_model_v1(pathDet)
predictor = dlib.shape_predictor(pathPred)

# Initialize helper class
helper = No_Preprocessing(img_width, img_height)

# Load the model
model = load_model('C:\\Users\\owner\\Desktop\\Dog-Emotions-master\\saveclassifier.keras')

def process_image(file_path):
    # Read image from path
    orig = cv2.imread(file_path)

    if orig is None:
        print(f"Error: Unable to load image from path {file_path}")
        return None

    if orig is not None and orig.any():
        # Resize image
        height, width, _ = orig.shape
        ratio = picSize / height
        image = cv2.resize(orig, None, fx=ratio, fy=ratio)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        dets = detector(gray, upsample_num_times=1)

        if len(dets) == 0:
            print(f"No faces detected in the image {file_path}")
            return None

        for i, d in enumerate(dets):
            # Save coordinates
            x1 = max(int(d.rect.left() / ratio), 1)
            y1 = max(int(d.rect.top() / ratio), 1)
            x2 = min(int(d.rect.right() / ratio), width - 1)
            y2 = min(int(d.rect.bottom() / ratio), height - 1)

            # Detect landmarks
            shape = face_utils.shape_to_np(predictor(gray, d.rect))
            points = []
            for index, (x, y) in enumerate(shape):
                x = int(round(x / ratio))
                y = int(round(y / ratio))
                if index in [2, 3, 5]:  # right eye, nose, left eye
                    points.append([x, y])
            points = np.array(points)

            # Rotate image if necessary
            if rotation:
                xLine = points[0][0] - points[2][0]
                yLine = abs(points[0][1] - points[2][1])
                angle = math.degrees(math.atan(yLine / xLine))
                rotated = imutils.rotate(orig, angle if points[2][1] < points[0][1] else 360 - angle)
            else:
                rotated = orig

            # Prepare image for prediction
            cropped_resized = cv2.resize(rotated[y1:y2, x1:x2], (img_width, img_height))
            pixel = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2GRAY)
            x = np.expand_dims(pixel, axis=0)
            x = x.reshape((-1, img_width, img_height, 1))

            # Predict emotion
            df = helper.predict_emotion(model, x)
            df = df.sort_values(by='prob', ascending=False)
            
            print("Prediction DataFrame:")
            print(df)
            
            emotion = df['emotion'].values[0]
            prob = str(round((df['prob'].values[0]) * 100, 2))

            result = {
                'file_path': file_path,
                'emotion': emotion,
                'probability': prob
            }

            # Overlay the result on the image
            text = f"{emotion}: {prob}%"
            cv2.rectangle(rotated, (0, 0), (width, 45), (0, 0, 0), -1)
            cv2.putText(rotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Save the resulting image
            result_image_path = os.path.splitext(file_path)[0] + '_result.jpg'
            cv2.imwrite(result_image_path, rotated)

            # Print result to console
            print(f"Processed image: {file_path}")
            print(f"Detected emotion: {emotion} with probability: {prob}%")

            # Display the result image
            display_image(result_image_path)

            return result

def display_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    args = parser.parse_args()

    result = process_image(args.image_path)
    if result:
        print("Analysis result saved and displayed above.")

if __name__ == '__main__':
    main()
