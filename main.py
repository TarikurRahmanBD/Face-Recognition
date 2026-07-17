# ====================================================
#
# Project: face-recognition-python-code
#
# Developer: Tarikur Rahman
#
# GitHub: [https://github.com/tarikurrahmanbd](https://github.com/tarikurrahmanbd)
#
# ====================================================

import cv2
import numpy as np
import os

# OpenCV LBPH-based face recognizer (no dlib required)
TRAIN_DIR = os.path.join('.', 'train')
TEST_IMAGE = os.path.join('.', 'test', 'test.jpg')
OUTPUT_IMAGE = os.path.join('.', 'output.jpg')

def prepare_training_data(train_dir):
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    images = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    labels = []
    faces = []
    label_map = {}
    next_label = 0

    for fname in images:
        name = os.path.splitext(fname)[0]
        label_name = name.capitalize()
        if label_name not in label_map:
            label_map[label_name] = next_label
            next_label += 1

        img_path = os.path.join(train_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_rects = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        # Use first detected face or whole image if none
        if len(faces_rects) > 0:
            (x, y, w, h) = faces_rects[0]
            face_roi = gray[y:y+h, x:x+w]
        else:
            face_roi = cv2.resize(gray, (200, 200))

        face_roi = cv2.resize(face_roi, (200, 200))
        faces.append(face_roi)
        labels.append(label_map[label_name])

    return faces, labels, {v:k for k,v in label_map.items()}

def main():
    if not os.path.isdir(TRAIN_DIR):
        print('Train directory not found:', TRAIN_DIR)
        return

    faces, labels, inv_label_map = prepare_training_data(TRAIN_DIR)
    if len(faces) == 0:
        print('No training faces found in', TRAIN_DIR)
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    # Load test image
    if not os.path.isfile(TEST_IMAGE):
        print('Test image not found:', TEST_IMAGE)
        return

    img = cv2.imread(TEST_IMAGE)
    if img is None:
        print('Failed to load test image')
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_rects = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces_rects:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (200, 200))
        label, confidence = recognizer.predict(face_resized)

        # LBPH: lower confidence == better match. Threshold may need tuning.
        name = 'Unknown'
        if confidence < 80:
            name = inv_label_map.get(label, 'Unknown')

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(img, (x, y+h-20), (x+w, y+h), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, f"{name}", (x+6, y+h-6), font, 0.6, (255, 255, 255), 1)

    cv2.imwrite(OUTPUT_IMAGE, img)
    print('Wrote', OUTPUT_IMAGE)

    # Display result if a GUI is available
    try:
        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == '__main__':
    main()
