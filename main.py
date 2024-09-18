import cv2
import mediapipe as mp

#read image
img_path = './data/testImg.png'
img = cv2.imread(img_path)

#detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    for detection in out.detections:
        location_data = detection.location_data
        bbox = location_data.relative_bounding_box

        x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

#blur faces

#save image