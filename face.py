import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained facial expression recognition model
model = load_model('model.hdf5')

# Define the emotions labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open video capture
video_capture = cv2.VideoCapture(0)  

while True:
    
    ret, frame = video_capture.read()
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    
    for (x, y, w, h) in faces:
        # Extract face region of interest (ROI)
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.astype("float") / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)
        
        # Perform emotion prediction
        emotion_prediction = model.predict(face_roi)[0]
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]
        
        # Draw bounding box and emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    
    cv2.imshow('Video', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()

cv2.destroyAllWindows()
