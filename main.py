import pathlib
import cv2
import numpy as np
import time
import torch
import torchvision
from torchvision import transforms
from collections import defaultdict

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))
camera = cv2.VideoCapture(0)
loaded_model = torch.load('models/Model.pth')
loaded_model.to("cuda")
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotionCount = defaultdict(int)

def prepare_face(face, size):
    face = cv2.resize(face, (size, size))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    face = transforms.ToTensor()(face).unsqueeze(0).to("cuda") 
    return face

while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.25,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        if len(faces) > 0:
            for(xframe, yframe, wframe, hframe) in faces:
                #Capture and prepare the face
                face = frame[yframe: yframe + hframe, xframe: xframe + wframe]
                face = prepare_face(face, 180)
                #Evaluate the emotion from the face
                with torch.inference_mode():  
                    outputs = loaded_model(face)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)

                # Display the predicted emotion on the face in the video stream
                emotion = emotions[predicted_class]
                emotionCount[emotion] = emotionCount.get(emotion, 0) + 1
                confidence = confidence.cpu().numpy()[0] * 100
                cv2.putText(frame, f"{emotion}" , (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
                cv2.putText(frame, f"confidence: {confidence:.2f}%",(x, y+175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord('q'):
        break 

emotionSum = 0
for i in emotionCount.values():
    emotionSum += i
['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
angry_percent = emotionCount.get('angry', 0) / emotionSum * 100
disgust_percent = emotionCount.get('disgust', 0) / emotionSum * 100
fear_percent = emotionCount.get('fear', 0) / emotionSum * 100
happy_percent = emotionCount.get('happy', 0) / emotionSum * 100
neutral_percent = emotionCount.get('neutral', 0) / emotionSum * 100
sad_percent = emotionCount.get('sad', 0) / emotionSum * 100
surprise_percent = emotionCount.get('surprise', 0) / emotionSum * 100
print(f"{angry_percent:.2f}% angry | {disgust_percent:.2f}% disgust| {fear_percent:.2f}% fear| {happy_percent:.2f}% happy| {neutral_percent:.2f}% neutral | {sad_percent:.2f}% sad| {surprise_percent:.2f}% surprise")
camera.release()
cv2.destroyAllWindows()
