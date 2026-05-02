import cv2
import os
import csv
from datetime import datetime

# Face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dataset path
dataset_path = r"C:\Users\srija\OneDrive\Desktop\SmartAttendance\dataset"

known_images = []
known_names = []

# Load dataset images
for file in os.listdir(dataset_path):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
        img = cv2.imread(os.path.join(dataset_path, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (300,300))
        known_images.append(gray)
        known_names.append(file.split(".")[0])

# Create attendance.csv if not exists
if not os.path.exists('attendance.csv'):
    with open('attendance.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name','Date','Time'])

# Function to mark attendance
def mark_attendance(name):
    with open('attendance.csv','r+') as f:
        data = f.readlines()
        names_list = [line.split(',')[0] for line in data]
        if name not in names_list:
            now = datetime.now()
            dt_string = now.strftime('%Y-%m-%d,%H:%M:%S')
            f.write(f'{name},{dt_string}\n')

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face = cv2.resize(face, (300,300))

        name = "Unknown"

        # Compare with dataset
        for i, known_face in enumerate(known_images):
            res = cv2.matchTemplate(face, known_face, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, _ = cv2.minMaxLoc(res)
            if confidence > 0.4:  # Updated threshold
                name = known_names[i]
                mark_attendance(name)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()