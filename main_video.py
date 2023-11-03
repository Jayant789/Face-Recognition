import cv2
from simple_facerec import SimpleFacerec
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime
import time

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)

# Initialize a variable to track the last time data was added
last_data_time = time.time()

while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        if name != "Unknown":
            cv2.putText(
                frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        else:
            cv2.putText(
                frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)
            current_time = time.time()
            if current_time - last_data_time >= 10:
                # Prepare the document data
                document_data = {
                    "Name": name,
                    "Date": datetime.now().date().strftime("%Y-%m-%d"),
                    "Time": datetime.now().time().strftime("%H:%M:%S"),
                }

                # Use a formatted timestamp as the document ID
                timestamp_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Add the document to the "suspects" collection using the timestamp as the ID
                db.collection("Suspects").document(timestamp_id).set(document_data)
                print(name)
                # Update the last_data_time
                last_data_time = current_time

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
