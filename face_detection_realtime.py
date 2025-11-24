import face_recognition
import cv2
import numpy as np

def load_face(name, files):
    encodings = []
    for file in files:
        image = face_recognition.load_image_file(file)
        face_encodings = face_recognition.face_encodings(image)
        if not face_encodings:
            print(f"Warning: No face found in image: {file}")
            continue
        encodings.append(face_encodings[0]) 
    return encodings, name


known_faces = [
    ("Dona", ["dona.jpg"]),
    ("Jamal", ["jamal.jpg"]),
    ("Veniz", ["me.jpg", "me 1.jpg", "me 2.jpg", "me 3.jpg", "me 4.jpg","me 6.jpg","me 7.jpg","me 8.jpg","me 9.jpg",]),
]


known_face_encodings = []
known_face_names = []

for name, image_files in known_faces:
    try:
        encodings, name = load_face(name, image_files)
        known_face_encodings.extend(encodings)  
        known_face_names.extend([name] * len(encodings)) 
    except ValueError as e:
        print(e)

# Open webcam
video_capture = cv2.VideoCapture(0)

def calculate_confidence(face_distance, threshold=0.65):
    if face_distance > threshold:
        return round((1 - face_distance) * 100, 2)
    else:
        return round((1.0 - face_distance / threshold) * 100, 2)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

   
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_small_frame = np.ascontiguousarray(rgb_small_frame, dtype=np.uint8)

   
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    face_confidences = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings,  )
        name = "Unknown"
        confidence = 0.0

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            confidence = calculate_confidence(face_distances[best_match_index])

          #MATCH IF 50 CINFIDENCE
            if matches[best_match_index] and confidence >= 50:  
                name = known_face_names[best_match_index]
            else:
                name = "Unknown" 

        face_names.append(name)
        face_confidences.append(confidence)

    # Draw boxes and labels
    for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, face_confidences):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

       
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        label = f"{name} ({confidence}%)" if name != "Unknown" else name
        cv2.putText(frame, label, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
