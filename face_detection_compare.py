import face_recognition
import numpy as np
import cv2


# Confidence function
def face_confidence(face_distance, threshold=0.6):
    if face_distance > threshold:
        return round((1 - face_distance) * 100, 2)
    else:
        return round((1.0 - face_distance / threshold) * 100, 2)


# Array of images
images_to_detect = [
    ("Joy", ["joy.jpg", "joy 4.jpg", "joy 3.jpg", "joy_check.jpg"]),
    ("Veniz", ["me.jpg", "me 2.jpg", "me 3.jpg", "me 4.jpg"]),
]

encodings = []
names = []

# Encode faces
for name, img_paths in images_to_detect:
    for img_path in img_paths:
        image = face_recognition.load_image_file(img_path)
        encodings_in_image = face_recognition.face_encodings(image)
        if encodings_in_image:
            encodings.append(encodings_in_image[0]) 
            names.append(name)
        else:
            print(f"Warning: No face found in {img_path}")

# Image to compare
unknown_image = face_recognition.load_image_file("me 1.jpg")
unknown_face_locations = face_recognition.face_locations(unknown_image)
unknown_face_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)

# Convert to process in OpenCV
img_to_show = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

# Process each face
for (top, right, bottom, left), face_encoding in zip(unknown_face_locations, unknown_face_encodings):
    # Calculate distances between faces
    distances = face_recognition.face_distance(encodings, face_encoding)
    
    # find best match
    best_match_index = np.argmin(distances)
    best_distance = distances[best_match_index]
    confidence = face_confidence(best_distance)

    # Compare face encoding with best match
    match = face_recognition.compare_faces([encodings[best_match_index]], face_encoding)[0]
    name = names[best_match_index] if match else "Unknown"

   
    cv2.rectangle(img_to_show, (left, top), (right, bottom), (0, 255, 0), 1)
    label = f"{name} ({confidence}%)"
    cv2.rectangle(img_to_show, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
    cv2.putText(img_to_show, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

cv2.imshow("Face Comparison", img_to_show)

if cv2.waitKey(0):
    cv2.destroyAllWindows()
