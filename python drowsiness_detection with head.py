import cv2
import dlib
import numpy as np
import pygame
import smtplib
import imghdr
from email.message import EmailMessage
from scipy.spatial import distance as dist

# Initialize Pygame for alert sound
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")  # Ensure alarm.mp3 is in the same folder

# Email Credentials (Replace with your details)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "snjmonisha@gmail.com"
EMAIL_PASS = "fpya znue hgob gqkd"  # Use generated App Password
TO_EMAIL = "gowtham4133@gmail.com"

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is in the script directory

# Indices for landmarks
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))
MOUTH = list(range(48, 68))  # Mouth landmarks for yawn detection

# EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) Thresholds
EAR_THRESHOLD = 0.25  # Below this, eyes are considered closed
MAR_THRESHOLD = 0.6  # Above this, mouth is considered open (yawning)
FRAME_THRESHOLD = 20  # Frames before alert triggers

frame_count = 0  # Counter for drowsy frames
email_sent = False  # Track if an email was sent

# Head Pose Detection (3D Model Points for Nose, Chin, etc.)
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
], dtype=np.float64)

# Camera matrix (Assuming focal length is width of the frame)
def get_camera_matrix(size):
    focal_length = size[1]
    return np.array([[focal_length, 0, size[1] / 2],
                     [0, focal_length, size[0] / 2],
                     [0, 0, 1]], dtype="double")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate Mouth Aspect Ratio (MAR) for Yawn Detection
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])  # Vertical distance
    B = dist.euclidean(mouth[14], mouth[18])  # Vertical distance
    C = dist.euclidean(mouth[15], mouth[17])  # Vertical distance
    D = dist.euclidean(mouth[12], mouth[16])  # Horizontal distance
    return (A + B + C) / (3.0 * D)

# Function to detect Head Pose
def detect_head_pose(shape, size):
    image_points = np.array([
        shape[30],  # Nose tip
        shape[8],   # Chin
        shape[36],  # Left eye left corner
        shape[45],  # Right eye right corner
        shape[48],  # Left mouth corner
        shape[54]   # Right mouth corner
    ], dtype="double")

    camera_matrix = get_camera_matrix(size)
    dist_coeffs = np.zeros((4, 1))  # No lens distortion

    _, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    angles, _ = cv2.Rodrigues(rotation_vector)
    return angles[1][0]  # Yaw angle

# Function to send an email alert with an image
def send_email_alert(image_path):
    msg = EmailMessage()
    msg["Subject"] = "🚨 Drowsiness Alert!"
    msg["From"] = EMAIL_USER
    msg["To"] = TO_EMAIL
    msg.set_content("The driver is showing signs of drowsiness. See attached image.")

    # Attach Image
    with open(image_path, "rb") as f:
        img_data = f.read()
        msg.add_attachment(img_data, maintype="image", subtype=imghdr.what(None, img_data))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        print("📧 Email Sent with Image!")
    except Exception as e:
        print(f"Error sending email: {e}")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    size = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        left_eye = shape[LEFT_EYE]
        right_eye = shape[RIGHT_EYE]
        mouth = shape[MOUTH]

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        MAR = mouth_aspect_ratio(mouth)
        head_tilt = detect_head_pose(shape, size)  # Detect head pose

        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)
        cv2.polylines(frame, [mouth], True, (255, 0, 0), 2)

        # Drowsiness or Yawning Detection
        if avg_EAR < EAR_THRESHOLD or MAR > MAR_THRESHOLD or abs(head_tilt) > 0.3:
            frame_count += 1
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            if frame_count >= FRAME_THRESHOLD:
                pygame.mixer.music.play()

                # Save image and send email alert
                if not email_sent:
                    img_path = "drowsy_capture.jpg"
                    cv2.imwrite(img_path, frame)
                    send_email_alert(img_path)
                    email_sent = True
        else:
            frame_count = 0
            email_sent = False

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
