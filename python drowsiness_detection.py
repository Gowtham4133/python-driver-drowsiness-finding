import cv2
import dlib
import numpy as np
import pygame
import smtplib
from email.mime.text import MIMEText
from scipy.spatial import distance as dist

# Initialize Pygame for playing alert sound
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")  # Load an alert sound file (must be in the same directory)

# Email Credentials (Replace with your Gmail details)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "snjmonisha@gmail.com"  # Your Gmail address
EMAIL_PASS = "fpya znue hgob gqkd"  # Use the generated App Password
TO_EMAIL = "gowtham4133@gmail.com"  # Alert recipient

# Function to send an email alert
def send_email_alert():
    subject = "🚨 Drowsiness Alert!"
    body = "The driver is showing signs of drowsiness. Please check immediately!"
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = TO_EMAIL
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure the connection
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, TO_EMAIL, msg.as_string())
        server.quit()
        print("📧 Email Sent Successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download and place this file in the script directory

# Indices for eye landmarks
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# Drowsiness detection parameters
EAR_THRESHOLD = 0.25  # EAR value below this means the eyes are closed
FRAME_THRESHOLD = 20  # Number of consecutive frames with closed eyes to trigger an alert

frame_count = 0  # Counter for drowsy frames
email_sent = False  # Track if an email was sent

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = detector(gray)  # Detect faces

    for face in faces:
        shape = predictor(gray, face)  # Get facial landmarks
        shape = np.array([[p.x, p.y] for p in shape.parts()])  # Convert to NumPy array

        # Get eye landmarks
        left_eye = shape[LEFT_EYE]
        right_eye = shape[RIGHT_EYE]

        # Compute EAR for both eyes
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0  # Average EAR

        # Draw eye landmarks
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)

        # Drowsiness detection logic
        if avg_EAR < EAR_THRESHOLD:
            frame_count += 1  # Increase counter
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Play alarm sound if threshold is reached
            if frame_count >= FRAME_THRESHOLD:
                pygame.mixer.music.play()
                
                # Send email alert (only once per drowsiness detection)
                if not email_sent:
                    send_email_alert()
                    email_sent = True
        else:
            frame_count = 0  # Reset counter
            email_sent = False  # Reset email flag if eyes open

    # Display output
    cv2.imshow("Drowsiness Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
