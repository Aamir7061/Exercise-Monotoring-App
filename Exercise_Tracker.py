import cv2
from pynput import keyboard
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
exercise_type = None
# Calculating Angle
def calculate_angle(a,b,c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def on_press(key):
    global exercise_type
    try:
        global count,status
        count = 0
        status = None
        if key.char == '1':
            exercise_type = 'Pull Up'
        if key.char == '2':
            exercise_type = 'Push Up'
        if key.char == '3':
            exercise_type = 'Sit Up'
        if key.char == '4':
            exercise_type = 'Squat'
        if key.char == '5':
            exercise_type = 'Biceps'
        
    except AttributeError:
        pass  # Ignore non-character keys

# Start the key listener
listener = keyboard.Listener(on_press=on_press)
listener.start()
# Open the video capture 
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence = 0.65, min_tracking_confidence = 0.75) as pose:
    while cap.isOpened():
    # Read a frame from the video capture
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
                    
        results = pose.process(image)

        # Recoloring to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        global landmarks
                
        cv2.putText(image, 'Press 1 to start PULL UPS Monitoring',(20,380),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Press 2 to start PUSH UPS Monitoring',(20,400),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Press 3 to start SIT UPS Monitoring',(20,420),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Press 4 to start SQUAT Monitoring',(20,440),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Press 5 to start BICEPS Monitoring',(20,460),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow('Exercise Monitoring', image)
        
        # if pull-up exercise is selected
        if exercise_type == 'Pull Up':
            try:
                landmarks = results.pose_landmarks.landmark
                
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                left_elbow_visible = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > 0.75
                right_elbow_visible = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > 0.75
                angle = calculate_angle(shoulder, elbow, wrist)

                cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 155, 55), 2, cv2.LINE_AA)
                
                if left_elbow_visible or right_elbow_visible:
                    if angle < 160 and status==None:
                        status = 'True'
                    if status == 'True' and angle < 60:
                        status = 'False'
                    if status =='False' and angle>120:
                        status = 'True'
                        count = count + 1
                        
                cv2.putText(image, 'Excercise : Pull Up', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                cv2.putText(image, 'Counter :' + str(count), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                cv2.putText(image, 'Status :' + str(status), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                
            except Exception as e:
                print(f"An error occurred: {e}")
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 120, 66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245, 76, 130), thickness=2, circle_radius=2))
            #Displaying image
            cv2.imshow('Exercise Monitoring', image)
        # if push-up exercise is selected
        if exercise_type == 'Push Up':
            try:
                landmarks = results.pose_landmarks.landmark
                
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                left_elbow_visible = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > 0.75
                right_elbow_visible = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > 0.75
                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 155, 55), 2, cv2.LINE_AA)

                if left_elbow_visible or right_elbow_visible:   
                    if angle < 160 and status==None:
                        status = 'True'
                    if status == 'True' and angle < 70:
                        status = 'False'
                    if status =='False' and angle>90:
                        status = 'True'
                        count = count + 1
                    
                cv2.putText(image, 'Excercise : Push Up', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                cv2.putText(image, 'Counter :' + str(count), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                cv2.putText(image, 'Status :' + str(status), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                
            except :
                pass
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 120, 66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245, 76, 130), thickness=2, circle_radius=2))
            #Displaying image
            cv2.imshow('Exercise Monitoring', image)
        
        # if sit-up exercise is selected
        if exercise_type == 'Sit Up':
            try:
                landmarks = results.pose_landmarks.landmark
                
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_knee_visible = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.75
                right_knee_visible = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.75
                
                angle = calculate_angle(shoulder,hip, knee)
                
                cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 155, 55), 2, cv2.LINE_AA)
                if left_elbow_visible or right_elbow_visible:
                    if angle < 105 and status==None and len(landmarks)>10:
                        status = 'True'
                    if status == 'True' and angle < 55:
                        status = 'False'
                    if status =='False' and angle>70 and len(landmarks)>10:
                        status = 'True'
                        count = count + 1
                    
                cv2.putText(image, 'Excercise : Sit Up', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                cv2.putText(image, 'Counter :' + str(count), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                cv2.putText(image, 'Status :' + str(status), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                
            except Exception as e:
                print(f"An error occurred: {e}")
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 120, 66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245, 76, 130), thickness=2, circle_radius=2))
            #Displaying image
            cv2.imshow('Exercise Monitoring', image)

        # if squat exercise is selected
        if exercise_type == 'Squat':
            try:
                landmarks = results.pose_landmarks.landmark
                
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                left_knee_visible = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.75
                right_knee_visible = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.75
                
                angle = calculate_angle(hip, knee, ankle)
                
                cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 155, 55), 2, cv2.LINE_AA)
                if left_elbow_visible or right_elbow_visible:
                    if angle < 160 and status==None and len(landmarks)>10:
                        status = 'True'
                    if status == 'True' and angle < 70:
                        status = 'False'
                    if status =='False' and angle>90 and len(landmarks)>10:
                        status = 'True'
                        count = count + 1
                    
                cv2.putText(image, 'Excercise : Squat', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                cv2.putText(image, 'Counter :' + str(count), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                cv2.putText(image, 'Status :' + str(status), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                
            except Exception as e:
                print(f"An error occurred: {e}")
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 120, 66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245, 76, 130), thickness=2, circle_radius=2))
            #Displaying image
            cv2.imshow('Exercise Monitoring', image)

        # if Biceps exercise is selected
        if exercise_type == 'Biceps':
            try:
                landmarks = results.pose_landmarks.landmark
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                left_elbow_visible = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > 0.75
                right_elbow_visible = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > 0.75
            
                angle = calculate_angle(shoulder, elbow, wrist)
                
                cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 155, 55), 2, cv2.LINE_AA)
                if left_elbow_visible or right_elbow_visible:
                    if angle < 160 and status==None and len(landmarks)>16:
                        status = 'True'
                    if status == 'True' and angle < 30:
                        status = 'False'
                    if status =='False' and angle>90 and len(landmarks)>16:
                        status = 'True'
                        count = count + 1
                    
                cv2.putText(image, 'Excercise : Biceps', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                cv2.putText(image, 'Counter :' + str(count), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                cv2.putText(image, 'Status :' + str(status), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 240, 94), 2, cv2.LINE_AA)
                
            except Exception as e:
                print(f"An error occurred: {e}")
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 120, 66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245, 76, 130), thickness=2, circle_radius=2))
            #Displaying image
            cv2.imshow('Exercise Monitoring', image)
    
        # Check for 'q' key to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
