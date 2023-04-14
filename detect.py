
import cv2
import mediapipe as mp
import numpy as np
import math


class fitNaysh():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
    
    def start_cap(self):
        self.cap = cv2.VideoCapture(0)
        self.counter = 0
        self.stage = None

    def calculate_angle(self,a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle

        return angle
    
    def isCamOpen(self):
        if self.cap.isOpened():
            return True
        else:
            return False
        
    def exit_cap(self):
        self.cap.release()
        cv2.destroyAllWindows()
    
    def UiElements(self,image,results):
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, self.stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )
        cv2.imshow('Counter', image)
    
    def measure_dist(self,a,b):
        x1, y1, z1 = a.x, a.y, a.z
        x2, y2, z2 = b.x, b.y, b.z
        
        distance=math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
        
        return distance

    def bicep_curl_counter(self):
        self.start_cap()
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                results = pose.process(image)
            
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    curl_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                    
                    cv2.putText(image, str(curl_angle), 
                                tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    if curl_angle > 160:
                        self.stage = "down"
                    if curl_angle < 60 and self.stage =='down':
                        self.stage="up"
                        self.counter +=1
                        print(self.counter)
                            
                except:
                    pass
                
                self.UiElements(image=image,results=results)               

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            self.exit_cap()
    
    def squat_counter(self):
        max_backl=0
        self.start_cap()
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                results = pose.process(image)
            
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_foot_index = [landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                    left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    back_l = self.measure_dist(left_shoulder, left_hip)

                    curl_angle = self.calculate_angle(left_hip, left_knee, left_ankle)                    

                    heel_angle = self.calculate_angle(left_knee, left_ankle, left_foot_index)

                    cv2.putText(image, str(curl_angle), 
                                tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    cv2.putText(image, str(back_l), 
                                tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    cv2.putText(image, str(heel_angle), 
                                tuple(np.multiply(left_ankle, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                    if back_l > max_backl:
                        max_backl = back_l
                    elif back_l < 0.8*max_backl:
                        print("Straighten your back")
                    
                    if heel_angle < 75:
                        print("Do not squat so LOW")

                    if curl_angle > 140:
                        self.stage = "up"
                    if curl_angle < 90 and self.stage =='up':
                        self.stage="down"
                        self.counter +=1
                        print(self.counter)
                    if self.stage=="down" and heel_angle>90:
                        print("Make sure your heels are toughing the ground")
                            
                except:
                    pass
                
                self.UiElements(image=image,results=results)               

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            self.exit_cap()
    
    def deadlift_counter(self):
        self.start_cap()
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                results = pose.process(image)
            
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    curl_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
                    
                    cv2.putText(image, str(curl_angle), 
                                tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    if curl_angle > 150:
                        self.stage = "up"
                    if curl_angle < 60 and self.stage =='up':
                        self.stage="down"
                        self.counter +=1
                        print(self.counter)
                            
                except:
                    pass
                
                self.UiElements(image=image,results=results)               

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            self.exit_cap()
