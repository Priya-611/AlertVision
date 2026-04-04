import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import winsound
import threading

# using threading --> This runs the alarm in background
# Without threading:
#-> Program would freeze while playing sound ❌
#-> Webcam would stop
# With threading:
#-> Alarm runs separately ✅
#-> Webcam keeps running smoothly



model = tf.keras.models.load_model(r"C:\Users\HP\OneDrive\Documents\Etc\OpenCV\Drowsiness Detection\model.h5")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")




# Eye cascade is not used because it becomes unreliable when eyes are closed or partially closed.
# Instead, we detect the face and extract the eye region using proportional slicing.
# This ensures consistent input to the model and improves stability of detection.



sleep_start = None
sleep_threshold = 5 # waiting 5 sec


frequency = 2500   #2500 Hz (sound pitch)
duration = 2000  #2000ms  (2 sec)  


alarm_on = False   #initailly alarm is off

def play_beep():
    global alarm_on    #allow modifying global variable inside function
    while alarm_on:    #keep playing sound till alarm_on is True
        winsound.Beep(frequency, duration)



 
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #convert to grayscale
    faces = faceCascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))   #detect face

    for (fx,fy,fw,fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255,0,0), 2)  #draw rect around face

        roi = frame[fy:fy+fh , fx:fx+fw]

        # Slightly relaxed parameters to improve detection when eyes are partially/fully closed.
        eye_region = roi[int(fh*0.25):int(fh*0.5), int(fw*0.15) : int(fw*0.85)]  #extract eye region 

        try:
            eye_resized=cv2.resize(eye_region, (224,224))   # resize to match model input size
        except:     
            continue    # skip frame if error occur
 
        x_input = np.array(eye_resized, dtype="float32") / 255.0     # convert  to array  and normalise from 0 to 1
        x_input = x_input.reshape(1,224,224,3)                   # add dimension required for model


        # Model predict
        prediction = model.predict(x_input, verbose=0)  #  verbose = 0 No progress bars, no logs printed in console


        # Make prediction
        if prediction[0][0] > 0.4:
            state ="Sleepy"

        else:
            state="Awake"


        # Get current time (in seconds)
        current_time = time.time()

        if state =="Sleepy":    # if state is Sleepy
            if sleep_start is None:    #if sleep_start is 0  i.e. Sleepy for the very first time
                sleep_start = current_time     # make sleep_start as the current time

            elapsed = current_time - sleep_start     # calculate duration
            
            if elapsed > sleep_threshold:     #if the duration is > the threshold
                result ="SLEEP ALERT"     # ALERT message of red color
                color =(0,0,255)
                
                if not alarm_on:    #if alarm if False , start alarm in background thread
                    alarm_on = True
                    threading.Thread(target=play_beep).start()    #call play_beep function

            else:   #if duration < threshold give warning as "Sleepy" in seconds 
                result = f"Sleepy ({int(elapsed)}s)"
                color =(0,165, 255)


        else:    #else result will be Awake in green color
            sleep_start = None
            alarm_on = False     # reset to stop alarm
            result = "Awake"
            color =(0,255,0)
            
          
            
            
        # show text for the state of person 
        
        cv2.putText(frame, result, (fx, fy-10), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
   
     
    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




