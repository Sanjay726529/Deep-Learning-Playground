# imports
import cv2
from pathlib import Path


# face classifier haar feature
cascade_file = str(Path(cv2.__file__).parent / 'data' / 'haarcascade_frontalface_default.xml')
faceclassifier = cv2.CascadeClassifier(cascade_file)


# start video capture and detect faces
video_capture = cv2.VideoCapture(0)


while True:
    # capture frames from video feed
    frames = video_capture.read()
    
    # convert the captured data into grayscale color scheme
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) # rgb_to_gray
    
    # detect faces in the gray images
    faces = faceclassifier.detectMultiScale(
                                        gray, # image to be used
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    
    
    # bound the detected faces
    for (x, y, width, height) in faces:
        cv2.rectangle(frames, (x, y), (x+width, y+height), (0, 255, 0), 2)
        
    # display detected frame
    cv2.imshow('Video', frames)

    # quit by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
