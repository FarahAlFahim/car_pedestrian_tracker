import cv2

# image file and video file
image_file = 'car_image.jpg'
video = cv2.VideoCapture('cars_and_pedestrian_video.mp4')

# pre-trained car and predestrian classifiers
car_classifier_file = 'car_detector.xml'
pedestrian_classifier_file = 'pedestrian_detector.xml'


# car and pedestrian classifiers
car_tracker = cv2.CascadeClassifier(car_classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier_file)


# loop through the whole video
while True:
    # reading the current frame
    (read_successful, frame) = video.read()

    # handling errors
    if read_successful:
        # convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # drawing rectangles around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # drawing rectangles around pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)







    # displaying the video with cars and pedestrians spotted
    cv2.imshow('Car and Pedestrian Detector', frame)


    # no autoclosing (waiting for a key press)
    cv2.waitKey(1)

    # stop if Q is pressed
    if cv2.waitKey(1) == ord('Q') or cv2.waitKey(1) == ord('q'):
        break

# releasing the VideoCapture object
video.release()



print('Have a Safe Ride!')