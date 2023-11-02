import cv2

# image file and video file
image_file = 'car_image.jpg'
video = cv2.VideoCapture('cars_and_pedestrian_video.mp4')

# pre-trained car classifier
car_classifier = 'car_detector.xml'

# car classifier
car_tracker = cv2.CascadeClassifier(car_classifier)


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

    # drawing rectangles around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)







    # display image
    cv2.imshow('Car and Pedestrian Detector', frame)


    # no autoclosing (waiting for a key press)
    cv2.waitKey(1)

'''
# opencv image
img = cv2.imread(image_file)

# convert to grayscale (required for haar cascade)
black_and_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# car classifier
car_tracker = cv2.CascadeClassifier(car_classifier)

# detect cars
cars = car_tracker.detectMultiScale(black_and_white)

# drawing rectangles around cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)




# display image
cv2.imshow('Car and Pedestrian Detector', video)


# no autoclosing (waiting for a key press)
cv2.waitKey()
'''

print('Have a Safe Ride!')