import cv2 as cv
import numpy as np

#note values
NOTE_RADIUS = 0.1778 # in meter

#camera values
CAMERA_PORT = 0
FOCAL_LENGTH = 4 / (31.335521697998047 * NOTE_RADIUS) # distance / (pixel radius * radius)
#threshold values
MIN_THRESHOLD = (0, 100, 200) #hsv min threshold
MAX_THRESHOLD = (50, 260, 350) #hsv max threshold

MIN_H = 0  #0
MAX_H = 50 #170
MIN_S = 100 #140
MAX_S = 260 #220
MIN_V = 200 #200
MAX_V = 350 #300

#pipeline values
ERODE_ITERATIONS = 0
DILATE_ITERATIONS = 00

#denoising pipeline
def pipeline(binary_frame, erode_iterations, dilate_iterations):
    # apply erosion on frame
    binary_frame = cv.erode(binary_frame, np.ones((5, 5), np.uint8), erode_iterations)
    # apply dilation on frame
    binary_frame = cv.dilate(binary_frame, np.ones((5, 5), np.uint8), dilate_iterations)
    # apply distance transform 
    transformed_frame = cv.distanceTransform(binary_frame, cv.DIST_L2, 5)
    return transformed_frame 

def main():
     # create video capture 
    vid = cv.VideoCapture(CAMERA_PORT) 
    ret, frame = vid.read()

    while(ret):
        #save camera frame
        ret, frame = vid.read()
        #convert frame from rgb to hsv
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV_FULL)
        #detect pixels in hsv range
        binary_frame = cv.inRange(hsv_frame, MIN_THRESHOLD, MAX_THRESHOLD)
        #apply pipeline denoising functions on frame
        transformed_frame = pipeline(binary_frame, ERODE_ITERATIONS, DILATE_ITERATIONS)
        #apply threshold on frame 
        final_frame = cv.inRange(transformed_frame, (0.5), (1)) 
        #detect contours
        cnts, hir = cv.findContours(final_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #get bouding circle
        if len(cnts) != 0:
          #find contour with largest area
          area = 0
          c = cnts[0]
          for cnt in cnts:
              if area < cv.contourArea(cnt):
                  c = cnt
                  area = cv.contourArea(cnt)
          #get center point and radius of the min bounding circle  
          center, radius = cv.minEnclosingCircle(cnt)
          print(radius)
          frame = cv.circle(frame, (int(center[0]), int(center[1])), int(radius), (255,0,0))
        #Display frame
        cv.imshow('detected pixels', final_frame) 
        cv.imshow('transformed', transformed_frame)
        cv.imshow('frame', frame)
        #close camera when 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'): 
            break

    # release video capture object 
    vid.release() 
    # kill open windows 
    cv.destroyAllWindows() 

if __name__ == '__main__':
   main()



