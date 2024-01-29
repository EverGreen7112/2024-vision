import cv2 as cv
import numpy as np
import math 

#note values
NOTE_RADIUS = 0.1778 # in meters

#camera values
CAMERA_PORT = 0
HORIZONTAL_FOV = 61.37
FOCAL_LENGTH = (2 * 60.59) / NOTE_RADIUS # distance * pixel radius / obj radius
DISTANCE_OFFSET = -0.15

#threshold values
MIN_THRESHOLD = (0, 150, 150) #hsv min threshold
MAX_THRESHOLD = (50, 300, 300) #hsv max threshold

#pipeline values
ERODE_ITERATIONS = 1
DILATE_ITERATIONS = 1
KERNEL_SIZE = 5

#denoising pipeline
def pipeline(binary_frame, erode_iterations, dilate_iterations):
    # apply erosion on frame
    binary_frame = cv.erode(binary_frame, np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8), erode_iterations)
    # apply dilation on frame
    binary_frame = cv.dilate(binary_frame, np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8), dilate_iterations)
    # apply distance transform 
    transformed_frame = cv.distanceTransform(binary_frame, cv.DIST_L2, KERNEL_SIZE)
    return transformed_frame 

def main():
     # create video capture 
    vid = cv.VideoCapture(CAMERA_PORT) 
    ret, frame = vid.read()
    #check if camera is working
    if ret == False or not vid.isOpened():
        print('no cap')
        exit()

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
        contours, hir = cv.findContours(final_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        #get bouding circle
        if len(contours) != 0:
          #find contour with largest area
          area = 0
          cnt = contours[0]
          for cnt in contours:
              if area < cv.contourArea(cnt):
                  c = cnt
                  area = cv.contourArea(cnt)

          #get center point and radius of the min bounding circle  
          center, radius = cv.minEnclosingCircle(c)
          
          #mark circle in the frame
          frame = cv.circle(frame, (int(center[0]), int(center[1])), int(radius), (255,0,0))
          
          #calculate distance 
          distance = (NOTE_RADIUS * FOCAL_LENGTH / radius) + DISTANCE_OFFSET
  
          #get resolutin of frame
          resX = frame.shape[0]     

          #calculate yaw angle from note
          yaw = (HORIZONTAL_FOV / resX) * (center[0] - resX / 2)       

          #get cartesian values of vector   
          x = math.cos(math.radians(yaw)) * distance
          y = math.sin(math.radians(yaw)) * distance

          print ('dis:', distance, 'yaw:', yaw)
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



