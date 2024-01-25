import cv2


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the HSV color directly
        hsv_color = tuple(frame_hsv[y, x])
        # Scale the HSV values to the standard ranges
        scaled_hsv_color = (
            int(hsv_color[0] * 2),  # Standard Hue is scaled to [0, 360]
            int(hsv_color[1] / 255 * 100),  # Standard Saturation is scaled to [0, 100]
            int(hsv_color[2] / 255 * 100)  # Standard Value is scaled to [0, 100]
        )
        # Normal people and applications use the scaled form of the HSV values
        # but cv2 uses the unscaled version
        print("HSV values:", hsv_color, "| Scaled HSV Values:", scaled_hsv_color)


# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No cap")
    exit()

# Set the mouse callback function
cv2.namedWindow("HSV Color Picker")
cv2.setMouseCallback('HSV Color Picker', mouse_callback)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Display the frame
    cv2.imshow('HSV Color Picker', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
