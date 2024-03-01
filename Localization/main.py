import math
import os
import socket
import struct
import threading
import time
import cv2
import numpy as np

import settings
import show_on_field
import tag
from april_tag_utils import *

LIFE_CAM_WIDTH = 800  # 1280  # 960  # 800
LIFE_CAM_HEIGHT = 448  # 720  # 544  # 448
cam = cv2.VideoCapture(0, cv2.CAP_ANY)
BRIO_HEIGHT = 1080
BRIO_WIDTH = 1920

# NOTE: ori and itay, these are not magic numbers, the weird number in the tan function is simply the cameras FOV
# divided by 2 in radians
F_LENGTH_X_BRIO = (1 / (math.tan(0.3841161543769188) * 2)) * BRIO_WIDTH
F_LENGTH_Y_BRIO = (1 / (math.tan(0.24892437646661184) * 2)) * BRIO_HEIGHT
F_LENGTH_X_LIFECAM = (1 / (math.tan(0.5355780593748425) * 2)) * LIFE_CAM_WIDTH
F_LENGTH_Y_LIFECAM = (1 / (math.tan(0.3221767906849529) * 2)) * LIFE_CAM_HEIGHT
lifecam_distortion_coefs = np.array([[1.01094557e-01, -8.10764739e-01, 3.23088490e-04, 4.97992890e-06, 1.48988740e+00]])
DATA_PORT = 5800  # the port where we send localization data
HEALTH_CHECK_PORT = 5801  # the port where we send a signal in regular intervals so the other end will know we are alive

new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
    np.array([[1.15929436e+03, 0, 6.43213888e+02],
              [0, 1.08801044e+03, 3.71547461e+02],
              [0, 0, 1]]), lifecam_distortion_coefs, (LIFE_CAM_WIDTH, LIFE_CAM_HEIGHT), 1,
    (LIFE_CAM_WIDTH, LIFE_CAM_HEIGHT))
# undistort
mapx, mapy = cv2.initUndistortRectifyMap(np.array([[1.15929436e+03, 0, 6.43213888e+02],
                                                   [0, 1.08801044e+03, 3.71547461e+02],
                                                   [0, 0, 1]]), lifecam_distortion_coefs, None, new_camera_mtx,
                                         (LIFE_CAM_WIDTH, LIFE_CAM_HEIGHT), 5)

# these values are here so we could check our momentary changes and see if they make any sense
last_pos_estimate = np.array([0.0, 0.0, 0.0])
last_rot_estimate = np.array([0.0, 0.0, 0.0])
last_time = 0  # the time of the last pos estimation, NOT necessarily the time of the last frame
last_is_accurate = False  # tells you if the last estimation is accurate

# these values are for refining the estimation
MAX_VEL = 4  # maximum velocity of the robot, if passed we can assume there was a problem with the pose estimation
MAX_ACCEL = 15000  # maximum acceleration of the robot, if passed we can assume there was a problem with the pose
MIN_CONFIDENCE = 0.14
SPEED_WEIGHT = 2  # how much weight we give speed in confidence estimation
ROT_WEIGHT = 1.1  # how much weight we give the rotation in confidence estimation
DISTANCE_FROM_AVG_WEIGHT = 6  # how much weight do we give to distance from the average in confidence estimation
QUANTIZATION_LEVELS = 64  # how many levels do we want to divide the image to


def denoise_frame(frame):
    processed_frame = copy.deepcopy(frame)
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.normalize(processed_frame, processed_frame, 0, 255, cv2.NORM_MINMAX)
    processed_frame = cv2.medianBlur(processed_frame, 3)
    processed_frame = cv2.GaussianBlur(processed_frame, (3, 3), sigmaX=0.1, sigmaY=0.1)
    # processed_frame = np.round(processed_frame * (QUANTIZATION_LEVELS / 255)) * (255 / QUANTIZATION_LEVELS)
    # processed_frame = np.uint8(np.round(processed_frame))
    # kernel = 1.35*np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])
    # processed_frame = cv2.filter2D(processed_frame, -1, kernel)

    return processed_frame


def draw_tag_axis(frame, camera_oriented_axis_mat, projected_points):
    projected_z = project_point(camera_oriented_axis_mat[:3, 2], F_LENGTH_X_LIFECAM,
                                F_LENGTH_Y_LIFECAM, LIFE_CAM_WIDTH, LIFE_CAM_HEIGHT)
    projected_y = project_point(camera_oriented_axis_mat[:3, 1], F_LENGTH_X_LIFECAM,
                                F_LENGTH_Y_LIFECAM, LIFE_CAM_WIDTH, LIFE_CAM_HEIGHT)
    projected_x = project_point(camera_oriented_axis_mat[:3, 0], F_LENGTH_X_LIFECAM,
                                F_LENGTH_Y_LIFECAM, LIFE_CAM_WIDTH, LIFE_CAM_HEIGHT)
    center = find_projected_tag_center(projected_points)
    cv2.line(frame, (int(projected_x[0]), int(projected_x[1])), (int(center[0]), int(center[1])),
             (255, 0, 0), 5)
    cv2.line(frame, (int(projected_y[0]), int(projected_y[1])), (int(center[0]), int(center[1])),
             (0, 255, 0), 5)
    cv2.line(frame, (int(projected_z[0]), int(projected_z[1])), (int(center[0]), int(center[1])),
             (0, 0, 255), 5)


def estimate_confidence(xyz, abs_distance, rotation, delta_time, tag_id):
    return 1 / (abs_distance +
                (SPEED_WEIGHT * (np.linalg.norm(last_pos_estimate - xyz) / delta_time)) +
                ((abs(settings.TAGS[tag_id].yaw + rotation[0]) % math.pi) * ROT_WEIGHT))


def estimate_confidence_by_avg(conf: float, count: int, avg: np.ndarray, xyz: np.ndarray):
    # NOTE: takes confidence to in case we'd want to expand the calculation
    return count / ((1 / conf) + (DISTANCE_FROM_AVG_WEIGHT * np.linalg.norm(avg - xyz)))


def submit_final_estimation(xyz: np.ndarray, rotation: list):
    show_on_field.xyz = [xyz[0], xyz[1], tag.FIELD_HEIGHT - xyz[2]]
    show_on_field.rotation = math.pi + rotation[0]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(struct.pack('ffff', xyz[0],
                                xyz[1],
                                tag.FIELD_HEIGHT - xyz[2],
                                math.degrees(rotation[0] + math.pi)),
                    ("255.255.255.255", DATA_PORT))


last_health_check = 0


def send_health_check():
    global last_health_check
    if (time.time() - last_health_check) >= settings.HELATH_CHECK_INTERVAL:
        last_health_check = time.time()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(b"/001",
                        ("255.255.255.255", HEALTH_CHECK_PORT))


def refine_estimation(pose_estimates, rot_estimates, estimation_confidences, delta_time):
    global last_is_accurate, last_rot_estimate, last_pos_estimate, last_time
    count = len(pose_estimates)
    # averaging out
    avg_xyz = last_pos_estimate * int(last_is_accurate)
    for p in pose_estimates:
        avg_xyz += p
    avg_xyz /= (count + int(last_is_accurate))

    for i in range(count):
        estimation_confidences[i] = estimate_confidence_by_avg(estimation_confidences[i], count, avg_xyz,
                                                               pose_estimates[i])

    conf = 0
    cam_xyz = last_pos_estimate
    rotation = last_rot_estimate
    for i in range(count):
        if estimation_confidences[i] > conf:
            cam_xyz = pose_estimates[i]
            rotation = rot_estimates[i]
            conf = estimation_confidences[i]

    # comparing to the last estimation
    delta_x = (cam_xyz - last_pos_estimate) * int(last_is_accurate)
    velocity = delta_x * (1 / delta_time)

    if (conf < MIN_CONFIDENCE) or (np.linalg.norm(velocity) > MAX_VEL):
        if last_is_accurate:
            cam_xyz = last_pos_estimate
            rotation = last_rot_estimate
        last_is_accurate = False
    else:
        last_is_accurate = True
    last_pos_estimate = cam_xyz
    last_rot_estimate = rotation

    return cam_xyz, rotation


def process_frame(frame):
    global last_is_accurate, last_time
    frame_width = LIFE_CAM_WIDTH
    frame_height = LIFE_CAM_HEIGHT
    focal_length_x = F_LENGTH_X_LIFECAM
    focal_length_y = F_LENGTH_Y_LIFECAM
    cur_time = time.time()
    send_health_check()
    processed_frame = denoise_frame(frame)

    delta_time = cur_time - last_time  # time between the last estimation and this one, NOT the time between 2 frames
    proj_squares, ids = detect_april_tags(processed_frame)
    draw(frame, proj_squares, ids)
    pose_estimates = []
    rot_estimates = []
    estimation_confidences = []
    for i in range(len(ids)):
        tag_id = ids[i]
        if tag_id in settings.TAGS_INVERSE.keys():  # only process tags we know
            projected_points = proj_squares[i]
            # return tag to origin
            field_oriented_inv_axis_matrix = settings.TAGS_INVERSE[tag_id]
            tag_transformation_matrix, abs_distance = tag_projected_points_to_transform(tag=projected_points,
                                                                                        width=frame_width,
                                                                                        height=frame_height,
                                                                                        tag_shape=tag.BASIS_TAG_COORDS_MATRIX,
                                                                                        focal_length_x=focal_length_x,
                                                                                        focal_length_y=focal_length_y)
            camera_oriented_axis_mat = tag_transformation_matrix @ tag.BASIS_AXIS_MATRIX
            extrinsic_matrix = camera_oriented_axis_mat @ field_oriented_inv_axis_matrix
            # robot extrinsic matrix is the transformation from the robots 0 point to camera oriented coordinates
            robot_extrinsic_matrix = extrinsic_matrix @ settings.CAMERA_TO_ROBOT_CENTER_TRANSFORMATION
            robot_xyz = extrinsic_matrix_to_camera_position(robot_extrinsic_matrix)
            rotation = extrinsic_matrix_to_rotation(extrinsic_matrix)

            # this part here does some epic pose estimation refinement
            pose_estimates.append(robot_xyz)
            rot_estimates.append(np.array(rotation))
            conf = estimate_confidence(robot_xyz, abs_distance, rotation, delta_time, tag_id)
            estimation_confidences.append(conf)

            # draw everything on the frame
            draw_tag_axis(frame, camera_oriented_axis_mat, projected_points)
            cv2.putText(frame, str(math.degrees(rotation[0] + math.pi)),
                        (int(projected_points[2][0]) - 20, int(projected_points[2][1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 3)

    if len(pose_estimates) > 0:
        last_time = cur_time
        robot_xyz, rotation = refine_estimation(pose_estimates, rot_estimates, estimation_confidences,
                                                delta_time)
        if last_is_accurate:
            submit_final_estimation(robot_xyz, rotation)
    else:
        robot_xyz = last_pos_estimate
        last_is_accurate = False
    return frame, robot_xyz


# def process_frame_headless(image):
#     global last_is_accurate, last_time
#     frame_width = LIFE_CAM_WIDTH
#     frame_height = LIFE_CAM_HEIGHT
#     focal_length_x = F_LENGTH_X_LIFECAM
#     focal_length_y = F_LENGTH_Y_LIFECAM
#     cur_time = time.time()
#     frame = image
#     processed_frame = denoise_frame(frame)
#
#     delta_time = cur_time - last_time  # time between the last estimation and this one, NOT the time between 2 frames
#     proj_squares, ids = detect_april_tags(processed_frame)
#     # draw(frame, proj_squares, ids)
#     pose_estimates = []
#     rot_estimates = []
#     estimation_confidences = []
#     for i in range(len(ids)):
#         tag_id = ids[i]
#         if tag_id in settings.TAGS_INVERSE.keys():  # only process tags we know
#             projected_points = proj_squares[i]
#             # return tag to origin
#             field_oriented_inv_axis_matrix = settings.TAGS_INVERSE[tag_id]
#             tag_transformation_matrix, abs_distance = tag_projected_points_to_transform(tag=projected_points,
#                                                                                         width=frame_width,
#                                                                                         height=frame_height,
#                                                                                         tag_shape=tag.BASIS_TAG_COORDS_MATRIX,
#                                                                                         focal_length_x=focal_length_x,
#                                                                                         focal_length_y=focal_length_y)
#             camera_oriented_axis_mat = tag_transformation_matrix @ tag.BASIS_AXIS_MATRIX
#             extrinsic_matrix = camera_oriented_axis_mat @ field_oriented_inv_axis_matrix
#             cam_xyz = extrinsic_matrix_to_camera_position(extrinsic_matrix)
#             rotation = extrinsic_matrix_to_rotation(extrinsic_matrix)
#
#             # this part here does some epic pose estimation refinement
#             pose_estimates.append(cam_xyz)
#             rot_estimates.append(np.array(rotation))
#             conf = estimate_confidence(cam_xyz, abs_distance, rotation, delta_time, tag_id)
#             estimation_confidences.append(conf)
#
#     if len(pose_estimates) > 0:
#         last_time = cur_time
#         cam_xyz, rotation = refine_estimation(pose_estimates, rot_estimates, estimation_confidences,
#                                               delta_time)
#         if last_is_accurate:
#             submit_final_estimation(cam_xyz, rotation)
#     else:
#         cam_xyz = last_pos_estimate
#         last_is_accurate = False
#     return frame, cam_xyz


def test_with_sample_images():  # sample images were also taken with lifecam
    folder = "2024VisionSampleImages"
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    show_on_field.thread.start()

    for img in images:
        # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        # # crop the image
        # x, y, w, h = roi
        # img = dst[y:y + h, x:x + w]
        while not ((cv2.waitKey(1) & 0xFF) == ord(" ")):
            # Generate random Gaussian noise
            mean = (0, 0, 0)
            stddev = (25, 25, 25)
            noise = np.zeros(img.shape, np.uint8)
            cv2.randn(noise, mean, stddev)

            # Add noise to image
            image = cv2.add(img, noise)
            image, _ = process_frame(image)
            cv2.imshow("display", image)
    cv2.destroyAllWindows()


def test_with_cam():
    global cam
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, LIFE_CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, LIFE_CAM_HEIGHT)
    # cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_EXPOSURE, settings.EXPOSURE)

    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)

    show_on_field.thread.start()  # NOTE: ori and itay, yes theres a different thread for showing where everything is
    # on the field, this is because in the actual competition its the driver station computer that's gonna do it and i
    # didn't feel like actually writing the full code that does updates the other thread with UDP,
    # you are welcome to cry about it

    while True:
        s = time.time()
        ok, frame = cam.read()
        if ok:
            # dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            # # crop the image
            # x, y, w, h = roi
            # frame = dst[y:y + h, x:x + w]
            frame, _ = process_frame(frame)
            cv2.imshow('Display', frame)
            cv2.waitKey(1)
            # print((time.time() - s))
        else:
            cv2.destroyWindow('Display')
            cam.release()
            cv2.waitKey(500)
            show_on_field.run_thread = False
            cam.open(0, cv2.CAP_ANY)
            show_on_field.thread = threading.Thread(target=show_on_field.show_on_field)
            raise Exception("camera failure")


def test_with_cam_headless():
    # cam.set(cv2.CAP_PROP_FPS, 25)
    cam.set(cv2.CAP_PROP_EXPOSURE, settings.EXPOSURE)
    # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    # _, frame = cam.read()
    # cam.set(cv2.CAP_PROP_FPS, 30.0)
    # _, frame = cam.read()
    # print("hi")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, LIFE_CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, LIFE_CAM_HEIGHT)
    # _, frame = cam.read()
    # cam.set(cv2.CAP_PROP_FPS, 30.0)

    while True:
        s = time.time()

        ok, frame = cam.read()
        if ok:
            # dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            # # crop the image
            # x, y, w, h = roi
            # frame = dst[y:y + h, x:x + w]
            frame, _ = process_frame(frame)
            cv2.waitKey(1)

            # print(f"fps: {1 / (time.time() - s)}  found: {last_is_accurate}")
            # print(cam.get(cv2.CAP_PROP_FPS))
        else:
            cam.release()
            cv2.waitKey(500)
            cam.open(0, cv2.CAP_ANY)
            raise Exception("camera failure")


if __name__ == '__main__':
    while True:
        try:
            # test_with_cam()
            test_with_cam_headless()
        except Exception as e:
            s = str(e)
            f = open("log.txt", "a+")
            if (os.stat("log.txt").st_size) > settings.MAX_LOG_SIZE:
                f.truncate(0)
            f.write(f"{s}\n")
            f.close()
    # test_with_sample_images()
