import math
import os
import socket
import struct
import time
import cv2
import numpy as np

import settings
import tag
from april_tag_utils import *

WIDTH = 1280
HEIGHT = 960

# NOTE: ori and itay, these are not magic numbers, the weird number in the tan function is simply the cameras FOV
# divided by 2 in radians
F_LENGTH_X = (1 / (math.tan(math.radians(63.3*0.5)) * 2)) * WIDTH
F_LENGTH_Y = (1 / (math.tan(math.radians(49.7*0.5)) * 2)) * HEIGHT
PORT = 5800


# these values are here so we could check our momentary changes and see if they make any sense
last_pos_estimate = np.array([0.0, 0.0, 0.0])
last_rot_estimate = np.array([0.0, 0.0, 0.0])
last_time = 0  # the time of the last pos estimation, NOT necessarily the time of the last frame
last_is_accurate = False  # tells you if the last estimation is accurate

# these values are for refining the estimation
MAX_VEL = 4  # maximum velocity of the robot, if passed we can assume there was a problem with the pose estimation
MAX_ACCEL = 15000  # maximum acceleration of the robot, if passed we can assume there was a problem with the pose
MIN_CONFIDENCE = 0.11
SPEED_WEIGHT = 2  # how much weight we give speed in confidence estimation
ROT_WEIGHT = 1.1  # how much weight we give the rotation in confidence estimation
DISTANCE_FROM_AVG_WEIGHT = 6  # how much weight do we give to distance from the average in confidence estimation
QUANTIZATION_LEVELS = 64  # how many levels do we want to divide the image to


def denoise_frame(frame):
    processed_frame = copy.deepcopy(frame)
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.normalize(processed_frame, processed_frame, 0, 255, cv2.NORM_MINMAX)
    processed_frame = cv2.medianBlur(processed_frame, 3)
    processed_frame = cv2.GaussianBlur(processed_frame, [3, 3], sigmaX=0.1, sigmaY=0.1)
    processed_frame = np.round(processed_frame * (QUANTIZATION_LEVELS / 255)) * (255 / QUANTIZATION_LEVELS)
    processed_frame = np.uint8(np.round(processed_frame))
    kernel = 1.35*np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    processed_frame = cv2.filter2D(processed_frame, -1, kernel)

    return processed_frame


def draw_tag_axis(frame, camera_oriented_axis_mat, projected_points):
    projected_z = project_point(camera_oriented_axis_mat[:3, 2], F_LENGTH_X,
                                F_LENGTH_Y, WIDTH, HEIGHT)
    projected_y = project_point(camera_oriented_axis_mat[:3, 1], F_LENGTH_X,
                                F_LENGTH_Y, WIDTH, HEIGHT)
    projected_x = project_point(camera_oriented_axis_mat[:3, 0], F_LENGTH_X,
                                F_LENGTH_Y, WIDTH, HEIGHT)
    center = find_projected_tag_center(projected_points)
    cv2.line(frame, (int(projected_x[0]), int(projected_x[1])), (int(center[0]), int(center[1])),
             (255, 0, 0), 5)
    cv2.line(frame, (int(projected_y[0]), int(projected_y[1])), (int(center[0]), int(center[1])),
             (0, 255, 0), 5)
    cv2.line(frame, (int(projected_z[0]), int(projected_z[1])), (int(center[0]), int(center[1])),
             (0, 0, 255), 5)


def estimate_confidence(xyz, abs_distance, rotation, delta_time, tag_id):
    return 1 / (abs_distance +
                (SPEED_WEIGHT * (np.linalg.norm(last_pos_estimate - xyz) / delta_time))
                + ((abs(settings.TAGS[tag_id].yaw + rotation[0]) % math.pi) * ROT_WEIGHT))


def estimate_confidence_by_avg(conf: float, count: int, avg: np.ndarray, xyz: np.ndarray):
    # NOTE: takes confidence to in case we'd want to expand the calculation
    return count / ((1 / conf) + (DISTANCE_FROM_AVG_WEIGHT * np.linalg.norm(avg - xyz)))


def submit_final_estimation(xyz: np.ndarray, rotation: list):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(struct.pack('ffff', xyz[0],
                                xyz[1],
                                tag.FIELD_HEIGHT - xyz[2],
                                math.degrees(rotation[0] + math.pi)),
                    ("255.255.255.255", PORT))


def refine_estimation(pose_estimates, rot_estimates, estimation_confidences, delta_time):
    global last_is_accurate, last_rot_estimate, last_pos_estimate, last_time
    count = len(pose_estimates)
    # averaging out
    avg_xyz = last_pos_estimate * int(last_is_accurate)
    for p in pose_estimates:
        avg_xyz += p
    avg_xyz /= (count+int(last_is_accurate))

    for i in range(count):
        estimation_confidences[i] = estimate_confidence_by_avg(estimation_confidences[i], count, avg_xyz, pose_estimates[i])

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

    # print(np.linalg.norm(velocity))
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


def runPipeline(image, llrobot):  # this function is in a format for putting it on the limelight
    global last_is_accurate, last_time
    cur_time = time.time()
    frame = image
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
                                                                                        width=WIDTH,
                                                                                        height=HEIGHT,
                                                                                        tag_shape=tag.BASIS_TAG_COORDS_MATRIX,
                                                                                        focal_length_x=F_LENGTH_X,
                                                                                        focal_length_y=F_LENGTH_Y)
            camera_oriented_axis_mat = tag_transformation_matrix @ tag.BASIS_AXIS_MATRIX
            extrinsic_matrix = camera_oriented_axis_mat @ field_oriented_inv_axis_matrix
            cam_xyz = extrinsic_matrix_to_camera_position(extrinsic_matrix)
            rotation = extrinsic_matrix_to_rotation(extrinsic_matrix)

            # this part here does some epic pose estimation refinement
            pose_estimates.append(cam_xyz)
            rot_estimates.append(np.array(rotation))
            conf = estimate_confidence(cam_xyz, abs_distance, rotation, delta_time, tag_id)
            estimation_confidences.append(conf)

            # draw everything on the frame
            draw_tag_axis(frame, camera_oriented_axis_mat, projected_points)
            cv2.putText(frame, str(conf), (int(projected_points[2][0]) + 10, int(projected_points[2][1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 3)

    if len(pose_estimates) > 0:
        last_time = cur_time
        cam_xyz, rotation = refine_estimation(pose_estimates, rot_estimates, estimation_confidences,
                                              delta_time)
        if last_is_accurate:
            submit_final_estimation(cam_xyz, rotation)
    else:
        cam_xyz = last_pos_estimate
        last_is_accurate = False

    return [], frame, cam_xyz



