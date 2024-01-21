import time

import cv2
import numpy as np

import settings
import show_on_field
import tag
from april_tag_utils import *

cam = cv2.VideoCapture(0)
WIDTH = 1280
HEIGHT = 720

# NOTE: ori and itay, these are not magic numbers, the weird number in the tan function is simply the cameras FOV
# divided by 2 in radians
F_LENGTH_X_LIFECAM = (1 / (math.tan(0.5355780593748425) * 2)) * WIDTH
F_LENGTH_Y_LIFECAM = (1 / (math.tan(0.3221767906849529) * 2)) * HEIGHT
lifecam_distortion_coefs = np.array([[1.01094557e-01, -8.10764739e-01, 3.23088490e-04, 4.97992890e-06, 1.48988740e+00]])

new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
    np.array([[1.15929436e+03, 0, 6.43213888e+02],
              [0, 1.08801044e+03, 3.71547461e+02],
              [0, 0, 1]]), lifecam_distortion_coefs, (WIDTH, HEIGHT), 1, (WIDTH, HEIGHT))
# undistort
mapx, mapy = cv2.initUndistortRectifyMap(np.array([[1.15929436e+03, 0, 6.43213888e+02],
                                                   [0, 1.08801044e+03, 3.71547461e+02],
                                                   [0, 0, 1]]), lifecam_distortion_coefs, None, new_camera_mtx,
                                         (WIDTH, HEIGHT), 5)

# these values are here so we could check our momentary changes and see if they make any sense
last_pos_estimate = np.array([0.0, 0.0, 0.0])
last_rot_estimate = np.array([0.0, 0.0, 0.0])
last_time = 0
last_is_accurate = False  # tells you if the last estimation is accurate

# # NOTE: ori and itay, ignore these, they are here for testing purposes
# debug_matrix = np.array([[1, 0, 0, -700],
#                          [0, 1, 0, 0],
#                          [0, 0, 1, 0],
#                          [0, 0, 0, 1]]) @ \
#                tag.rotation_matrix_affine_yaw_pitch_roll(yaw=math.pi / 2) @ \
#                np.array([[tag.SIDE_LENGTH / 2, 0, 0, 0],
#                          [0, tag.SIDE_LENGTH / 2, 0, 0],
#                          [0, 0, tag.SIDE_LENGTH / 2, 0],
#                          [1, 1, 1, 1]])
# debug_dict = {1: np.linalg.inv(np.array([[1, 0, 0, -7],
#                                          [0, 1, 0, 0],
#                                          [0, 0, 1, 0],
#                                          [0, 0, 0, 1]]) @ \
#                                tag.rotation_matrix_affine_yaw_pitch_roll(yaw=math.pi / 2) @ \
#                                np.array([[tag.SIDE_LENGTH / 2, 0, 0, 0],
#                                          [0, tag.SIDE_LENGTH / 2, 0, 0],
#                                          [0, 0, tag.SIDE_LENGTH / 2, 0],
#                                          [1, 1, 1, 1]])),
#               2: np.linalg.inv(np.array([[1, 0, 0, -7],
#                                          [0, 1, 0, 0],
#                                          [0, 0, 1, 0.56515],
#                                          [0, 0, 0, 1]]) @ \
#                                tag.rotation_matrix_affine_yaw_pitch_roll(yaw=math.pi / 2) @ \
#                                np.array([[tag.SIDE_LENGTH / 2, 0, 0, 0],
#                                          [0, tag.SIDE_LENGTH / 2, 0, 0],
#                                          [0, 0, tag.SIDE_LENGTH / 2, 0],
#                                          [1, 1, 1, 1]])),
#               3: np.linalg.inv(np.array([[1, 0, 0, -7],
#                                          [0, 1, 0, 0],
#                                          [0, 0, 1, 1.5],
#                                          [0, 0, 0, 1]]) @ \
#                                tag.rotation_matrix_affine_yaw_pitch_roll(yaw=math.pi / 2) @ \
#                                np.array([[tag.SIDE_LENGTH / 2, 0, 0, 0],
#                                          [0, tag.SIDE_LENGTH / 2, 0, 0],
#                                          [0, 0, tag.SIDE_LENGTH / 2, 0],
#                                          [1, 1, 1, 1]]))
#               }

# these values are for refining the estimation
MAX_VEL = 5  # maximum velocity of the robot, if passed we can assume there was a problem with the pose estimation
MAX_ACCEL = 15000  # maximum acceleration of the robot, if passed we can assume there was a problem with the pose
MIN_CONFIDENCE = 0.02
SPEED_WEIGHT = 10  # how much weight we give speed in confidence estimation
ROT_WEIGHT = 100  # how much weight we give the rotation in confidence estimation
QUANTIZATION_LEVELS = 12  # how many levels do we want to divide the image to

def denoise_frame(frame):
    processed_frame = copy.deepcopy(frame)
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.normalize(processed_frame, processed_frame, 0, 255, cv2.NORM_MINMAX)
    processed_frame = cv2.GaussianBlur(processed_frame, [3, 3], sigmaX=0.1, sigmaY=0.1)
    processed_frame = cv2.medianBlur(processed_frame, 3)
    processed_frame = np.round(processed_frame * (QUANTIZATION_LEVELS / 255)) * (255 / QUANTIZATION_LEVELS)
    processed_frame = np.uint8(np.round(processed_frame))
    return processed_frame


def draw_tag_axis(frame, camera_oriented_axis_mat, projected_points):
    projected_z = project_point(camera_oriented_axis_mat[:3, 2], F_LENGTH_X_LIFECAM,
                                F_LENGTH_Y_LIFECAM, WIDTH, HEIGHT)
    projected_y = project_point(camera_oriented_axis_mat[:3, 1], F_LENGTH_X_LIFECAM,
                                F_LENGTH_Y_LIFECAM, WIDTH, HEIGHT)
    projected_x = project_point(camera_oriented_axis_mat[:3, 0], F_LENGTH_X_LIFECAM,
                                F_LENGTH_Y_LIFECAM, WIDTH, HEIGHT)
    center = find_projected_tag_center(projected_points)
    cv2.line(frame, (int(projected_x[0]), int(projected_x[1])), (int(center[0]), int(center[1])),
             (255, 0, 0), 5)
    cv2.line(frame, (int(projected_y[0]), int(projected_y[1])), (int(center[0]), int(center[1])),
             (0, 255, 0), 5)
    cv2.line(frame, (int(projected_z[0]), int(projected_z[1])), (int(center[0]), int(center[1])),
             (0, 0, 255), 5)


def estimate_confidence(xyz, abs_distance, rotation, delta_time):
    return 1 / (abs_distance +
                (SPEED_WEIGHT * (np.linalg.norm(last_pos_estimate - xyz) / delta_time) * int(last_is_accurate)) +
                (ROT_WEIGHT * rotation[2]))


def submit_final_estimation(xyz: np.ndarray, rotation: list):
    # TODO: add the part that sends the data to the robot here
    show_on_field.xyz = xyz
    show_on_field.rotation = rotation[1]


def refine_estimation(pose_estimates, rot_estimates, estimation_confidences, delta_time):
    global last_is_accurate, last_rot_estimate, last_pos_estimate, last_time
    # this part refines estimation

    conf = 0
    cam_xyz = last_pos_estimate
    rotation = last_rot_estimate
    for i in range(len(pose_estimates)):
        if estimation_confidences[i] > conf:
            cam_xyz = pose_estimates[i]
            rotation = rot_estimates[i]
            conf = estimation_confidences[i]

    # comparing to the last estimation
    delta_x = cam_xyz - last_pos_estimate
    velocity = delta_x * (1 / delta_time)
    if (conf < MIN_CONFIDENCE) and (np.linalg.norm(velocity) > MAX_VEL):
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
    dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    frame = dst[y:y + h, x:x + w]

    processed_frame = denoise_frame(frame)

    # TODO: delete this later as this is for debugging
    cv2.imshow("debug", processed_frame)

    delta_time = cur_time - last_time

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
                                                                                        focal_length_x=F_LENGTH_X_LIFECAM,
                                                                                        focal_length_y=F_LENGTH_Y_LIFECAM)
            camera_oriented_axis_mat = tag_transformation_matrix @ tag.BASIS_AXIS_MATRIX
            extrinsic_matrix = camera_oriented_axis_mat @ field_oriented_inv_axis_matrix
            cam_xyz = extrinsic_matrix_to_camera_position(extrinsic_matrix)
            rotation = extrinsic_matrix_to_rotation(extrinsic_matrix)

            # this part here does some epic pose estimation refinement
            pose_estimates.append(cam_xyz)
            rot_estimates.append(np.array(rotation))
            estimation_confidences.append(estimate_confidence(cam_xyz, abs_distance, rotation, delta_time))

            # draw everything on the frame
            draw_tag_axis(frame, camera_oriented_axis_mat, projected_points)
    if len(pose_estimates) > 0:
        cam_xyz, rotation = refine_estimation(pose_estimates, rot_estimates, estimation_confidences,
                                              delta_time)
        print(cam_xyz)
        if last_is_accurate:
            submit_final_estimation(cam_xyz, rotation)
    else:
        cam_xyz = last_pos_estimate
        last_is_accurate = False
    last_time = cur_time
    return [], frame, cam_xyz



def main():
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    # cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_EXPOSURE, -7)

    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)

    show_on_field.thread.start()  # NOTE: ori and itay, yes theres a different thread for showing where everything is
    # on the field, this is because in the actual competition its the driver station computer that's gonna do it and i
    # didn't feel like actually writing the full code that does updates the other thread with UDP,
    # you are welcome to cry about it

    while True:
        ok, frame = cam.read()
        if ok:
            _, frame, _ = runPipeline(frame, None)
            cv2.imshow('Display', frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()
