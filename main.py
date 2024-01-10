import time

import cv2
import show_on_field
import tag
from april_tag_utils import *

cam = cv2.VideoCapture(0)
width = 1280
height = 720
F_LENGTH_X_LIFECAM = (1 / (math.tan(0.5355780593748425) * 2)) * width
F_LENGTH_Y_LIFECAM = (1 / (math.tan(0.3221767906849529) * 2)) * height

# this tag's center is located at (0, 0, 1)
debug_matrix = np.array([[1, 0, 0, -700], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ \
               tag.rotation_matrix_affine_yaw_pitch_roll(yaw=math.pi / 2) @ \
               np.array([[tag.SIDE_LENGTH / 2, 0, 0, 0],
                         [0, tag.SIDE_LENGTH / 2, 0, 0],
                         [0, 0, tag.SIDE_LENGTH / 2, 0],
                         [1, 1, 1, 1]])

max_v = 500  # maximum velocity of the robot, if passed we can assume there was a problem with the pose estimation
max_a = 1500  # maximum acceleration of the robot, if passed we can assume there was a problem with the pose
# estimation

def main():
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)

    show_on_field.thread.start()

    last_pos_estimate = np.array([0.0, 0.0, 0.0])
    last_rot_estimate = np.array([0.0, 0.0, 0.0])
    last_velocity = np.array([0, 0, 0])
    last_time = 0

    last_is_accurate = False  # tells you if the last estimation is accurate
    while True:
        cur_time = time.time()
        ok, frame = cam.read()
        if ok:
            proj_squares, ids = detect_april_tags(frame)
            draw(frame, proj_squares, ids)
            pose_estimates = []
            rot_estimates = []
            for i in range(len(ids)):
                square = proj_squares[i]
                extrinsic_matrix_tag = tag_projected_points_to_camera_extrinsic_matrix(tag=square, width=width,
                                                                                       height=height,
                                                                                       tag_shape=
                                                                                       tag.BASIS_TAG_COORDS_MATRIX,
                                                                                       focal_length_x=
                                                                                       F_LENGTH_X_LIFECAM,
                                                                                       focal_length_y=
                                                                                       F_LENGTH_Y_LIFECAM)
                camera_oriented_axis_mat = extrinsic_matrix_tag @ tag.BASIS_AXIS_MATRIX
                extrinsic_matrix = camera_oriented_axis_mat @ np.linalg.inv(debug_matrix)
                cam_xyz = extrinsic_matrix_to_camera_position(extrinsic_matrix)
                rotation = extrinsic_matrix_to_rotation(extrinsic_matrix)
                pose_estimates.append(cam_xyz)
                rot_estimates.append(np.array(rotation))

                # draw everything on the frame
                projected_z = project_point(camera_oriented_axis_mat[:3, 2], F_LENGTH_X_LIFECAM,
                                            F_LENGTH_Y_LIFECAM, width, height)
                projected_y = project_point(camera_oriented_axis_mat[:3, 1], F_LENGTH_X_LIFECAM,
                                            F_LENGTH_Y_LIFECAM, width, height)
                projected_x = project_point(camera_oriented_axis_mat[:3, 0], F_LENGTH_X_LIFECAM,
                                            F_LENGTH_Y_LIFECAM, width, height)
                center = find_projected_tag_center(square)

                cv2.line(frame, (int(projected_x[0]), int(projected_x[1])), (int(center[0]), int(center[1])),
                         (255, 0, 0), 5)
                cv2.line(frame, (int(projected_y[0]), int(projected_y[1])), (int(center[0]), int(center[1])),
                         (0, 255, 0), 5)
                cv2.line(frame, (int(projected_z[0]), int(projected_z[1])), (int(center[0]), int(center[1])),
                         (0, 0, 255), 5)

            # this part refines estimation
            if (len(proj_squares) > 0):

                # averaging estimations from all tags
                cam_xyz = np.array([0.0, 0.0, 0.0])
                for estimate in pose_estimates:
                    cam_xyz += estimate
                cam_xyz *= (1 / len(pose_estimates))

                rotation = np.array([0.0, 0.0, 0.0])
                for estimate in rot_estimates:
                    rotation += estimate
                rotation *= (1 / len(pose_estimates))

                # comparing to the last estimation
                delta_time = cur_time - last_time
                delta_x = cam_xyz - last_pos_estimate
                velocity = delta_x * (1 / delta_time)
                acceleration = np.linalg.norm(velocity - last_velocity) / delta_time

                if (acceleration > max_a) and last_is_accurate:
                    last_is_accurate = False
                    cam_xyz = last_pos_estimate + last_velocity*delta_time
                    rotation = last_rot_estimate
                else:
                    print(acceleration)
                    last_is_accurate = True
                last_pos_estimate = cam_xyz
                last_rot_estimate = rotation
                last_velocity = velocity

                # show on field
                show_on_field.xyz = cam_xyz * (-0.01)
                show_on_field.rotation = rotation[1]
            cv2.imshow('Display', frame)
            cv2.waitKey(1)
        last_time = cur_time


if __name__ == '__main__':
    main()
