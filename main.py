import cv2

import show_on_field
import tag
from april_tag_utils import *

def main():
    # TODO: check if focal length values are actually good
    TAG_SIDE_LENGTH = 15.24
    TAG_DIAG_LENGTH = TAG_SIDE_LENGTH * (2 ** 0.5)

    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
    cam = cv2.VideoCapture(0)
    width = 1280
    height = 720
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, 30)
    diag = (width ** 2 + height ** 2) ** 0.5
    F_LENGTH_X_LIFECAM = (1/(math.tan(0.5355780593748425) * 2)) * width
    F_LENGTH_Y_LIFECAM = (1/(math.tan(0.3221767906849529) * 2)) * height

    # this tag's center is located at (0, 0, 1)
    debug_matrix = np.array([[1, 0, 0, -700], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ \
                   tag.rotation_matrix_affine_yaw_pitch_roll(yaw=math.pi / 2) @ \
                   np.array([[TAG_SIDE_LENGTH/2, 0, 0, 0],
                             [0, TAG_SIDE_LENGTH/2, 0, 0],
                             [0, 0, TAG_SIDE_LENGTH/2, 0],
                             [1, 1, 1, 1]])
    show_on_field.thread.start()
    while True:
        ok, frame = cam.read()
        if ok:
            proj_squares, ids = detect_april_tags(frame)
            draw(frame, proj_squares, ids)
            for square in proj_squares:
                extrinsic_matrix_tag = tag_projected_points_to_camera_extrinsic_matrix(tag=square, width=width,
                                                                                         height=height,
                                                                                         tag_shape=tag.BASIS_TAG_COORDS_MATRIX,
                                                                                         focal_length_x=F_LENGTH_X_LIFECAM,
                                                                                         focal_length_y=F_LENGTH_Y_LIFECAM)
                # camera_oriented_axis_mat = corners_to_camera_oriented_axis_matrix(camera_oriented_coords,
                #                                                                   TAG_SIDE_LENGTH/2)
                # corners_camera_oriented = extrinsic_matrix @ np.vstack([np.array(tag.BASIS_TAG_COORDS_MATRIX).transpose(),
                #                                                        [1, 1, 1, 1]])
                # camera_oriented_axis_mat = corners_to_camera_oriented_axis_matrix(corners_camera_oriented,
                #                                                                   tag.SIDE_LENGTH/2)
                camera_oriented_axis_mat = extrinsic_matrix_tag @ tag.BASIS_AXIS_MATRIX
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
                # cv2.circle(frame, (int(projected_z[0]), int(projected_z[1])), 5, [255, 0, 255], 5)
                # print(math.degrees(math.asin(0.5*((camera_oriented_coords[:3,0] + camera_oriented_coords[:3,2]) -
                #       (camera_oriented_coords[:3,1] + camera_oriented_coords[:3,3]))[2] /
                #       (0.5*(camera_oriented_coords[:3,1] + camera_oriented_coords[:3,3]))[2])))
                # print(np.linalg.norm(0.25 * ((camera_oriented_coords[:3, 0] + camera_oriented_coords[:3, 2]) +
                #                              (camera_oriented_coords[:3, 1] + camera_oriented_coords[:3, 3]))))
                # print(camera_oriented_coords)
                # extrinsic_matrix = camera_oriented_to_extrinsic_matrix(camera_oriented_axis_mat,
                #                                                        np.linalg.inv(debug_matrix))
                extrinsic_matrix = camera_oriented_axis_mat @ np.linalg.inv(debug_matrix)
                cam_xyz = extrinsic_matrix_to_camera_position(extrinsic_matrix)
                show_on_field.xyz = cam_xyz * (-0.01)
                rotation = extrinsic_matrix_to_rotation(extrinsic_matrix)
                show_on_field.rotation = rotation[1]
                print(cam_xyz)
            cv2.imshow('Display', frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()