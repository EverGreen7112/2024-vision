import copy
import math
import numpy as np
import cv2


def detect_april_tags(image: np.ndarray) -> tuple[list[list[list[int]]], list[int]]:
    """
    :param image: the image where we want to detect
    :return: an array of the detected april tags and an array of their id's
    """
    processed_frame = copy.deepcopy(image)
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.DICT_APRILTAG_16h5  # cv2.aruco.DICT_APRILTAG_36H11
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco_dict), parameters)
    proj_squares, ids, rejected_img_points = detector.detectMarkers(processed_frame)
    if (proj_squares is ()):
        return [], []
    return [a[0] for a in proj_squares], [a[0] for a in ids]


def project_point(point: np.ndarray, focal_length_x: float, focal_length_y: float, width: int,
                  height: int) -> np.ndarray:
    """
    :param point: the camera oriented 3d point you want to project
    :param focal_length_x: the focal length in the x axis (tan(fov_x/2)*2) * diagonal
    :param focal_length_y: the focal length in the y axis (tan(fov_y/2)*2) * diagonal
    :param width: frame width
    :param height: frame height
    :return: the point in the frame
    """
    intrinsic = np.array([[focal_length_x, 0, width],
                          [0, focal_length_y, height],
                          [0, 0, 1]])
    p = intrinsic @ point
    p /= p[2]
    return p[:2] - (0.5 * np.array([width, height]))


def find_projected_tag_center(tag: list[list[int or float]]) -> list[int or float]:
    """
    :param tag: a 2d list where each point is a point on the tag as detected by the detector
    :return: a list that represents the 2d vector of the tag's center coordinates on the image
    """
    p1 = tag[0]
    p2 = tag[1]
    p3 = tag[2]
    p4 = tag[3]
    line1 = (p1, p3)
    line2 = (p2, p4)
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]


def screen_point_to_normalized_vector(point: list[int or float], width: int, height: int,
                                      focal_length_x: float, focal_length_y: float) -> np.ndarray:
    v = np.array([(point[0] - (width * 0.5)) / focal_length_x,
                  (point[1] - (height * 0.5)) / focal_length_y,
                  1])
    v /= np.linalg.norm(v)
    return v


# def p3p_collinear_equal_distances(p1_normalized: np.ndarray, p2_normalized: np.ndarray,
#                                   center_normalized: np.ndarray,
#                                   diagonal_length: float) -> tuple[
#                                     np.ndarray, np.ndarray, float, float]:
#     """
#     :param p1_normalized:  the first point normalized to 1
#     :param p2_normalized:  the oposite point to p1 normalized to 1
#     :param center_normalized:  the center point normalized to 1
#     :param diagonal_length: the length of the diagonal
#     :return: 2 guesses for p2 normalized to fit |p1| = 1 and the fitting 2 guesses for the scale
#     """
#     cos = math.cos
#     alpha = math.acos(np.dot(p1_normalized, center_normalized))  # the angle between the center and p1
#     beta = math.acos(np.dot(p2_normalized, center_normalized))  # the angle between center and p2
#     gama = alpha + beta  # the angle between p1 and p2
#
#     # TODO: replace the variables to optimize
#     # a = diagonal_length * 0.5
#     # b = a
#     # c = diagonal_length
#     # abc_rep = ((a**2-c**2)/(b**2))
#
#     # factors of the polynomial
#     # a4 = (abc_rep - 1)**2 - (4*c**2/b**2)*cos(alpha)**2
#     # a3 = 4*(abc_rep * (1-abc_rep) * cos(beta) - (1-((a**2+c**2)/b**2)) * cos(alpha)*cos(gama) + 2*((c**2)/b**2)*(cos(alpha)**2)*cos(beta))
#     # a2 = 2*((abc_rep**2) - 1 + (2*(abc_rep**2)*(cos(beta)**2)) + (2*abc_rep*(cos(alpha)**2)) - (4*((a**2+c**2)/(b**2))*cos(alpha)*cos(beta)*cos(gama)) + 2*abc_rep*(cos(gama)**2))
#     # a1 = 4*((-abc_rep)*(1+abc_rep)*cos(beta) + ((2*a**2) / (b**2))*(cos(gama)**2)*cos(beta) - (1 - ((a**2+c**2)/b**2))*cos(alpha)*cos(gama))
#
#     a4 = 16 * (1 - cos(alpha) ** 2)
#     a3 = 16 * ((-3 * cos(beta)) + (cos(alpha) * cos(gama)) + (2 * (cos(alpha) ** 2) * cos(beta)))
#     a2 = 16 + (12 * ((3 * (cos(beta) ** 2)) - (cos(alpha) ** 2) - ((20 / 6) * cos(alpha) * cos(beta) * cos(gama)) - (
#                 cos(gama) ** 2)))
#     a1 = 8 * ((-3 * cos(beta)) + ((cos(gama) ** 2) * cos(beta)) + (2 * cos(alpha) * cos(gama)))
#     a0 = 4 * (1 - (cos(gama) ** 2))
#     # all of these variables are either taken from the quadratic formula or written as variables so the lines wont be
#     # too long and so we wouldn't have to calculate the same number over and over for different solutions
#
#     d0 = (a2 ** 2) - (3 * a3 * a1) + 12 * (a4 * a0)
#     d1 = (2 * (a2 ** 3)) - (9 * a3 * a2 * a1) + (27 * (a3 ** 2) * a0) + (27 * a4 * (a1 ** 2)) - (72 * a4 * a2 * a0)
#     p = ((8 * a4 * a2) - (3 * (a3 ** 2))) / (8 * (a4 ** 2))
#     q = ((a3 ** 3) - (4 * a4 * a2) + (8 * (a4 ** 2) * a1)) / (8 * (a4 ** 3))
#     t = (0.5 * (d1 + (((d1 ** 2) + (4 * (d0 ** 3))) ** 0.5))) ** (1 / 3)
#     s = 0.5 * (((-p * (2 / 3)) + ((t + (d0 / t)) / (3 * a4))) ** 0.5)
#
#     r1 = 0.5 * (((-4 * (s ** 2)) - ((2 * p) + (q / s))) ** 0.5)
#     r2 = 0.5 * (((-4 * (s ** 2)) - ((2 * p) - (q / s))) ** 0.5)
#     w = -(a3 / (4 * a4))
#
#     # let v be the ratio |center|/|p1|
#     v1 = w - s + r1
#     v2 = w - s - r1
#     v3 = w + s + r2
#     v4 = w + s - r2
#
#     r1 = 0
#     r2 = 0
#     if (not isinstance(v4, complex)) and (v4 > 0):
#         r1 = v4
#     for v in [v3, v2, v1]:
#         if (not isinstance(v, complex)) and (v > 0):
#             if not r1:
#                 r1 = v
#             else:
#                 r2 = v
#         if (r2):
#             break
#     scale1 = diagonal_length / np.linalg.norm(2 * ((center_normalized * r1) - p1_normalized))
#     scale2 = diagonal_length / np.linalg.norm(2 * ((center_normalized * r2) - p1_normalized))
#
#     # scale = diagonal_length / np.linalg.norm(p2 - p1)
#     # p1 *= scale
#     # p2 *= scale
#     return p2_normalized*r1, p2_normalized*r2, scale1, scale2


def tag_projected_points_to_camera_extrinsic_matrix(tag: list[list[int or float]],
                                                    width: int, height: int, tag_shape: np.ndarray,
                                                    focal_length_x: float, focal_length_y: float) -> np.ndarray:
    """
    :param tag_shape: a matrix representing the shape of the tag if the center point was (0,0,0), it starts with the
    down right corner and goes clockwise from there
    :param tag: the tag corners as projected in the image in the same order as they were identified
    :param width: the images's width in pixels
    :param height: the image's height in pixels
    :param focal_length_x: the focal length in the x axis (tan(fov_x/2)*2) * diagonal
    :param focal_length_y: the focal length in the y axis (tan(fov_y/2)*2) * diagonal
    :return: the 4*4 extrinsic matrix of the camera in tag oriented coordinates
    """
    success, rotation, translation = cv2.solvePnP(np.array([np.array([0, 0, 0])]+tag_shape),
                                                  np.array([find_projected_tag_center(tag)] + tag.tolist()),
                                                  np.array(
                                                      [[focal_length_x, 0, width / 2], [0, focal_length_y, height / 2],
                                                       [0, 0, 1]]),
                                                  distCoeffs=None)

    # normalized_corners = [screen_point_to_normalized_vector(corner, width, height, focal_length_x, focal_length_y)
    #                       for corner in tag]
    # normalized_center = screen_point_to_normalized_vector(find_projected_tag_center(tag),
    #                                                       width, height, focal_length_x, focal_length_y)
    # p1, p3, scale11, scale12 = p3p_collinear_equal_distances(normalized_corners[0], normalized_corners[2],
    #                                                          normalized_center, diagonal_length)
    # p2, p4, scale21, scale22 = p3p_collinear_equal_distances(normalized_corners[1], normalized_corners[3],
    #                                                          normalized_center, diagonal_length)
    #
    # center1 = (p1 + p3) * 0.5
    # center2 = (p2 + p4) * 0.5
    # scale_tup = (scale11, scale21)
    # min_mag = np.linalg.norm(center1 * scale11 - center2 * scale21)
    # for scale_1 in (scale11, scale12):
    #     for scale_2 in (scale21, scale22):
    #         mag = np.linalg.norm(center1 * scale_1 - center2 * scale_2)
    #         print(center1 * scale_1)
    #         if (mag < min_mag):
    #             min_mag = mag
    #             scale_tup = (scale_1, scale_2)
    # print(min_mag)
    #
    # # print(np.linalg.norm(center1 * scale11 - center2 * scale21))
    # # print(np.linalg.norm(center1 * scale11 - center2 * scale22))
    # # print(np.linalg.norm(center1 * scale12 - center2 * scale21))
    # # print(np.linalg.norm(center1 * scale12 - center2 * scale22))
    # d1_scale, d2_scale = scale_tup
    # p1 *= d1_scale
    # p3 *= d1_scale
    # p2 *= d2_scale
    # p4 *= d2_scale
    # return np.column_stack([p1, p2, p3, p4])
    # print([math.degrees(r) for r in rotation])
    mat, _ = cv2.Rodrigues(rotation)
    # translation = mat @ np.array([t[0] for t in translation] + [1])
    # print(translation)
    mat = np.column_stack([mat, [0, 0, 0]])
    mat = np.row_stack([mat, [0, 0, 0, 1]])
    mat[0][3] = translation[0][0]
    mat[1][3] = translation[1][0]
    mat[2][3] = translation[2][0]
    return mat


def corners_to_camera_oriented_axis_matrix(corners: np.ndarray, half_tag_side_length: float) -> np.ndarray:
    """
    :param corners: the 3*4 matrix of the tags corners
    :param half_tag_side_length: the length of half a tag's side
    :return: a 4*4 matrix of the location of the xyz vectors with the length of half the side length and the 3d 0 vector
    with the last row being 1's in camera oriented coordinates
    """
    center = 0.5 * (corners[:3, 0] + corners[:3, 2])

    x = 0.5 * (corners[:3, 0] + corners[:3, 3]) - center
    x *= (half_tag_side_length / np.linalg.norm(x))
    x += center
    y = 0.5 * (corners[:3, 2] + corners[:3, 3]) - center
    y *= (half_tag_side_length / np.linalg.norm(y))
    y += center

    cross = np.cross(corners[:3, 1] - center, corners[:3, 0] - center)
    z = center + (cross / np.linalg.norm(cross) * half_tag_side_length)
    return np.vstack([np.column_stack([x, y, z, center]), [1, 1, 1, 1]])


def camera_oriented_to_extrinsic_matrix(camera_oriented_tag_matrix: np.ndarray, field_oriented_tag_inverse_matrix) \
        -> np.ndarray:
    """
    :param camera_oriented_tag_matrix:  the 4*4 matrix of the camera oriented position of 4 points with
    the last row being all 1's
    :param field_oriented_tag_inverse_matrix: the inverse matrix of the 4*4 matrix that represents the field oriented
    position of each point in camera_oriented_tag_matrix and the last row being all 1's, you can compute this matrix one
    time for every tag before runtime as it is constant for each tag
    :return: the camera extrinsic matrix (4*4)
    """
    return camera_oriented_tag_matrix @ field_oriented_tag_inverse_matrix


def extrinsic_matrix_to_camera_position(extrinsic_matrix: np.ndarray) -> np.ndarray:
    """
    :param extrinsic_matrix: 4*4 extrinsic camera matrix
    :return: 3d vector representing
    """
    rotation_matrix = np.delete(np.delete(extrinsic_matrix, 3, 0), 3, 1)
    det = np.linalg.det(rotation_matrix)
    # rotation_matrix /= det
    # inverse_rotation = np.linalg.inv(rotation_matrix*(det**(-1/3)))
    inverse_rotation = np.linalg.inv(rotation_matrix)
    return (inverse_rotation @ extrinsic_matrix[:3, 3])


def extrinsic_matrix_to_rotation(extrinsic_matrix: np.ndarray) -> list[float]:
    """
    :param extrinsic_matrix: 4*4 extrinsic camera matrix
    :return: the list [yaw, pitch, roll]
    """
    sy = math.sqrt(extrinsic_matrix[0, 0] * extrinsic_matrix[0, 0] + extrinsic_matrix[1, 0] * extrinsic_matrix[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(extrinsic_matrix[2, 1], extrinsic_matrix[2, 2])
        y = math.atan2(-extrinsic_matrix[2, 0], sy)
        z = math.atan2(extrinsic_matrix[1, 0], extrinsic_matrix[0, 0])
    else:
        x = math.atan2(-extrinsic_matrix[1, 2], extrinsic_matrix[1, 1])
        y = math.atan2(-extrinsic_matrix[2, 0], sy)
        z = 0

    return [x, y, z]


def draw(frame: np.ndarray, proj_tags: list[list[list[int or float]]], ids: list[int]):
    """
    :param frame: the frame on which we want to draw the tags
    :param proj_tags: a list of the projected tags coordinate on the image
    :param ids: a list of the id's of the tags
    """
    for i in range(len(ids)):
        middle = find_projected_tag_center(proj_tags[i])
        cv2.circle(frame, (int(middle[0]), int(middle[1])), 5, [0, 0, 255], 5)
        for j in range(len(proj_tags[i])):
            cv2.circle(frame, (int(proj_tags[i][j][0]), int(proj_tags[i][j][1])), 5, [255, 0, 0], 5)
            cv2.putText(frame, str(j), (int(proj_tags[i][j][0]) + 10, int(proj_tags[i][j][1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 3)
