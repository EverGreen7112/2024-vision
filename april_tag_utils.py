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
    # TODO: make sure to switch it back to 36H11 for the actual code
    # NOTE: ori and itay you can cry me a river about the apriltag format not being a constant / parameter
    cv2.aruco_dict = cv2.aruco.DICT_APRILTAG_16h5  # cv2.aruco.DICT_APRILTAG_36H11

    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco_dict), parameters)
    proj_squares, ids, rejected_img_points = detector.detectMarkers(processed_frame)

    # this part is because for some reason if the function doesnt detect anything it doesnt simply return 2 empty lists
    if len(proj_squares) == 0:
        return [], []
    # the list comprehensions are because it likes to put every variable in the lists in a list containing only itself
    # for some ungodly reason
    return [a[0] for a in proj_squares], [a[0] for a in ids]


def project_point(point: np.ndarray, focal_length_x: float, focal_length_y: float, width: int,
                  height: int) -> np.ndarray:
    """
    this function performs a perspective projection on a given point (assumes the point was already transformed using
    the extrinsic matrix)
    :param point: the camera oriented 3d point you want to project
    :param focal_length_x: the focal length in the x axis (tan(fov_x/2)*2) * diagonal
    :param focal_length_y: the focal length in the y axis (tan(fov_y/2)*2) * diagonal
    :param width: frame width
    :param height: frame height
    :return: the 2d point in the image
    """
    intrinsic = np.array([[focal_length_x, 0, 0.5*width],
                          [0, focal_length_y, 0.5*height],
                          [0, 0, 1]])
    p = intrinsic @ point
    p /= p[2]
    return p[:2]


def find_projected_tag_center(tag: list[list[int or float]]) -> list[int or float]:
    """
    this function finds the center point of the tag in frame using the intersection of the diagonals
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
    """
    this function takes a point on the image and returns the normalized vector representing its direction in camera
    oriented coordinates
    :param point: a 2d vector of the points position in the image
    :param width: the images width (pixels)
    :param height: the images height (pixels)
    :param focal_length_x: the cameras focal length in the x axis
    :param focal_length_y: the cameras focal length in the y axis
    :return:
    """
    v = np.array([(point[0] - (width * 0.5)) / focal_length_x,
                  (point[1] - (height * 0.5)) / focal_length_y,
                  1])
    v /= np.linalg.norm(v)
    return v


def tag_projected_points_to_transform(tag: list[list[int or float]],
                                      width: int, height: int, tag_shape: np.ndarray,
                                      focal_length_x: float, focal_length_y: float) -> tuple[np.ndarray, float]:
    """
    :param tag_shape: a matrix representing the shape of the tag if the center point was (0,0,0), it starts with the
    down right corner and goes clockwise from there
    :param tag: the tag corners as projected in the image in the same order as they were identified
    :param width: the image's width in pixels
    :param height: the image's height in pixels
    :param focal_length_x: the focal length in the x axis (tan(fov_x/2)*2) * diagonal
    :param focal_length_y: the focal length in the y axis (tan(fov_y/2)*2) * diagonal
    :return: the 4*4 affine transformation matrix of the tag in camera oriented coordinates and the absolute distance
    from the camera so you could estimate the accuracy of the global coordinates calculation later
    """
    # NOTE: a little interesting thing we do here that most people don't is that we do p5p instead of p4p
    # we can do this because we also know where the tags center is located in frame, this means a slight increase in
    # accuracy
    success, rotation, translation = cv2.solvePnP(np.array([np.array([0, 0, 0])]+tag_shape),
                                                  np.array([find_projected_tag_center(tag)] + tag.tolist()),
                                                  np.array(  # this is just the intrinsic camera matrix
                                                      [[focal_length_x, 0, width / 2],
                                                       [0, focal_length_y, height / 2],
                                                       [0, 0, 1]]),
                                                  distCoeffs=None)
    mat, _ = cv2.Rodrigues(rotation)  # NOTE: ori and itay, this function is not called rodrigues because its a funny
    # name for a function (even though it is) but because its based on the rodrigues transform for rotation

    # this part here simply takes it from a 3*3 linear transformation matrix that only rotates stuff in place to a 4*4
    # affine transformation matrix that can also move stuff in 3d space
    # this part makes it 4*4
    mat = np.column_stack([mat, [0, 0, 0]])
    mat = np.row_stack([mat, [0, 0, 0, 1]])
    # and this part adds the translation to the transformation
    translation = np.array([t[0] for t in translation])
    mat[0][3] = translation[0]
    mat[1][3] = translation[1]
    mat[2][3] = translation[2]
    return mat, np.linalg.norm(translation)


def extrinsic_matrix_to_camera_position(extrinsic_matrix: np.ndarray) -> np.ndarray:
    """
    :param extrinsic_matrix: 4*4 extrinsic camera matrix
    :return: 3d vector representing the cameras position in global coordinates
    """
    rotation_matrix = np.delete(np.delete(extrinsic_matrix, 3, 0), 3, 1)
    inverse_rotation = np.linalg.inv(rotation_matrix)
    return -(inverse_rotation @ extrinsic_matrix[:3, 3])


def extrinsic_matrix_to_rotation(extrinsic_matrix: np.ndarray) -> list[float]:
    """
    :param extrinsic_matrix: 4*4 extrinsic camera matrix
    :return: the rotation around each axis
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
