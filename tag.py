import math
import numpy as np

INCHES_TO_METER = 2.54 / 100
# TODO: change to this seasons tag
SIDE_LENGTH = 15.24#0.1651
BASIS_AXIS_MATRIX = np.array([[SIDE_LENGTH/2, 0, 0, 0],
                              [0, SIDE_LENGTH/2, 0, 0],
                              [0, 0, SIDE_LENGTH/2, 0],
                              [1, 1, 1, 1]])

BASIS_TAG_COORDS_MATRIX = [np.array([0.5*SIDE_LENGTH, -0.5*SIDE_LENGTH, 0]),
                           np.array([-0.5*SIDE_LENGTH, -0.5*SIDE_LENGTH, 0]),
                           np.array([-0.5*SIDE_LENGTH, 0.5*SIDE_LENGTH, 0]),
                           np.array([0.5*SIDE_LENGTH, 0.5*SIDE_LENGTH, 0])]


def rotation_matrix_affine_yaw_pitch_roll(yaw=0, pitch=0, roll=0) -> np.ndarray:
    c1 = np.cos(pitch)
    s1 = np.sin(pitch)
    c2 = np.cos(yaw)
    s2 = np.sin(yaw)
    c3 = np.cos(roll)
    s3 = np.sin(roll)
    return np.array([[c2 * c3, -c2 * s3, s2, 0],
                     [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1, 0],
                     [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2, 0],
                     [0, 0, 0, 1]])

class Tag:
    def __init__(self, x, y, z, yaw_degrees, pitch_degrees=0, roll_degrees=0, is_inches=True):
        if is_inches:
            self.x = INCHES_TO_METER * x
            self.y = INCHES_TO_METER * y
            self.z = INCHES_TO_METER * z
        else:
            self.x = x
            self.y = y
            self.z = z
        self.yaw = math.radians(yaw_degrees)
        self.pitch = math.radians(pitch_degrees)
        self.roll = math.radians(roll_degrees)

    def to_field_axis_matrix(self) -> np.ndarray:
        return rotation_matrix_affine_yaw_pitch_roll(self.yaw, self.pitch, self.roll) @ (
                np.array([[1, 0, 0, self.x], [0, 1, 0, self.y], [0, 0, 1, self.z], [0, 0, 0, 1]]) @ BASIS_AXIS_MATRIX)

    def get_inv_field_axis_matrix(self):
        return np.linalg.inv(self.to_field_axis_matrix())


