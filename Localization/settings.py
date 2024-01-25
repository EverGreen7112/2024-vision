from tag import Tag


TAGS = {1: Tag(x=593.68, z=9.68, y=53.38, yaw_degrees=120),
        2: Tag(x=637.21, z=34.79, y=53.38, yaw_degrees=120),
        3: Tag(x=652.73, z=196.17, y=57.13, yaw_degrees=180),
        4: Tag(x=652.73, z=218.42, y=57.13, yaw_degrees=180),
        5: Tag(x=578.77, z=323.00, y=53.38, yaw_degrees=270),
        6: Tag(x=72.5, z=323.00, y=53.38, yaw_degrees=270),
        7: Tag(x=-1.50, z=218.42, y=57.13, yaw_degrees=0),
        8: Tag(x=-1.50, z=196.17, y=57.13, yaw_degrees=0),
        9: Tag(x=14.02, z=34.79, y=53.38, yaw_degrees=60),
        10: Tag(x=57.54, z=9.68, y=53.38, yaw_degrees=60),
        11: Tag(x=468.69, z=146.19, y=52.00, yaw_degrees=300),
        12: Tag(x=468.69, z=177.10, y=52.00, yaw_degrees=60),
        13: Tag(x=441.74, z=161.62, y=52.00, yaw_degrees=180),
        14: Tag(x=209.48, z=161.62, y=52.00, yaw_degrees=0),
        15: Tag(x=182.73, z=177.10, y=52.00, yaw_degrees=120),
        16: Tag(x=182.73, z=146.19, y=52.00, yaw_degrees=240)
        }

TAGS_INVERSE = {tag_id: TAGS[tag_id].get_inv_field_axis_matrix() for tag_id in TAGS.keys()}
