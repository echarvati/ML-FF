import numpy as np
import math

def calc_dh(coords_c, coords_v, coords_u, coords_w):

    b1 = coords_c - coords_v
    b2 = coords_v - coords_u
    b3 = coords_u - coords_w

    # Dihedral
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    unit_b2 = b2 / np.linalg.norm(b2)
    m1 = np.cross(n1, unit_b2)

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    dihedral = math.degrees(np.arctan2(x, y))
    if dihedral < 0.0:
        dihedral = dihedral + 360


    return dihedral



def calc_angle(coords_c, coords_v, coords_u ):

    cv = coords_v - coords_c
    vu = coords_v - coords_u

    ang = math.degrees(np.arccos(np.inner(cv, vu) / (np.linalg.norm(cv) * np.linalg.norm(vu))))

    return ang