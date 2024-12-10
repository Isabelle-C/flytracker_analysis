import numpy as np


def wrap2pi(ang: np.array) -> np.array:
    """
    Wrap a set of values to fit between zero and 2Pi.
    """

    positiveValues = ang > 0
    wrappedAng = ang % (2 * np.pi)
    # Handle edge case where angle is 0
    wrappedAng[(ang == 0) & positiveValues] = 2 * np.pi

    return wrappedAng

def cart2pol(x, y):
    '''
    Returns radius * theta in radians

    Arguments:
        x -- _description_
        y -- _description_
    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def get_bodypart_angle(dataframe, partName1, partName2) -> np.array:
    """
    Retrieves the angle between two bodyparts.
    """

    bpt1 = dataframe.xs(partName1, level="bodyparts", axis=1).to_numpy()
    bpt2 = dataframe.xs(partName2, level="bodyparts", axis=1).to_numpy()

    angle = np.arctan2(bpt2[:, 1] - bpt1[:, 1], bpt2[:, 0] - bpt1[:, 0])
    return angle


def get_animal_centroid(dataframe):
    x_coords = dataframe.xs("x", level="coords", axis=1).values
    y_coords = dataframe.xs("y", level="coords", axis=1).values

    x_center, y_center = np.nanmean(x_coords, axis=1), np.nanmean(y_coords, axis=1)

    return x_center, y_center


def circular_distance(ang1, ang2):
    # efficiently computes the circular distance between two angles (Tom/Rufei)

    circdist = np.angle(np.exp(1j * ang1) / np.exp(1j * ang2))

    return circdist

def get_relative_orientations(ani1, ani2, ori_var='heading', xvar='centroid_x', yvar='centroid_y'):

    '''
    returns facing_angle and angle_between -- facing_angle is relative to ani1

    Returns:
        _description_
    '''
    normPos = ani2[[xvar, yvar]] - ani1[[xvar, yvar]]
    absoluteAngle = np.arctan2(normPos[yvar], normPos[xvar])
    fA = circular_distance(absoluteAngle, ani1[ori_var])
    aBetween = circular_distance(ani1[ori_var], ani2[ori_var])

    return fA, aBetween


def get_bodypart_distance(dataframe, partname1, partname2):
    # retrieves the pixel distance between two bodyparts (from Tom)

    bpt1 = dataframe.xs(partname1, level="bodyparts", axis=1).to_numpy()
    bpt2 = dataframe.xs(partname2, level="bodyparts", axis=1).to_numpy()

    bptDistance = np.sqrt(np.sum(np.square(bpt1[:, [0, 1]] - bpt2[:, [0, 1]]), axis=1))
    return bptDistance
