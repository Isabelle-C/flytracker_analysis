import numpy as np
import pandas as pd

import scipy.signal as signal

from src.calculate_dlc import *

def get_scorer(dataframe):
    scorer = dataframe.columns.get_level_values(0).unique()[0]
    return scorer


def get_bodyparts(dataframe):
    return dataframe.columns.get_level_values(1).unique()

def remove_jumps(dataframe, maxJumpLength):
    """
    Remove large jumps in the x/y position of bodyparts, usually resulting from swaps between animals.
    """

    # get all column names
    scorer = dataframe.columns.get_level_values(0)[0]
    bps = list(
        dataframe.columns.get_level_values(1).unique()
    )  # list(dataframe.columns.levels[1])
    params = list(dataframe.columns.levels[2])
    dataframeMod = dataframe.copy()

    for i, partName in enumerate(bps):

        xDiff = pd.Series(np.diff(dataframe[scorer][partName]["x"]))
        yDiff = pd.Series(np.diff(dataframe[scorer][partName]["y"]))

        xJumpsPositive = signal.find_peaks(xDiff.interpolate(), threshold=200)
        xJumpsNegative = signal.find_peaks(xDiff.interpolate() * -1, threshold=200)
        yJumpsPositive = signal.find_peaks(yDiff.interpolate(), threshold=200)
        yJumpsNegative = signal.find_peaks(yDiff.interpolate() * -1, threshold=200)

        toKill = np.zeros((len(yDiff),), dtype=bool)

        for j in range(len(xJumpsPositive[0])):
            if np.any(
                (xJumpsNegative[0] > xJumpsPositive[0][j])
                & (xJumpsNegative[0] < xJumpsPositive[0][j] + maxJumpLength)
            ):
                endIdx = np.where(
                    (xJumpsNegative[0] > xJumpsPositive[0][j])
                    & (xJumpsNegative[0] < xJumpsPositive[0][j] + maxJumpLength)
                )
                toKill[xJumpsPositive[0][j] : xJumpsNegative[0][endIdx[0][0]]] = True
            else:
                toKill[xJumpsPositive[0][j]] = True

        for j in range(len(xJumpsNegative[0])):

            if np.any(
                (xJumpsPositive[0] > xJumpsNegative[0][j])
                & (xJumpsPositive[0] < xJumpsNegative[0][j] + maxJumpLength)
            ):
                endIdx = np.where(
                    (xJumpsPositive[0] > xJumpsNegative[0][j])
                    & (xJumpsPositive[0] < xJumpsNegative[0][j] + maxJumpLength)
                )
                toKill[xJumpsNegative[0][j] : xJumpsPositive[0][endIdx[0][0]]] = True
            else:
                toKill[xJumpsNegative[0][j]] = True

        for j in range(len(yJumpsPositive[0])):
            if np.any(
                (yJumpsNegative[0] > yJumpsPositive[0][j])
                & (yJumpsNegative[0] < yJumpsPositive[0][j] + maxJumpLength)
            ):
                endIdx = np.where(
                    (yJumpsNegative[0] > yJumpsPositive[0][j])
                    & (yJumpsNegative[0] < yJumpsPositive[0][j] + maxJumpLength)
                )
                toKill[yJumpsPositive[0][j] : yJumpsNegative[0][endIdx[0][0]]] = True
            else:
                toKill[yJumpsPositive[0][j]] = True

        for j in range(len(yJumpsNegative[0])):
            if np.any(
                (yJumpsPositive[0] > yJumpsNegative[0][j])
                & (yJumpsPositive[0] < yJumpsNegative[0][j] + maxJumpLength)
            ):
                endIdx = np.where(
                    (yJumpsPositive[0] > yJumpsNegative[0][j])
                    & (yJumpsPositive[0] < yJumpsNegative[0][j] + maxJumpLength)
                )
                toKill[yJumpsNegative[0][j] : yJumpsPositive[0][endIdx[0][0]]] = True
            else:
                toKill[yJumpsNegative[0][j]] = True

        toKill = np.insert(toKill, 1, False)

        dataframeMod.loc[toKill, (scorer, partName, params)] = np.nan

    return dataframeMod


def get_fly_params(flypos, win=5, fps=60):
    """
        Convert tracked DLC coords to flytracker params.
        TODO: change 'heading' to 'ori'

        Arguments:
            flypos -- _description
    _

        Keyword Arguments:
            cop_ix -- _description_ (default: {None})
            win -- _description_ (default: {5})
            fps -- _description_ (default: {60})

        Returns:
            _description_
    """
    x_center, y_center = get_animal_centroid(flypos)
    ori = get_bodypart_angle(flypos, "abdomentip", "head")

    data_dict = {"ori": ori, "centroid_x": x_center, "centroid_y": y_center}
    flydf = pd.DataFrame(data_dict)

    flydf["lin_speed"] = np.concatenate(
        (
            np.zeros(1),
            np.sqrt(
                np.sum(
                    np.square(np.diff(np.column_stack((x_center, y_center)), axis=0)),
                    axis=1,
                )
            ),
        )
    ) / (win / fps)

    leftw = get_bodypart_angle(flypos, "thorax", "wingL")
    rightw = get_bodypart_angle(flypos, "thorax", "wingR")

    flydf["left_wing_angle"] = (
        wrap2pi(circular_distance(flydf["ori"].interpolate(), leftw)) - np.pi
    )
    flydf["right_wing_angle"] = (
        wrap2pi(circular_distance(flydf["ori"].interpolate(), rightw)) - np.pi
    )
    flydf["inter_wing_dist"] = get_bodypart_distance(flypos, "wingR", "wingL")
    flydf["body_length"] = get_bodypart_distance(flypos, "head", "abdomentip")

    return flydf
