import numpy as np
import pandas as pd


from src.calculate_dlc import *


def get_scorer(dataframe):
    scorer = dataframe.columns.get_level_values(0).unique()[0]
    return scorer


def get_bodyparts(dataframe):
    return dataframe.columns.get_level_values(1).unique()


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
