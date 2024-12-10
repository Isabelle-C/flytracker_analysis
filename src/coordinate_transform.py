from typing import Union, Optional

import numpy as np
import pandas as pd


class CoordinateTransform:
    @staticmethod
    def wrap2pi(ang: np.array) -> np.array:
        """
        Wrap a set of values to fit between zero and 2Pi.
        """

        positiveValues = ang > 0
        wrappedAng = ang % (2 * np.pi)
        # Handle edge case where angle is 0
        wrappedAng[(ang == 0) & positiveValues] = 2 * np.pi

        return wrappedAng

    @staticmethod
    def cart2pol(x, y):
        """
        Returns radius * theta in radians

        Arguments:
            x -- _description_
            y -- _description_
        """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    @staticmethod
    def center_coordinate_system(
        df: pd.DataFrame,
        frame_width: Union[float, int],
        frame_height: Union[float, int],
        xvar: Optional[str] = "pos_x",
        yvar: Optional[str] = "pos_y",
        ctrx: Optional[str] = "ctr_x",
        ctry: Optional[str] = "ctr_y",
    ):
        """
        Center the x and y coordinates of a DataFrame by subtracting half the frame width and height from the respective coordinates.
        
        This effectively translates the coordinates so that the center of the frame is at (0, 0).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the coordinates.
        frame_width : Union[float, int]
            Width of the frame, corresponds to fly x-pos.
        frame_height : Union[float, int]
            Height of the frame, corresponds to fly y-pos.
        xvar : Optional[str]
            Column name for the x-coordinate (default is 'pos_x').
        yvar : Optional[str]
            Column name for the y-coordinate (default is 'pos_y').
        ctrx : Optional[str]
            Column name for the centered x-coordinate (default is 'ctr_x').
        ctry : Optional[str]
            Column name for the centered y-coordinate (default is 'ctr_y').

        Returns
        -------
        df : pd.DataFrame
            DataFrame with new columns `ctr_x` and `ctr_y` containing the centered coordinates.
        """
        df[ctrx] = df[xvar] - frame_width / 2
        df[ctry] = df[yvar] - frame_height / 2

        return df