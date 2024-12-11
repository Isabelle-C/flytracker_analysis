from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.coordinate_transform import CoordinateTransform
from src.calculation.calculation import Calculation


class Transform:
    @staticmethod
    def get_heading_vector(orientation: float, f_len: float) -> np.ndarray:
        """
        Calculate the heading vector of a fly based on its orientation and length.

        Parameters
        ----------
        orientation : float
            Orientation of the fly (in degrees, -180 to 180; 0 faces east, positive is counterclockwise)
        f_len : float
            Length of the fly (in pixels)

        Returns
        -------
        np.ndarray
            A numpy array representing the heading vector [x, y].
        """
        th = np.deg2rad(orientation)
        y_ = f_len / 2 * np.sin(th)
        x_ = f_len / 2 * np.cos(th)
        return np.array([x_, y_])

    @staticmethod
    def calculate_female_size_deg(xi: float, yi: float, f_ori: float, f_len: float):
        """
        Calculate size of target (defined by f_ori, f_len) in degrees of visual angle.

        Finds vector orthogonal to focal and target flies. Calculates heading of target using f_ori and f_len. Then, projects target heading onto orthogonal vector.

        Size is calculated as 2*arctan(fem_sz/(2*dist_to_other)).

        Parameters
        ----------
        xi : float
            X coordinate of the vector between the focal and target flies.
        yi : float
            Y coordinate of the vector between the focal and target flies.
        f_ori : float
            Orientation of the target fly (from FlyTracker, -180 to 180; 0 faces east, positive is counterclockwise).
        f_len : float
            Length of the target fly (from FlyTracker, in pixels).

        Returns
        -------
        float
            Calculated angular size.

        Notes
        -----
        Make sure the units are consistent (e.g., pixels for `f_len`, `xi`, `yi`).
        """
        # get orthogonal vector
        ortho_ = [yi, -xi]

        # project female heading vec onto orthog. vec
        fem_vec = Transform.get_heading_vector(f_ori, f_len)  # np.array([x_, y_])
        vproj_ = Calculation.proj_a_onto_b(fem_vec, ortho_)

        # calculate detg vis angle
        female_size = np.sqrt(vproj_[0] ** 2 + vproj_[1] ** 2) * 2  # size of female fly
        distance_to_other = np.sqrt(xi**2 + yi**2)  # euclidean distance
        female_size_deg = 2 * np.arctan(
            female_size / (2 * distance_to_other)
        )  # angular size in radians

        return female_size_deg

    @staticmethod
    def get_relative_velocity(
        df, window: Optional[int] = 1, value_var="dist_to_other", time_var="sec"
    ):
        """
        Calculate relative velocity between two flies, relative metric (one fly).

        If using FlyTracker feat.mat, dist_to_other is in mm, and time is sec.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data.
        window : Optional[int]
            Window size for calculating the velocity, by default 1.
        value_var : Optional[str]
            Relative distance variable to calculate position difference, by default 'dist_to_other'.
        time_var : Optional[str]
            Time variable to calculate time difference, by default 'sec'.

        Returns
        -------
        pd.DataFrame
            DataFrame with relative velocity columns added.
        """
        # fill nan of 1st value with 0
        # if dist incr, will be pos, if distance decr, will be neg
        df[f"{value_var}_diff"] = df[value_var].interpolate().diff().fillna(0)
        df[f"{time_var}_diff"] = df[time_var].interpolate().diff().fillna(0)

        mean_time_diff = df[f"{time_var}_diff"].mean()
        df["rel_vel"] = df[f"{value_var}_diff"] / (window * mean_time_diff)
        df["rel_vel_abs"] = df[f"{value_var}_diff"].abs() / (window * mean_time_diff)

        return df

    @staticmethod
    def get_target_sizes_df(
        fly1: pd.DataFrame, fly2: pd.DataFrame, xvar="pos_x", yvar="pos_y"
    ):
        """
        Calculate the size of the target in degrees for provided DataFrames of tracks.

        Parameters
        ----------
        fly1 : pd.DataFrame
            DataFrame of tracks for fly1 (male or focal fly).
        fly2 : pd.DataFrame
            DataFrame of tracks for fly2 (female or target fly).
        xvar : str, optional
            Position variable to use for calculating vectors (default is 'pos_x').
        yvar : str, optional
            Position variable to use for calculating vectors (default is 'pos_y').

        Returns
        -------
        DataFrame
            DataFrame fly1 with new columns 'targ_ang_size' and 'targ_ang_size_deg' representing
            the angular size of the target in radians and degrees, respectively.

        Notes
        -----
        The function calculates the angular size of the target (fly2) from the perspective of fly1.
        It uses the major and minor axis lengths of fly2 to determine the size and converts it to degrees.
        """

        def calculate_size(ix):
            xi = fly2.loc[ix, xvar] - fly1.loc[ix, xvar]
            yi = fly2.loc[ix, yvar] - fly1.loc[ix, yvar]
            f_ori = fly2.loc[ix, "ori"]
            f_len_maj = fly2.loc[ix, "major_axis_len"]
            f_len_min = fly2.loc[ix, "minor_axis_len"]

            size_maj = Transform.calculate_female_size_deg(xi, yi, f_ori, f_len_maj)
            size_min = Transform.calculate_female_size_deg(xi, yi, f_ori, f_len_min)
            return max(size_maj, size_min)

        fly1["targ_ang_size"] = fly1.index.map(calculate_size)
        fly1["targ_ang_size_deg"] = np.rad2deg(fly1["targ_ang_size"])

        return fly1

    @staticmethod
    def rotate_point(p: np.ndarray, angle: float, origin: tuple = (0, 0)) -> np.ndarray:
        """
        Rotate a point around a given origin by a specified angle.

        Parameters
        ----------
        p : np.ndarray
            The point to rotate, given as an array-like object [x, y].
        angle : float
            The angle to rotate the point, in radians.
        origin : tuple, optional
            The origin around which to rotate the point, by default (0, 0).

        Returns
        -------
        np.ndarray
            The rotated point as a numpy array [x', y'].
        """
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((rotation_matrix @ (p.T - o.T) + o.T).T)

    @staticmethod
    def translate_coordinates_to_focal_fly(
        fly1: pd.DataFrame, fly2: pd.DataFrame
    ) -> tuple:
        """
        Translate coordinates so that x, y of focal fly (fly1) is (0, 0).
        Assumes coordinates have been centered already (ctr_x, ctr_y).

        Parameters
        ----------
        fly1 : pd.DataFrame
            DataFrame of the focal fly.
        fly2 : pd.DataFrame
            DataFrame of the target fly.

        Returns
        -------
        tuple
            Tuple containing fly1 and fly2 DataFrames with new columns 'trans_x' and 'trans_y'.
            The coordinates of fly1 will be translated to (0, 0).
        """
        assert "ctr_x" in fly1.columns, "No 'ctr_x' column in fly1 DataFrame"
        assert "ctr_y" in fly1.columns, "No 'ctr_y' column in fly1 DataFrame"
        assert "ctr_x" in fly2.columns, "No 'ctr_x' column in fly2 DataFrame"
        assert "ctr_y" in fly2.columns, "No 'ctr_y' column in fly2 DataFrame"

        fly1["trans_x"] = 0
        fly1["trans_y"] = 0
        fly2["trans_x"] = fly2["ctr_x"] - fly1["ctr_x"]
        fly2["trans_y"] = fly2["ctr_y"] - fly1["ctr_y"]

        return fly1, fly2

    @staticmethod
    def rotate_coordinates_to_focal_fly(
        fly1: pd.DataFrame, fly2: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply rotation to fly2 so that fly1 is at 0 heading.
        Assumes 'ori' is a column in fly1 and fly2. (from FlyTracker)

        Parameters
        ----------
        fly1 : pd.DataFrame
            DataFrame of the focal fly.
        fly2 : pd.DataFrame
            DataFrame of the target fly.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            fly1, fly2 with columns 'rot_x' and 'rot_y'. fly1 is 0.
        """
        # Ensure the necessary columns are present
        assert "trans_x" in fly1.columns, "trans_x not found in fly1 DataFrame"

        # Calculate the orientation adjustment to align fly1 to 0 heading
        ori_vals = fly1["ori"].values  # Orientation values in radians (-pi to pi)
        ori = -1 * ori_vals  # Adjust orientation to 0 heading (East)

        # Initialize rotation columns
        fly2[["rot_x", "rot_y"]] = np.nan
        fly1[["rot_x", "rot_y"]] = 0

        # Rotate fly2 coordinates based on the adjusted orientation
        fly2[["rot_x", "rot_y"]] = [
            Transform.rotate_point(pt, ang)
            for pt, ang in zip(fly2[["trans_x", "trans_y"]].values, ori)
        ]

        # Adjust the orientation of fly2 and fly1
        fly2["rot_ori"] = fly2["ori"] + ori
        fly1["rot_ori"] = fly1["ori"] + ori  # Should be 0

        return fly1, fly2
