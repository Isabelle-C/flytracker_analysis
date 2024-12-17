from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.coordinate_transform import CoordinateTransform
from src.calculation.calculation import Calculation
from src.video import VideoInfo


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

        df.reset_index(drop=True)

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

    @staticmethod
    def transform_flytracker_feature(
        tracking_data, feature_data, window, value_var, time_var
    ) -> pd.DataFrame:
        """
        Perform transformations on the tracking data to center and rotate coordinates relative to the focal fly, and calculate additional metrics.

        Parameters
        ----------
        tracking_data : pd.DataFrame
            DataFrame containing tracking data.
        feature_data : pd.DataFrame
            DataFrame containing feature data.
        window : int
            Window size for calculating the velocity.
        value_var : str
            Relative distance variable to calculate position difference.
        time_var : str
            Time variable to calculate time difference.
        """
        feature_data_processed = (
            feature_data.groupby("id")
            .apply(Transform.get_relative_velocity(window, value_var, time_var))
            .reset_index(drop=True)
        )

        feat = feature_data_processed
        df = pd.concat(
            [
                tracking_data,
                feat.drop(
                    columns=[c for c in feat.columns if c in tracking_data.columns]
                ),
            ],
            axis=1,
        )

        assert (
            df.shape[0] == tracking_data.shape[0]
        ), f"Bad merge: {feat.shape}, {tracking_data.shape}"

    def do_transformations_on_df(
        tracking_data: pd.DataFrame,
        video: VideoInfo,
        window: int,
        value_var: str,
        time_var,
        feat_data: Optional[pd.DataFrame] = None,
        flyid1=0,
        flyid2=1,
    ) -> pd.DataFrame:
        """
        Perform transformations on the tracking data to center and rotate coordinates relative to the focal fly, and calculate additional metrics.

        Parameters
        ----------
        tracking_data : pd.DataFrame
            DataFrame containing tracking data.
        video : VideoInfo
            VideoInfo object containing video metadata.
        window : int
            Window size for calculating the velocity.
        value_var : str
            Transform value variable.
        time_var : str
            Time variable.
        feat_data : Optional[pd.DataFrame]
            DataFrame containing feature data.
        flyid1 : int
            ID of the first fly, default is 0.
        flyid2 : int
            ID of the second fly, default is 1.

        Returns
        -------
        DataFrame
            Transformed coordinates and additional metrics.
        """

        tracking_data = CoordinateTransform.center_coordinate_system(
            tracking_data, video.frame_width, video.frame_height
        )

        # Separate fly1 and fly2
        fly1 = (
            tracking_data[tracking_data["id"] == flyid1].copy().reset_index(drop=True)
        )
        fly2 = (
            tracking_data[tracking_data["id"] == flyid2].copy().reset_index(drop=True)
        )

        def transform_add_polar_conversion(fly1, fly2):
            # Transform coordinates for fly1
            fly1, fly2 = Transform.translate_coordinates_to_focal_fly(fly1, fly2)
            fly1, fly2 = Transform.rotate_coordinates_to_focal_fly(fly1, fly2)

            # Add polar conversion for fly1
            # TODO: check if need to flip y-axis
            polarcoords = CoordinateTransform.cart2pol(fly2["rot_x"], fly2["rot_y"])

            fly1["targ_pos_radius"] = polarcoords[0]
            fly1["targ_pos_theta"] = polarcoords[1]
            fly1["targ_rel_pos_x"] = fly2["rot_x"]
            fly1["targ_rel_pos_y"] = fly2["rot_y"]
            return fly1, fly2

        fly1, fly2 = transform_add_polar_conversion(fly1, fly2)
        fly1, fly2 = transform_add_polar_conversion(fly2, fly1)

        # Get sizes and aggregate tracking data
        fly1 = Transform.get_target_sizes_df(fly1, fly2, xvar="pos_x", yvar="pos_y")
        fly2 = Transform.get_target_sizes_df(fly2, fly1, xvar="pos_x", yvar="pos_y")

        if feat_data is not None:
            df = Transform.transform_flytracker_feature(
                tracking_data, feat_data, window, value_var, time_var
            )
            return df
        else:
            tracking_data_processed = (
                tracking_data.groupby("id")
                .apply(Transform.get_relative_velocity(window, value_var, time_var))
                .reset_index(drop=True)
            )
            return tracking_data_processed

    @staticmethod
    def smooth_and_calculate_velocity_circvar(
        df, smooth_var="ori", vel_var="ang_vel", time_var="sec", winsize=3
    ):
        """
        Smooth circular var and then calculate velocity. Takes care of NaNs.
        Assumes 'id' is in df.

        Arguments:
            df -- _description_

        Keyword Arguments:
            smooth_var -- _description_ (default: {'ori'})
            vel_var -- _description_ (default: {'ang_vel'})
            time_var -- _description_ (default: {'sec'})
            winsize -- _description_ (default: {3})

        Returns:
            _description_
        """
        df[vel_var] = np.nan
        df["{}_smoothed".format(smooth_var)] = np.nan
        for i, df_ in df.groupby("id"):
            # unwrap for continuous angles, then interpolate NaNs
            nans = df_[df_[smooth_var].isna()].index
            unwrapped = pd.Series(
                np.unwrap(df_[smooth_var].interpolate().ffill().bfill()),
                index=df_.index,
            )  # .interpolate().values))
            # replace nans
            # unwrapped.loc[nans] = np.nan
            # interpolate over nans now that the values are unwrapped
            oris = unwrapped.interpolate()
            # revert back to -pi, pi
            # oris = [util.set_angle_range_to_neg_pos_pi(i) for i in oris]
            # smooth with rolling()
            smoothed = Calculation.smooth_orientations_pandas(
                oris, winsize=winsize
            )  # smoothed = smooth_orientations(df_['ori'], winsize=3)
            # unwrap again to take difference between oris -- should look similar to ORIS
            smoothed_wrap = pd.Series(
                [Calculation.set_angle_range_to_neg_pos_pi(i) for i in smoothed]
            )
            # smoothed_wrap_unwrap = pd.Series(np.unwrap(smoothed_wrap), index=df_.index)
            # take difference
            smoothed_diff = smoothed_wrap.diff()
            smoothed_diff_range = [
                Calculation.set_angle_range_to_neg_pos_pi(i) for i in smoothed_diff
            ]
            # smoothed_diff = np.concatenate([[0], smoothed_diff])
            ori_diff_range = [
                Calculation.set_angle_range_to_neg_pos_pi(i) for i in oris.diff()
            ]
            # get angular velocity
            ang_vel_smoothed = smoothed_diff_range / df_[time_var].diff().mean()
            ang_vel = ori_diff_range / df_[time_var].diff().mean()

            df.loc[df["id"] == i, vel_var] = ang_vel
            df.loc[df["id"] == i, "{}_diff".format(smooth_var)] = ori_diff_range

            df.loc[df["id"] == i, "{}_smoothed".format(vel_var)] = ang_vel_smoothed
            df.loc[df["id"] == i, "{}_smoothed".format(smooth_var)] = smoothed_wrap
            df.loc[df["id"] == i, "{}_smoothed_range".format(smooth_var)] = [
                Calculation.set_angle_range_to_neg_pos_pi(i) for i in smoothed_wrap
            ]

        # df.loc[df[df[smooth_var].isna()].index, :] = np.nan
        bad_ixs = df[df[smooth_var].isna()]["frame"].dropna().index.tolist()
        cols = [
            c for c in df.columns if c not in ["frame", "id", "acquisition", "species"]
        ]
        df.loc[bad_ixs, cols] = np.nan

        df["{}_abs".format(vel_var)] = np.abs(df[vel_var])

        return df

    @staticmethod
    def post_transform_smoothing(
        transformed_df: pd.DataFrame, winsize=3
    ):  # heading_var='ori'):
        """
        Transform variables measured from keypoints to relative positions and angles.

        Returns:
            _description_
        """

        transformed_df["ori_deg"] = np.rad2deg(transformed_df["ori"])

        # convert centered cartesian to polar
        rad, th = CoordinateTransform.cart2pol(
            transformed_df["ctr_x"].values, transformed_df["ctr_y"].values
        )
        transformed_df["pos_radius"] = rad
        transformed_df["pos_theta"] = th

        # angular velocity
        transformed_df = Transform.smooth_and_calculate_velocity_circvar(
            transformed_df,
            smooth_var="ori",
            vel_var="ang_vel",
            time_var="sec",
            winsize=winsize,
        )

        transformed_df["ang_vel_deg"] = np.rad2deg(transformed_df["ang_vel"])
        transformed_df["ang_vel_abs"] = np.abs(transformed_df["ang_vel"])

        # targ_pos_theta
        transformed_df["targ_pos_theta_abs"] = np.abs(transformed_df["targ_pos_theta"])
        transformed_df = Transform.smooth_and_calculate_velocity_circvar(
            transformed_df,
            smooth_var="targ_pos_theta",
            vel_var="targ_ang_vel",
            time_var="sec",
            winsize=winsize,
        )

        # % smooth x, y,
        transformed_df["pos_x_smoothed"] = transformed_df.groupby("id")[
            "pos_x"
        ].transform(lambda x: x.rolling(winsize, 1).mean())
        # sign = -1 if input_is_flytracker else 1
        sign = 1
        transformed_df["pos_y_smoothed"] = sign * transformed_df.groupby("id")[
            "pos_y"
        ].transform(lambda x: x.rolling(winsize, 1).mean())

        # calculate heading
        for i, d_ in transformed_df.groupby("id"):
            transformed_df.loc[transformed_df["id"] == i, "traveling_dir"] = np.arctan2(
                d_["pos_y_smoothed"].diff(), d_["pos_x_smoothed"].diff()
            )
        transformed_df["traveling_dir_deg"] = np.rad2deg(
            transformed_df["traveling_dir"]
        )
        transformed_df = Transform.smooth_and_calculate_velocity_circvar(
            transformed_df,
            smooth_var="traveling_dir",
            vel_var="traveling_dir_dt",
            time_var="sec",
            winsize=3,
        )

        transformed_df["heading_travel_diff"] = (
            np.abs(
                np.rad2deg(transformed_df["ori"])
                - np.rad2deg(transformed_df["traveling_dir"])
            )
            % 180
        )  # % 180 #np.pi

        transformed_df["vel_smoothed"] = transformed_df.groupby("id")["vel"].transform(
            lambda x: x.rolling(winsize, 1).mean()
        )

        # calculate theta_error
        f1 = transformed_df[transformed_df["id"] == 0].copy().reset_index(drop=True)
        f2 = transformed_df[transformed_df["id"] == 1].copy().reset_index(drop=True)

        f1 = Calculation.calculate_theta_error(f1, f2, xvar="pos_x", yvar="pos_y")
        f2 = Calculation.calculate_theta_error(f1, f2, xvar="pos_x", yvar="pos_y")

        transformed_df.loc[transformed_df["id"] == 0, "theta_error"] = f1["theta_error"]
        transformed_df.loc[transformed_df["id"] == 1, "theta_error"] = f2["theta_error"]

        return transformed_df
