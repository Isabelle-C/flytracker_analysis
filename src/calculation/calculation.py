from math import remainder, tau

import numpy as np
import pandas as pd


class Calculation:
    @staticmethod
    def circular_distance(ang1, ang2):
        """
        Compute the circular distance between two angles.
        """

        circdist = np.angle(np.exp(1j * ang1) / np.exp(1j * ang2))

        return circdist

    @staticmethod
    def proj_a_onto_b(a: np.array, b: np.array):
        """
        Projects vector a onto vector b.
        """
        return b * (np.dot(a, b) / np.linalg.norm(b) ** 2)

    @staticmethod
    def set_angle_range_to_neg_pos_pi(ang):
        return remainder(ang, tau)

    @staticmethod
    def smooth_orientations_pandas(x, winsize=3):
        # 'unwrap' the angles so there is no wrap around
        x1 = pd.Series(np.rad2deg(np.unwrap(x)))
        # smooth the data with a moving average
        # note: this is pandas 17.1, the api changed for version 18
        x2 = x1.rolling(winsize, min_periods=1).mean()  # pd.rolling_mean(x1, window=3)
        # convert back to wrapped data if desired
        x3 = x2 % 360
        return np.deg2rad(x3)

    @staticmethod
    def calculate_theta_error(f1, f2, xvar="pos_x", yvar="pos_y"):
        vec_between = f2[[xvar, yvar]] - f1[[xvar, yvar]]
        abs_ang = np.arctan2(vec_between[yvar], vec_between[xvar])
        th_err = Calculation.circular_distance(abs_ang, f1["ori"])

        f1["abs_ang_between"] = (
            abs_ang  # this is the line-of-sigh, line btween pursuer and target
        )
        f1["theta_error"] = th_err
        f1["theta_error_dt"] = (
            pd.Series(np.unwrap(f1["theta_error"].interpolate().ffill().bfill())).diff()
            / f1["sec_diff"].mean()
        )
        f1["theta_error_deg"] = np.rad2deg(f1["theta_error"])

        return f1
