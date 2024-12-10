import numpy as np

from src.calculation.calculation import Calculation


class RelativeMeasurement:
    @staticmethod
    def get_relative_orientations(
        ani1, ani2, ori_var="heading", xvar="centroid_x", yvar="centroid_y"
    ):
        """
        returns facing_angle and angle_between -- facing_angle is relative to ani1

        Returns:
            _description_
        """
        normPos = ani2[[xvar, yvar]] - ani1[[xvar, yvar]]
        absoluteAngle = np.arctan2(normPos[yvar], normPos[xvar])
        fA = Calculation.circular_distance(absoluteAngle, ani1[ori_var])
        aBetween = Calculation.circular_distance(ani1[ori_var], ani2[ori_var])

        return fA, aBetween

    @staticmethod
    def get_interfly_params(df1, df2, df1_name: str, df2_name: str):
        """
        Get interfly parameters for two dataframes.
        """
        df1_ctr = df1[["centroid_x", "centroid_y"]].values
        df2_ctr = df2[["centroid_x", "centroid_y"]].values

        interfly_dist = np.sqrt(np.sum(np.square(df2_ctr - df1_ctr), axis=1))

        df1["inter_obj_dist"] = interfly_dist
        df1[f"{df1_name}_facing_angle"], df1[f"{df1_name}_ang_between"] = (
            RelativeMeasurement.get_relative_orientations(df1, df2, ori_var="ori")
        )
        df2[f"{df2_name}_facing_angle"], df2[f"{df2_name}_ang_between"] = (
            RelativeMeasurement.get_relative_orientations(df2, df1, ori_var="ori")
        )
        return df1, df2
