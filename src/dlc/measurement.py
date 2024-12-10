import numpy as np


class Measurement:
    @staticmethod
    def circular_distance(ang1, ang2):
        """
        Compute the circular distance between two angles.
        """

        circdist = np.angle(np.exp(1j * ang1) / np.exp(1j * ang2))

        return circdist

    @staticmethod
    def get_bodypart_angle(dataframe, partName1, partName2) -> np.array:
        """
        Retrieves the angle between two bodyparts.
        """

        bpt1 = dataframe.xs(partName1, level="bodyparts", axis=1).to_numpy()
        bpt2 = dataframe.xs(partName2, level="bodyparts", axis=1).to_numpy()

        angle = np.arctan2(bpt2[:, 1] - bpt1[:, 1], bpt2[:, 0] - bpt1[:, 0])
        return angle

    @staticmethod
    def get_animal_centroid(dataframe):
        x_coords = dataframe.xs("x", level="coords", axis=1).values
        y_coords = dataframe.xs("y", level="coords", axis=1).values

        x_center, y_center = np.nanmean(x_coords, axis=1), np.nanmean(y_coords, axis=1)

        return x_center, y_center

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
        fA = Measurement.circular_distance(absoluteAngle, ani1[ori_var])
        aBetween = Measurement.circular_distance(ani1[ori_var], ani2[ori_var])

        return fA, aBetween

    @staticmethod
    def get_bodypart_distance(dataframe, partname1, partname2):
        # retrieves the pixel distance between two bodyparts (from Tom)

        bpt1 = dataframe.xs(partname1, level="bodyparts", axis=1).to_numpy()
        bpt2 = dataframe.xs(partname2, level="bodyparts", axis=1).to_numpy()

        bptDistance = np.sqrt(
            np.sum(np.square(bpt1[:, [0, 1]] - bpt2[:, [0, 1]]), axis=1)
        )
        return bptDistance

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
        fA = Measurement.circular_distance(absoluteAngle, ani1[ori_var])
        aBetween = Measurement.circular_distance(ani1[ori_var], ani2[ori_var])

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
            Measurement.get_relative_orientations(df1, df2, ori_var="ori")
        )
        df2[f"{df2_name}_facing_angle"], df2[f"{df2_name}_ang_between"] = (
            Measurement.get_relative_orientations(df2, df1, ori_var="ori")
        )
        return df1, df2
