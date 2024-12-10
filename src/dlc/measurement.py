import numpy as np


class Measurement:
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
    def get_bodypart_distance(dataframe, partname1, partname2):
        # retrieves the pixel distance between two bodyparts (from Tom)

        bpt1 = dataframe.xs(partname1, level="bodyparts", axis=1).to_numpy()
        bpt2 = dataframe.xs(partname2, level="bodyparts", axis=1).to_numpy()

        bptDistance = np.sqrt(
            np.sum(np.square(bpt1[:, [0, 1]] - bpt2[:, [0, 1]]), axis=1)
        )
        return bptDistance
