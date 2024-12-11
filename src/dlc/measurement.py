from typing import Tuple

import numpy as np
import pandas as pd


class Measurement:
    @staticmethod
    def get_bodypart_angle(
        dataframe: pd.DataFrame, partName1: str, partName2: str
    ) -> np.ndarray:
        """
        Retrieves the angle between two bodyparts.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing bodypart coordinates.
        partName1 : str
            The name of the first bodypart.
        partName2 : str
            The name of the second bodypart.

        Returns
        -------
        np.ndarray
            An array of angles between the two bodyparts.
        """

        bpt1 = dataframe.xs(partName1, level="bodyparts", axis=1).to_numpy()
        bpt2 = dataframe.xs(partName2, level="bodyparts", axis=1).to_numpy()

        angle = np.arctan2(bpt2[:, 1] - bpt1[:, 1], bpt2[:, 0] - bpt1[:, 0])
        return angle

    @staticmethod
    def get_animal_centroid(dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the centroid (average x and y coordinates) of an animal from a given DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame)
            Data containing the x and y coordinates of the animal.

            The DataFrame is expected to have a multi-index with 'coords' as one of the levels.

        Returns
        -------
        : Tuple[np.ndarray, np.ndarray]
            - x_center: The average x coordinates of the animal.
            - y_center: The average y coordinates of the animal.
        """
        x_coords = dataframe.xs("x", level="coords", axis=1).values
        y_coords = dataframe.xs("y", level="coords", axis=1).values

        x_center, y_center = np.nanmean(x_coords, axis=1), np.nanmean(y_coords, axis=1)

        return x_center, y_center

    @staticmethod
    def get_bodypart_distance(dataframe, partname1, partname2):
        """
        Calculate the Euclidean distance between two specified body parts.

        This method retrieves the pixel distance between two body parts from a given DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Data containing body part coordinates.
        partname1 : str
            The name of the first body part.
        partname2 : str
            The name of the second body part.

        Returns
        -------
        : np.ndarray
            An array of distances between the specified body parts for each frame.
        """
        bpt1 = dataframe.xs(partname1, level="bodyparts", axis=1).to_numpy()
        bpt2 = dataframe.xs(partname2, level="bodyparts", axis=1).to_numpy()

        bptDistance = np.sqrt(
            np.sum(np.square(bpt1[:, [0, 1]] - bpt2[:, [0, 1]]), axis=1)
        )
        return bptDistance
