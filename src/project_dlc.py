from typing import Optional
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from src.dlc.measurement import Measurement
from src.dlc.qc import QualityControl
from src.coordinate_transform import CoordinateTransform
from src.video import VideoInfo
from src.calculation.calculation import Calculation


class ProjectDLC:
    @staticmethod
    def load_data(file_path):
        return pd.read_hdf(file_path)

    @staticmethod
    def get_likelihoods(dataframe):
        likelihoods = dataframe.xs("likelihood", level="coords", axis=1)
        return likelihoods

    @staticmethod
    def filter_based_on_likelihood(flypos, pcutoff):
        likelihoods = ProjectDLC.get_likelihoods(flypos)
        cell_indices = [
            (row, col) for row, col in zip(*np.where(likelihoods < pcutoff))
        ]
        for row, col in cell_indices:
            flypos.loc[row, likelihoods.columns[col]] = np.nan

        return flypos

    @staticmethod
    def aggressive_filter_based_on_likelihood(flypos, pcutoff):
        likelihoods = ProjectDLC.get_likelihoods(flypos)
        indices = likelihoods[(likelihoods < pcutoff).any(axis=1)].index.tolist()
        print(len(indices))
        flypos.loc[indices, :] = np.nan
        return flypos

    @staticmethod
    def select_data(full_data: pd.DataFrame, obj_id: str):
        return full_data.xs(obj_id, level="individuals", axis=1)

    @staticmethod
    def get_scorer(dataframe):
        scorer = dataframe.columns.get_level_values(0).unique()[0]
        return scorer

    @staticmethod
    def get_bodyparts(dataframe):
        return dataframe.columns.get_level_values(1).unique()

    @staticmethod
    def get_fly_params(flypos, win=5, fps=60):
        """
        Convert tracked DLC coords to flytracker params.
        TODO: change 'heading' to 'ori'

        Arguments:
            flypos -- _description

        Keyword Arguments:
            cop_ix -- _description_ (default: {None})
            win -- _description_ (default: {5})
            fps -- _description_ (default: {60})

        Returns:
            _description_
        """
        x_center, y_center = Measurement.get_animal_centroid(flypos)
        ori = Measurement.get_bodypart_angle(flypos, "abdomentip", "head")

        data_dict = {"ori": ori, "centroid_x": x_center, "centroid_y": y_center}
        flydf = pd.DataFrame(data_dict)

        flydf["lin_speed"] = np.concatenate(
            (
                np.zeros(1),
                np.sqrt(
                    np.sum(
                        np.square(
                            np.diff(np.column_stack((x_center, y_center)), axis=0)
                        ),
                        axis=1,
                    )
                ),
            )
        ) / (win / fps)

        leftw = Measurement.get_bodypart_angle(flypos, "thorax", "wingL")
        rightw = Measurement.get_bodypart_angle(flypos, "thorax", "wingR")

        flydf["left_wing_angle"] = (
            CoordinateTransform.wrap2pi(
                Calculation.circular_distance(flydf["ori"].interpolate(), leftw)
            )
            - np.pi
        )
        flydf["right_wing_angle"] = (
            CoordinateTransform.wrap2pi(
                Calculation.circular_distance(flydf["ori"].interpolate(), rightw)
            )
            - np.pi
        )
        flydf["inter_wing_dist"] = Measurement.get_bodypart_distance(
            flypos, "wingR", "wingL"
        )
        flydf["body_length"] = Measurement.get_bodypart_distance(
            flypos, "head", "abdomentip"
        )

        return flydf

    @staticmethod
    def process_single_fly(
        full_data,
        obj_id,
        video: VideoInfo,
        pcutoff: Optional[float] = 0.9,
        crop_ix: Optional[int] = None,
    ):
        """
        Select and process data for a single fly.
        """
        flypos = ProjectDLC.select_data(full_data, obj_id)

        flypos = QualityControl.remove_jumps(flypos, 6)

        if crop_ix is not None:
            flypos.iloc[: int(crop_ix)]

        flypos = ProjectDLC.filter_based_on_likelihood(flypos, pcutoff)
        flydf = ProjectDLC.get_fly_params(flypos)

        flydf["time"] = video.get_timestamp()

        return flydf

    @staticmethod
    def convert_dlc_to_flytracker(df, mm_per_pix=None, body_length_mm=2):
        """
        Convert DLC output to FlyTracker format.

        Parameters
        ----------
        df : pd.DataFrame
            DLC output DataFrame.
        mm_per_pix : float, optional
            Millimeters per pixel, by default None.
        body_length_mm : int, optional
            Body length in mm, by default 2.

        Returns
        -------
        pd.DataFrame
            DataFrame with converted columns.
        """
        df = df.rename(columns={"dist_to_other": "dist_to_other_pix"})

        if mm_per_pix is None:
            arena_size_mm = 38 - body_length_mm * 2
            max_dist_found = df["dist_to_other_pix"].max()
            mm_per_pix = arena_size_mm / max_dist_found

        # convert units to mm/s and mm (like FlyTracker)
        df["vel"] = df["lin_speed"] * mm_per_pix
        df["dist_to_other"] = df["dist_to_other_pix"] * mm_per_pix
        df["pos_x_mm"] = df["centroid_x"] * mm_per_pix
        df["pos_y_mm"] = df["centroid_y"] * mm_per_pix

        df["mm_to_pix"] = mm_per_pix

        # rename columns to get RELATIVE pos info
        df = df.rename(
            columns={
                "centroid_x": "pos_x",
                "centroid_y": "pos_y",
                "heading": "ori",
                "body_length": "major_axis_len",
                "inter_wing_dist": "minor_axis_len",
                "time": "sec",
            }
        )
        return df

    # @staticmethod
    # def merge_dfs_and_add_interfly_data(df_fly1, df_fly2, species_name):
    #     """
    #     Merge two fly dataframes and calculate interfly metrics.
    #     """
    #     df_fly1["id"] = 0
    #     df_fly2["id"] = 1

    #     # Get metrics between the two objects
    #     df_fly1, df_fly2 = Measurement.get_interfly_params(
    #         df_fly1, df_fly2, "fly1", "fly2"
    #     )

    #     df = pd.concat([df_fly1, df_fly2], axis=0)
    #     df["species"] = species_name

    #     return df
