from typing import Optional, List

import mat73
import scipy

from src.video import VideoInfo
from src.calculation.transform import Transform
from src.dlc.measurement import Measurement
from src.coordinate_transform import CoordinateTransform


class Project:
    def __init__(self, project_path, project_name: Optional[str] = None):
        self.project_path = project_path
        self.project_name = project_name

    @staticmethod
    def read_mat_file(file_name: str):
        try:
            return scipy.io.loadmat(file_name)
        except NotImplementedError:
            return mat73.loadmat(file_name)

    def read_project_data(self):
        loaded_data = []
        data_ext: List[str] = [
            "-bg.mat",
            "-calibration.mat",
            "-feat.mat",
            "-params.mat",
            "-track.mat",
        ]  # "_JAABA",

        for i in data_ext:
            loaded_data.append(
                Project.read_mat_file(
                    f"{self.project_path}/{self.project_name}/{self.project_name}{i}"
                )
            )
        return tuple(loaded_data)

    def do_transformations_on_df(
        tracking_data, video: VideoInfo, feat_=None, flyid1=0, flyid2=1
    ):
        """
        Perform transformations on the tracking data to center and rotate coordinates relative to the focal fly, and calculate additional metrics.

        Arguments:
            trk_ -- DataFrame containing tracking data
            video -- VideoInfo object containing video metadata
            feat_ -- DataFrame containing feature data (default: {None})
            flyid1 -- ID of the first fly (default: {0})
            flyid2 -- ID of the second fly (default: {1})

        Returns:
            DataFrame with transformed coordinates and additional metrics
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
            polarcoords = Measurement.cart2pol(fly2["rot_x"], fly2["rot_y"])
            fly1["targ_pos_radius"] = polarcoords[0]
            fly1["targ_pos_theta"] = polarcoords[1]
            fly1["targ_rel_pos_x"] = fly2["rot_x"]
            fly1["targ_rel_pos_y"] = fly2["rot_y"]
            return fly1

        fly1, fly2 = transform_add_polar_conversion(fly1, fly2)
        fly1, fly2 = transform_add_polar_conversion(fly2, fly1)

        # Get sizes and aggregate tracking data
        fly1 = Transform.get_target_sizes_df(fly1, fly2, xvar="pos_x", yvar="pos_y")
        fly2 = Transform.get_target_sizes_df(fly2, fly1, xvar="pos_x", yvar="pos_y")

        # Get relative velocity and aggregate feature data if provided
        if feat_ is not None:
            f_list = []
            for fi, df_ in feat_.groupby("id"):
                df_ = get_relative_velocity(
                    df_, win=1, value_var="dist_to_other", time_var="sec"
                )
                f_list.append(df_.reset_index(drop=True).iloc[:cop_ix])
            feat = pd.concat(f_list, axis=0).reset_index(drop=True)
            df = pd.concat(
                [
                    trk_,
                    feat.drop(columns=[c for c in feat.columns if c in trk_.columns]),
                ],
                axis=1,
            )
            assert df.shape[0] == trk_.shape[0], "Bad merge: {}, {}".format(
                feat.shape, trk_.shape
            )
        else:
            f_list = []
            assert (
                "dist_to_other" in trk_.columns
            ), "No feat df provided. Need dist_to_other."
            for fi, df_ in trk_.groupby("id"):
                df_ = get_relative_velocity(
                    df_, win=1, value_var="dist_to_other", time_var="sec"
                )
                f_list.append(df_.reset_index(drop=True).iloc[:cop_ix])
            df = pd.concat(f_list, axis=0).reset_index(drop=True)

        return df
