from typing import Optional, List

import mat73
import scipy


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
