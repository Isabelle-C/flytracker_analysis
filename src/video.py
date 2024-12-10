import cv2
import numpy as np


class VideoInfo:
    def __init__(self, video_file):
        self.video_file = video_file
        self.video = cv2.VideoCapture(video_file)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        self.frame_height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        self.video.release()

    def get_fps(self):
        """
        Get the FPS of a video file.

        Returns:
            float: Frames per second of the video.
        """
        return self.fps

    def get_total_frames(self):
        """
        Get the total number of frames in a video file.

        Returns:
            int: Total number of frames in the video.
        """
        return self.total_frames

    def get_timestamp(self):
        """
        Get the timestamp for each frame in the video.

        Returns:
            numpy.ndarray: Array of timestamps for each frame.
        """
        return np.linspace(0, self.total_frames * 1 / self.fps, self.total_frames)
