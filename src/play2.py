import os
import numpy as np
import pandas as pd

from src.project_dlc import ProjectDLC


def transform_dlc_to_relative(df_, video_fpath=None, winsize=3):  # heading_var='ori'):
    """
    Transform variables measured from keypoints to relative positions and angles.

    Returns:
        _description_
    """
    # % Get video info
    cap = cv2.VideoCapture(video_fpath)

    # get frame info
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    # print(frame_width, frame_height) # array columns x array rows

    # input is DLC
    df_ = rem.do_transformations_on_df(df_, frame_width, frame_height)  # , fps=fps)
    df_["ori_deg"] = np.rad2deg(df_["ori"])
    # df['targ_pos_theta'] = -1*df['targ_pos_theta']

    # convert centered cartesian to polar
    rad, th = util.cart2pol(df_["ctr_x"].values, df_["ctr_y"].values)
    df_["pos_radius"] = rad
    df_["pos_theta"] = th

    # angular velocity
    df_ = util.smooth_and_calculate_velocity_circvar(
        df_, smooth_var="ori", vel_var="ang_vel", time_var="sec", winsize=winsize
    )
    # df_.loc[ (df_['ang_vel']>200) | (df_['ang_vel']<-200), 'ang_vel' ] = np.nan
    df_["ang_vel_deg"] = np.rad2deg(df_["ang_vel"])
    df_["ang_vel_abs"] = np.abs(df_["ang_vel"])

    # targ_pos_theta
    df_["targ_pos_theta_abs"] = np.abs(df_["targ_pos_theta"])
    df_ = util.smooth_and_calculate_velocity_circvar(
        df_,
        smooth_var="targ_pos_theta",
        vel_var="targ_ang_vel",
        time_var="sec",
        winsize=winsize,
    )

    # % smooth x, y,
    df_["pos_x_smoothed"] = df_.groupby("id")["pos_x"].transform(
        lambda x: x.rolling(winsize, 1).mean()
    )
    # sign = -1 if input_is_flytracker else 1
    sign = 1
    df_["pos_y_smoothed"] = sign * df_.groupby("id")["pos_y"].transform(
        lambda x: x.rolling(winsize, 1).mean()
    )

    # calculate heading
    for i, d_ in df_.groupby("id"):
        df_.loc[df_["id"] == i, "traveling_dir"] = np.arctan2(
            d_["pos_y_smoothed"].diff(), d_["pos_x_smoothed"].diff()
        )
    df_["traveling_dir_deg"] = np.rad2deg(
        df_["traveling_dir"]
    )  # np.rad2deg(np.arctan2(df_['pos_y_smoothed'].diff(), df_['pos_x_smoothed'].diff()))
    df_ = util.smooth_and_calculate_velocity_circvar(
        df_,
        smooth_var="traveling_dir",
        vel_var="traveling_dir_dt",
        time_var="sec",
        winsize=3,
    )

    df_["heading_travel_diff"] = (
        np.abs(np.rad2deg(df_["ori"]) - np.rad2deg(df_["traveling_dir"])) % 180
    )  # % 180 #np.pi

    df_["vel_smoothed"] = df_.groupby("id")["vel"].transform(
        lambda x: x.rolling(winsize, 1).mean()
    )

    # calculate theta_error
    f1 = df_[df_["id"] == 0].copy().reset_index(drop=True)
    f2 = df_[df_["id"] == 1].copy().reset_index(drop=True)
    # f1 = pp.calculate_theta_error(f1, f2, heading_var=heading_var)
    # f2 = pp.calculate_theta_error(f2, f1, heading_var=heading_var)
    f1 = the.calculate_theta_error(f1, f2, xvar="pos_x", yvar="pos_y")
    f2 = the.calculate_theta_error(f1, f2, xvar="pos_x", yvar="pos_y")

    df_.loc[df_["id"] == 0, "theta_error"] = f1["theta_error"]
    df_.loc[df_["id"] == 1, "theta_error"] = f2["theta_error"]

    return df_


def get_metrics_relative_to_focal_fly(
    acqdir,
    flyid1=0,
    flyid2=1,
    plot_checks=False,
    savedir=None,
):
    """
    Load -feat.mat and -trk.mat, do some processing, save processed df to savedir.

    Arguments:
        acqdir -- _description_

    Keyword Arguments:
        fps -- _description_ (default: {60})
        cop_ix -- _description_ (default: {None})
        movie_fmt -- _description_ (default: {'avi'})
        flyid1 -- _description_ (default: {0})
        flyid2 -- _description_ (default: {1})
        plot_checks -- _description_ (default: {False})
        savedir -- _description_ (default: {None})
    """
    # TODO: load flytracker calib_, trk_, feat_

    trk_["ori"] = (
        -1 * trk_["ori"]
    )  # flip for FT to match DLC and plot with 0, 0 at bottom left
    df_ = do_transformations_on_df(
        trk_, frame_width, frame_height, feat_=feat_, flyid1=0, flyid2=1
    )

    df_fpath = os.path.join(savedir, "{}_df.pkl".format(acq))
    with open(df_fpath, "wb") as f:
        pkl.dump(df_, f)
    print("Saved: {}".format(df_fpath))

    # % plot - sanity checks
    if plot_checks:
        fly1 = df_[df_["id"] == flyid1]
        fly2 = df_[df_["id"] == flyid2]
        # check affine transformations for centering and rotating male
        ix = 6500  # 5000 #2500 #590
        fig = plot_frame_check_affines(ix, fly1, fly2, cap, frame_width, frame_height)
        fig.text(0.1, 0.95, os.path.split(acqdir)[-1], fontsize=4)

        # check projections for calculating size based on distance and angle
        ix = 100  # 213527 #5000 #2500 #590
        for ix in [100, 3000, 5000]:
            fig = plot_frame_target_projection(
                ix, fly1, fly2, cap, xvar="pos_x", yvar="pos_y"
            )
            fig.text(0.1, 0.95, os.path.split(acqdir)[-1], fontsize=4)

    return df_


def load_and_transform_dlc(
    df,
    heading_var="ori",
    winsize=10,
):

    # transform to FlyTracker format
    df_ = convert_dlc_to_flytracker(df)

    # input is DLC
    df_ = rem.do_transformations_on_df(df_, frame_width, frame_height)  # , fps=fps)
    df_["ori_deg"] = np.rad2deg(df_["ori"])
    # df['targ_pos_theta'] = -1*df['targ_pos_theta']

    # convert centered cartesian to polar
    rad, th = util.cart2pol(df_["ctr_x"].values, df_["ctr_y"].values)
    df_["pos_radius"] = rad
    df_["pos_theta"] = th

    # angular velocity
    df_ = util.smooth_and_calculate_velocity_circvar(
        df_, smooth_var="ori", vel_var="ang_vel", time_var="sec", winsize=winsize
    )

    df_["ang_vel_deg"] = np.rad2deg(df_["ang_vel"])
    df_["ang_vel_abs"] = np.abs(df_["ang_vel"])

    # targ_pos_theta
    df_["targ_pos_theta_abs"] = np.abs(df_["targ_pos_theta"])
    df_ = util.smooth_and_calculate_velocity_circvar(
        df_,
        smooth_var="targ_pos_theta",
        vel_var="targ_ang_vel",
        time_var="sec",
        winsize=winsize,
    )

    # % smooth x, y,
    df_["pos_x_smoothed"] = df_.groupby("id")["pos_x"].transform(
        lambda x: x.rolling(winsize, 1).mean()
    )
    # sign = -1 if input_is_flytracker else 1
    sign = 1
    df_["pos_y_smoothed"] = sign * df_.groupby("id")["pos_y"].transform(
        lambda x: x.rolling(winsize, 1).mean()
    )

    # calculate heading
    for i, d_ in df_.groupby("id"):
        df_.loc[df_["id"] == i, "traveling_dir"] = np.arctan2(
            d_["pos_y_smoothed"].diff(), d_["pos_x_smoothed"].diff()
        )
    df_["traveling_dir_deg"] = np.rad2deg(
        df_["traveling_dir"]
    )  # np.rad2deg(np.arctan2(df_['pos_y_smoothed'].diff(), df_['pos_x_smoothed'].diff()))
    df_ = util.smooth_and_calculate_velocity_circvar(
        df_,
        smooth_var="traveling_dir",
        vel_var="traveling_dir_dt",
        time_var="sec",
        winsize=3,
    )

    df_["heading_travel_diff"] = (
        np.abs(np.rad2deg(df_["ori"]) - np.rad2deg(df_["traveling_dir"])) % 180
    )  # % 180 #np.pi

    df_["vel_smoothed"] = df_.groupby("id")["vel"].transform(
        lambda x: x.rolling(winsize, 1).mean()
    )

    # calculate theta_error
    f1 = df_[df_["id"] == 0].copy().reset_index(drop=True)
    f2 = df_[df_["id"] == 1].copy().reset_index(drop=True)
    f1 = pp.calculate_theta_error(f1, f2, heading_var=heading_var)
    f2 = pp.calculate_theta_error(f2, f1, heading_var=heading_var)
    df_.loc[df_["id"] == 0, "theta_error"] = f1["theta_error"]
    df_.loc[df_["id"] == 1, "theta_error"] = f2["theta_error"]

    return df_
