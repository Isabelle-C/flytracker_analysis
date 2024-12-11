import os
import numpy as np
import pandas as pd

from src.project_dlc import ProjectDLC
from src.dlc.measurement import Measurement
from src.coordinate_transform import CoordinateTransform
from src.video import VideoInfo


def get_metrics_relative_to_focal_fly(
    acqdir,
    mov_is_upstream=False,
    fps=60,
    cop_ix=None,
    movie_fmt="avi",
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
        trk_, frame_width, frame_height, feat_=feat_, cop_ix=cop_ix, flyid1=0, flyid2=1
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


def get_target_sizes_df(fly1, fly2, xvar="pos_x", yvar="pos_y"):
    """
    For provided df of tracks (FlyTracker), calculates the size of target in deg.

    Arguments:
        fly1 -- df of tracks.mat for fly1 (male or focal fly)
        fly2 -- df of tracks.mat for fly2 (female or target fly)

    Keyword Arguments:
        xvar -- position var to use for calculating vectors (default: {'pos_x'})
        yvar -- same as xvar (default: {'pos_y'})

    Returns:
        fly2 -- returns fly2 with new column 'size_deg'
    """
    fem_sizes = []
    for ix in fly1.index.tolist():
        xi = fly2.loc[ix][xvar] - fly1.loc[ix][xvar]
        yi = fly2.loc[ix][yvar] - fly1.loc[ix][yvar]
        f_ori = fly2.loc[ix]["ori"]
        f_len_maj = fly2.loc[ix]["major_axis_len"]
        f_len_min = fly2.loc[ix]["minor_axis_len"]
        # take into account major/minor axes of ellipse
        fem_sz_deg_maj = util.calculate_female_size_deg(xi, yi, f_ori, f_len_maj)
        fem_sz_deg_min = util.calculate_female_size_deg(xi, yi, f_ori, f_len_min)
        fem_sz_deg = np.max([fem_sz_deg_maj, fem_sz_deg_min])
        fem_sizes.append(fem_sz_deg)

    # fly2['targ_ang_size'] = fem_sizes
    # fly2['targ_ang_size_deg'] = np.rad2deg(fly2['targ_ang_size'])
    # copy same info for f1
    fly1["targ_ang_size"] = fem_sizes
    fly1["targ_ang_size_deg"] = np.rad2deg(fly1["targ_ang_size"])

    return fly1  # , fly2


def do_transformations_on_df(trk_, video: VideoInfo, feat_=None, flyid1=0, flyid2=1):

    # center x- and y-coordinates
    trk_ = CoordinateTransform.center_coordinate_system(
        trk_, video.frame_width, video.frame_height
    )

    # separate fly1 and fly2
    fly1 = trk_[trk_["id"] == flyid1].copy().reset_index(drop=True)
    fly2 = trk_[trk_["id"] == flyid2].copy().reset_index(drop=True)

    # FIRST, do fly1: -------------------------------------------------
    # translate coordinates so that focal fly is at origin
    fly1, fly2 = util.translate_coordinates_to_focal_fly(fly1, fly2)

    # rotate coordinates so that fly1 is facing 0 degrees (East)
    # Assumes fly1 ORI goes from 0 to pi CCW, with y-axis NOT-inverted.
    # if using FlyTracker, trk_['ori'] = -1*trk_['ori']
    fly1, fly2 = util.rotate_coordinates_to_focal_fly(fly1, fly2)

    # add polar conversion
    # FLIP y-axis? TODO check this
    polarcoords = Measurement.cart2pol(fly2["rot_x"], fly2["rot_y"])
    fly1["targ_pos_radius"] = polarcoords[0]
    fly1["targ_pos_theta"] = polarcoords[1]
    fly1["targ_rel_pos_x"] = fly2["rot_x"]
    fly1["targ_rel_pos_y"] = fly2["rot_y"]

    # NOW, do fly2: ----------------------------------------------------
    # translate coordinates so that focal fly is at origin
    fly2, fly1 = util.translate_coordinates_to_focal_fly(fly2, fly1)

    # rotate coordinates so that fly1 is facing 0 degrees (East)
    # Assumes fly1 ORI goes from 0 to pi CCW, with y-axis NOT-inverted.
    # if using FlyTracker, trk_['ori'] = -1*trk_['ori']
    fly2, fly1 = util.rotate_coordinates_to_focal_fly(fly2, fly1)

    # add polar conversion
    # FLIP y-axis? TODO check this
    polarcoords = Measurement.cart2pol(fly1["rot_x"], fly1["rot_y"])
    fly2["targ_pos_radius"] = polarcoords[0]
    fly2["targ_pos_theta"] = polarcoords[1]
    fly2["targ_rel_pos_x"] = fly1["rot_x"]
    fly2["targ_rel_pos_y"] = fly1["rot_y"]

    # % copulation index - TMP: fix this!
    if cop_ix is None or np.isnan(cop_ix):
        cop_ix = len(fly1)
        copulation = False
    else:
        copulation = True
    cop_ix = int(cop_ix)

    # % Get all sizes and aggregate trk df
    fly1 = get_target_sizes_df(fly1, fly2, xvar="pos_x", yvar="pos_y")
    # Repeat for fly2:
    fly2 = get_target_sizes_df(fly2, fly1, xvar="pos_x", yvar="pos_y")

    # recombine trk df
    trk = pd.concat([fly1.iloc[:cop_ix], fly2.iloc[:cop_ix]], axis=0).reset_index(
        drop=True
    )  # .sort_index()
    trk["copulation"] = copulation

    # Get relative velocity and aggregate feat df
    if feat_ is not None:
        f_list = []
        for fi, df_ in feat_.groupby("id"):
            df_ = get_relative_velocity(
                df_, win=1, value_var="dist_to_other", time_var="sec"
            )
            f_list.append(df_.reset_index(drop=True).iloc[:cop_ix])
        feat = pd.concat(f_list, axis=0).reset_index(drop=True)  # .sort_index()
        feat["copulation"] = copulation
        print(trk.iloc[-1].name, feat.iloc[-1].name)
        df = pd.concat(
            [trk, feat.drop(columns=[c for c in feat.columns if c in trk.columns])],
            axis=1,
        )
        assert df.shape[0] == trk.shape[0], "Bad merge: {}, {}".format(
            feat.shape, trk.shape
        )
    else:
        f_list = []
        assert (
            "dist_to_other" in trk.columns
        ), "No feat df provided. Need dist_to_other."
        for fi, df_ in trk.groupby("id"):
            df_ = get_relative_velocity(
                df_, win=1, value_var="dist_to_other", time_var="sec"
            )
            f_list.append(df_.reset_index(drop=True).iloc[:cop_ix])
        df = pd.concat(f_list, axis=0).reset_index(drop=True)  # .sort_index()

    # acq = os.path.split(acqdir)[-1]

    return df


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
    f1 = pp.calculate_theta_error(f1, f2, heading_var=heading_var)
    f2 = pp.calculate_theta_error(f2, f1, heading_var=heading_var)
    df_.loc[df_["id"] == 0, "theta_error"] = f1["theta_error"]
    df_.loc[df_["id"] == 1, "theta_error"] = f2["theta_error"]

    return df_


def convert_dlc_to_flytracker(df, mm_per_pix=None):
    # assign IDs like FlyTracker DFs

    # convert units from pix to mm
    # mm_per_pix = 3 / trk_['body_length'].mean()
    df = df.rename(columns={"dist_to_other": "dist_to_other_pix"})

    if mm_per_pix is None:
        arena_size_mm = 38 - 4  # arena size minus 2 body lengths
        max_dist_found = df["dist_to_other_pix"].max()
        mm_per_pix = arena_size_mm / max_dist_found

    # convert units to mm/s and mm (like FlyTracker)
    df["vel"] = df["lin_speed"] * mm_per_pix
    df["dist_to_other"] = df["dist_to_other_pix"] * mm_per_pix
    df["pos_x_mm"] = df["centroid_x"] * mm_per_pix
    df["pos_y_mm"] = df["centroid_y"] * mm_per_pix

    df["mm_to_pix"] = mm_per_pix

    # % rename columns to get RELATIVE pos info
    df = df.rename(
        columns={
            "centroid_x": "pos_x",
            "centroid_y": "pos_y",
            "heading": "ori",  # should rename this
            "body_length": "major_axis_len",
            "inter_wing_dist": "minor_axis_len",
            "time": "sec",
        }
    )
    return df
