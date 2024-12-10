

import numpy as np
import pandas as pd

from coordinate_transform import CoordinateTransform


def get_heading_vector(f_ori, f_len):
    # female ori and length
    # get female heading as a vector
    th = (np.pi - f_ori) % np.pi
    y_ = f_len / 2 * np.sin(th)
    x_ = f_len / 2 * np.cos(th)
    return np.array([x_, y_])


def calculate_female_size_deg(xi, yi, f_ori, f_len):
    """
    Calculate size of target (defined by f_ori, f_len) in degrees of visual angle.

    Finds vector orthogonal to focal and target flies. Calculates heading of target using f_ori and f_len. Then, projects target heading onto orthogonal vector.

    Size is calculated as 2*arctan(fem_sz/(2*dist_to_other)).
    Note: make sure units are consistent (e.g., pixels for f_len, xi, yi).

    Arguments:
        xi -- x coordinate of vector between focal and target flies
        yi -- y coordinate of vector between focal and target flies
        f_ori -- orientation of target fly (from FlyTracker, -180 to 180; 0 faces east, positive is CCW)
        f_len -- length of target fly (from FlyTracker, in pixels)

    Returns:
        Returns calculated size in deg for provided inputs.
    """
    # get vector between male and female
    # xi = fly2.loc[ix][xvar] - fly1.loc[ix][xvar]
    # yi = fly2.loc[ix][yvar] - fly1.loc[ix][yvar]

    # get vector orthogonal to male's vector to female
    ortho_ = [yi, -xi]  # ortho_hat = ortho_ / np.linalg.norm(ortho_)

    # project female heading vec onto orthog. vec
    # f_ori = fly2.loc[ix]['ori']
    # f_len = fly2.loc[ix]['major_axis_len']
    fem_vec = get_heading_vector(f_ori, f_len)  # np.array([x_, y_])
    # female_hat = fem_vec / np.linalg.norm(fem_vec)
    vproj_ = proj_a_onto_b(fem_vec, ortho_)

    # calculate detg vis angle
    fem_sz = np.sqrt(vproj_[0] ** 2 + vproj_[1] ** 2) * 2
    dist_to_other = np.sqrt(xi**2 + yi**2)
    fem_sz_deg = 2 * np.arctan(fem_sz / (2 * dist_to_other))

    return fem_sz_deg


def get_relative_velocity(df_, win=1, value_var="dist_to_other", time_var="sec"):
    """
        Calculate relative velocity between two flies, relative metric (one fly).
        If using FlyTracker feat.mat, dist_to_other is in mm, and time is sec.

        Arguments:
            fly1 -- feat_ dataframe for fly1

        Keyword Argumentsprint(figdir, figname)

    :
            value_var -- relative dist variable to calculate position diff (default: {'dist_to_other'})
            time_var -- time variable to calculate time diff (default: {'sec'})
    """
    # fill nan of 1st value with 0
    df_["{}_diff".format(value_var)] = (
        df_[value_var].interpolate().diff().fillna(0)
    )  # if dist incr, will be pos, if distance decr, will be neg
    df_["{}_diff".format(time_var)] = (
        df_[time_var].interpolate().diff().fillna(0)
    )  # if dist incr, will be pos, if distance decr, will be neg

    df_["rel_vel"] = df_["{}_diff".format(value_var)] / (
        win * df_["{}_diff".format(time_var)].mean()
    )
    df_["rel_vel_abs"] = df_["{}_diff".format(value_var)].abs() / (
        win * df_["{}_diff".format(time_var)].mean()
    )

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
        fem_sz_deg_maj = calculate_female_size_deg(xi, yi, f_ori, f_len_maj)
        fem_sz_deg_min = calculate_female_size_deg(xi, yi, f_ori, f_len_min)
        fem_sz_deg = np.max([fem_sz_deg_maj, fem_sz_deg_min])
        fem_sizes.append(fem_sz_deg)

    # fly2['targ_ang_size'] = fem_sizes
    # fly2['targ_ang_size_deg'] = np.rad2deg(fly2['targ_ang_size'])
    # copy same info for f1
    fly1["targ_ang_size"] = fem_sizes
    fly1["targ_ang_size_deg"] = np.rad2deg(fly1["targ_ang_size"])

    return fly1  # , fly2

def rotate_point(p, angle, origin=(0, 0)):  # degrees=0):
    """
    Calculate rotation matrix R and perform R.dot(p.T) to get rotated coords.

    Returns:
        _description_
    """
    # angle = np.deg2rad(degrees)
    R = np.squeeze(
        np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    )
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def translate_coordinates_to_focal_fly(fly1, fly2):
    """
    Translate coords so that x, y of focal fly (fly1) is (0, 0).
    Assumes coordsinates have been centered already (ctr_x, ctr_y).

    Arguments:
        fly1 -- _description_
        fly2 -- _description_

    Returns:
        fly1, fly2 with columns 'trans_x' and 'trans_y'. fly1 is 0.
    """
    assert "ctr_x" in fly1.columns, "No 'ctr_x' column in fly1 df"
    fly1["trans_x"] = fly1["ctr_x"] - fly1["ctr_x"]
    fly1["trans_y"] = fly1["ctr_y"] - fly1["ctr_y"]
    fly2["trans_x"] = fly2["ctr_x"] - fly1["ctr_x"]
    fly2["trans_y"] = fly2["ctr_y"] - fly1["ctr_y"]

    return fly1, fly2


def rotate_coordinates_to_focal_fly(fly1, fly2):
    """
    Apply rotation to fly2 so that fly1 is at 0 heading.
    Assumes 'ori' is a column in fly1 and fly2. (from FlyTracker)

    Arguments:
        fly1 -- _description_
        fly2 -- _description_

    Returns:
        fly1, fly2 with columns 'rot_x' and 'rot_y'. fly1 is 0.

    """
    assert "trans_x" in fly1.columns, "trans_x not found in fly1 DF"
    ori_vals = fly1["ori"].values  # -pi to pi
    ori = -1 * ori_vals + np.deg2rad(0)  # ori - ori is 0 heading

    fly2[["rot_x", "rot_y"]] = np.nan
    fly1[["rot_x", "rot_y"]] = 0

    fly2[["rot_x", "rot_y"]] = [
        rotate_point(pt, ang)
        for pt, ang in zip(fly2[["trans_x", "trans_y"]].values, ori)
    ]

    fly2["rot_ori"] = fly2["ori"] + ori
    fly1["rot_ori"] = fly1["ori"] + ori  # should be 0

    return fly1, fly2


def do_transformations_on_df(
    trk_, frame_width, frame_height, feat_=None, cop_ix=None, flyid1=0, flyid2=1
):
    if feat_ is None:
        assert (
            "dist_to_other" in trk_.columns
        ), "No feat df provided. Need dist_to_other."

    # center x- and y-coordinates
    trk_ = center_coordinates(trk_, frame_width, frame_height)

    # separate fly1 and fly2
    fly1 = trk_[trk_["id"] == flyid1].copy().reset_index(drop=True)
    fly2 = trk_[trk_["id"] == flyid2].copy().reset_index(drop=True)

    # FIRST, do fly1: -------------------------------------------------
    # translate coordinates so that focal fly is at origin
    fly1, fly2 = translate_coordinates_to_focal_fly(fly1, fly2)

    # rotate coordinates so that fly1 is facing 0 degrees (East)
    # Assumes fly1 ORI goes from 0 to pi CCW, with y-axis NOT-inverted.
    # if using FlyTracker, trk_['ori'] = -1*trk_['ori']
    fly1, fly2 = rotate_coordinates_to_focal_fly(fly1, fly2)

    # add polar conversion
    # FLIP y-axis? TODO check this
    polarcoords = CoordinateTransform.cart2pol(fly2["rot_x"], fly2["rot_y"])
    fly1["targ_pos_radius"] = polarcoords[0]
    fly1["targ_pos_theta"] = polarcoords[1]
    fly1["targ_rel_pos_x"] = fly2["rot_x"]
    fly1["targ_rel_pos_y"] = fly2["rot_y"]

    # NOW, do fly2: ----------------------------------------------------
    # translate coordinates so that focal fly is at origin
    fly2, fly1 = translate_coordinates_to_focal_fly(fly2, fly1)

    # rotate coordinates so that fly1 is facing 0 degrees (East)
    # Assumes fly1 ORI goes from 0 to pi CCW, with y-axis NOT-inverted.
    # if using FlyTracker, trk_['ori'] = -1*trk_['ori']
    fly2, fly1 = rotate_coordinates_to_focal_fly(fly2, fly1)

    # add polar conversion
    # FLIP y-axis? TODO check this
    polarcoords = CoordinateTransform.cart2pol(fly1["rot_x"], fly1["rot_y"])
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
