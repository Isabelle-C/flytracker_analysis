#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:47:00 2020
@author: julianarhee
@email: juliana.rhee@gmail.com  
"""
#%%
import sys
import os
import glob
import cv2
import numpy as np
import pandas as pd
import pylab as pl  
import seaborn as sns
import utils as util
import matplotlib as mpl
import pickle as pkl
import argparse

# import some custom funcs
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

#%%
def plot_frame_check_affines(ix, fly1, fly2, cap, frame_width=None, frame_height=None):
    '''
    Plot frame and rotations with markers oriented to fly's heading. IX is FRAME NUMBER.

    Arguments:
        ix -- _description_
        fly1 -- _description_
        fly2 -- _description_
        cap -- _description_

    Keyword Arguments:
        frame_width -- _description_ (default: {None})
        frame_height -- _description_ (default: {None})

    Returns:
        _description_
    '''
    if frame_width is None:
        frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    # set fly oris as arrows
    fly_marker = '$\u2192$' # https://en.wikipedia.org/wiki/Template:Unicode_chart_Arrows
    m_ori = np.rad2deg(fly1[fly1['frame']==ix]['rot_ori'])
    f_ori = np.rad2deg(fly2[fly2['frame']==ix]['rot_ori'])
    marker_m = mpl.markers.MarkerStyle(marker=fly_marker)
    marker_m._transform = marker_m.get_transform().rotate_deg(m_ori)
    marker_f = mpl.markers.MarkerStyle(marker=fly_marker)
    marker_f._transform = marker_f.get_transform().rotate_deg(f_ori)
    #print(np.rad2deg(fly1.loc[ix]['ori'])) #m_ori)
    #print(f_ori)

    cap.set(1, ix)
    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

    fig = pl.figure(figsize=(8,4))
    ax = fig.add_subplot(121) # axn = pl.subplots(1, 2)
    ax.imshow(im, cmap='gray')
    ax.set_title("Frame {}".format(ix), fontsize=8, loc='left')
    ax.invert_yaxis()

    #ax = fig.add_subplot(142) 
    ax.plot(fly1[fly1['frame']==ix]['pos_x'], fly1[fly1['frame']==ix]['pos_y'], 'r*')
    ax.plot(fly2[fly2['frame']==ix]['pos_x'], fly2[fly2['frame']==ix]['pos_y'], 'bo')
    ax.set_aspect(1)
    ax.set_xlim(0, frame_width)
    ax.set_ylim(0, frame_height)
    #ax.invert_yaxis()

    ax = fig.add_subplot(122)
    ax.set_title('centered and rotated to focal (*)', fontsize=8, loc='left') 
    # make a markerstyle class instance and modify its transform prop
    ax.plot([0, float(fly1[fly1['frame']==ix]['rot_x'].iloc[0])], 
            [0, float(fly1[fly1['frame']==ix]['rot_y'].iloc[0])], 'r', 
            marker=marker_m, markerfacecolor='r', markersize=10) 
    ax.plot([fly2[fly2['frame']==ix]['rot_x']], [fly2[fly2['frame']==ix]['rot_y']], 'b',
            marker=marker_f, markerfacecolor='b', markersize=10) 
    ax.set_aspect(1)
    ax.set_xlim(0-frame_width, frame_width)
    ax.set_ylim(0-frame_height, frame_height)
    #ax.invert_yaxis()

    return fig

def check_rotation_transform(ix, trk_, cap, id_colors=['r', 'b']):
    '''
    Note that ix should be frame.

    Arguments:
        ix -- _description_
        trk_ -- _description_
        cap -- _description_

    Keyword Arguments:
        id_colors -- _description_ (default: {['r', 'b']})

    Returns:
        _description_
    '''
    # get image
    cap.set(1, ix)
    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

    fig = pl.figure(figsize=(12,5)) #pl.subplots(1, 2)
    # plot frame
    ax = fig.add_subplot(131)
    ax.imshow(im, cmap='gray')
    ax.invert_yaxis()
    # plot positions
    for i, d_ in trk_.groupby('id'):
        print('pos:', i, d_[d_['frame']==ix]['pos_x'], d_[d_['frame']==ix]['pos_y'])
        ax.plot(d_[d_['frame']==ix]['pos_x'], d_[d_['frame']==ix]['pos_y'], 
                marker='o', color=id_colors[i], markersize=3)

    fly1 = trk_[trk_['id']==0].copy().reset_index(drop=True)
    fly2 = trk_[trk_['id']==1].copy().reset_index(drop=True)
    # plot rotated positions, male faces EAST on cartesian
    ax = fig.add_subplot(132) #, projection='polar')
    for i, d_ in enumerate([fly1, fly2]):
        #print('rot:', i, d_.iloc[ix]['rot_x'], d_.iloc[ix]['rot_y'])
        pt = np.squeeze(np.array(d_[d_['frame']==ix][['trans_x', 'trans_y']].values))

        print(pt.shape)
        #ang = rotation_angs[ix]        
        #rx, ry = rotate([0, 0], pt, ang)
        ang = -1*fly1[fly1['frame']==ix]['ori'] 
        rotmat = np.array([[np.cos(ang), -np.sin(ang)],
                            [np.sin(ang), np.cos(ang)]])
        #rx, ry = (rotmat @ pt.T).T
        rx, ry = util.rotate_point(pt, ang) #[0, 0], pt, ang)
        print('rot:', i, rx, ry)
        ax.plot(rx, ry,marker='o', color=id_colors[i], markersize=3)
    ax.set_aspect(1)
    ax.set_title('rot')

    # POLAR
    ax = fig.add_subplot(133, projection='polar')
    for i, d_ in enumerate([fly1, fly2]):
        if i==0:
            ax.plot(0, 0, 'r*')
        #ang = fly1.iloc[ix]['ori'] #* -1
        pt = [d_[d_['frame']==ix]['trans_x'], 
              d_[d_['frame']==ix]['trans_y']]
        #ang = rotation_angs[ix]  
        #rx, ry = rotate((0,0), pt, ang)      
        #rx, ry = rotate2(pt, ang) #[0, 0], pt, ang)
        rad, th = util.cart2pol(rx, ry)
        ax.plot(th, rad, marker='o', color=id_colors[i], markersize=3)
    ax.set_aspect(1)
    ax.set_title('polar')
    #ax.set_theta_direction(-1)
    #print(ang)
    fig.suptitle('{}, ang={:.2f}'.format(ix, np.rad2deg(float(ang))))

    return fig




def plot_frame_target_projection(ix, fly1, fly2, cap, xvar='pos_x', yvar='pos_y'):
    cap.set(1, ix)
    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

    # get vector between male and female
    xi = fly2.loc[ix][xvar] - fly1.loc[ix][xvar] 
    yi = fly2.loc[ix][yvar] - fly1.loc[ix][yvar]

    # get vector orthogonal to male's vector to female
    ortho_ = [yi, -xi] #ortho_hat = ortho_ / np.linalg.norm(ortho_)

    # project female heading vec onto orthog. vec
    f_ori = fly2.loc[ix]['ori']
    f_len = fly2.loc[ix]['major_axis_len']
    fem_vec = util.get_heading_vector(f_ori, f_len) #np.array([x_, y_])
    #female_hat = fem_vec / np.linalg.norm(fem_vec)
    vproj_ = util.proj_a_onto_b(fem_vec, ortho_)

    # plot
    fig, axn =pl.subplots(1, 2, figsize=(8, 4))
    ax = axn[0]
    ax.imshow(im, cmap='gray')
    ax.set_title("Frame {}".format(ix)) 
    # plot original positions
    x0, y0 = fly1.loc[ix][[xvar, yvar]]
    x1, y1 = fly2.loc[ix][[xvar, yvar]]
    # plot vector between
    ax.plot([x0, x0+xi], [y0, y0+yi])
    # plot orthogonal
    ax.plot([x1, x1+ortho_[0]], [y1, y1+ortho_[1]], 'orange')
    ax.set_aspect(1)
    # plot female heading
    ax.plot([x1, x1+fem_vec[0]], [y1, y1+fem_vec[1]], 'magenta')
    # plot proj
    ax.plot([x1, x1+vproj_[0]], [y1, y1+vproj_[1]], 'cyan')
    ax.plot([x1, x1-vproj_[0]], [y1, y1-vproj_[1]], 'cyan')

    # plot the vectors only
    ax=axn[1]
    ax.plot([0, xi], [0, yi], 'b')
    ax.plot([0, fem_vec[0]], [0, fem_vec[1]], 'magenta')
    ax.plot([0, ortho_[0]], [0, ortho_[1]], 'orange')
    ax.invert_yaxis()
    ax.plot([0, vproj_[0]], [0, vproj_[1]], 'cyan')
    #ax.plot([0, proj_[0]], [0, proj_[1]], 'magenta')
    ax.set_aspect(1)

    # check
    #diff = (vproj_ - np.array([fem_vec[0], fem_vec[1]]))
    #print(np.dot(diff, ortho_)) #(diff[0] * ortho_hat[0] ) + (diff[1] * ortho_hat[1])

    #ppm = calib_['PPM']
    #print(np.sqrt(xi**2 + yi**2)/ppm)
    #print(feat_.loc[ix]['dist_to_other']) 

    # len of projected female
    fem_sz = np.sqrt(vproj_[0]**2 + vproj_[1]**2) * 2
    dist_to_other = np.sqrt(xi**2 + yi**2)
    fem_sz_deg = 2*np.arctan(fem_sz/(2*dist_to_other))
    ax.set_title('Targ is {:.2f} deg. vis. ang'.format(np.rad2deg(fem_sz_deg)))

    return fig

#%%
def get_copulation_ix(acq):
    cop_ele = {
        '20231213-1103_fly1_eleWT_5do_sh_eleWT_5do_gh': 52267,
        '20231213-1154_fly3_eleWT_6do_sh_eleWT_5do_gh': 17243,
        '20231214-1051_fly2_eleWT_3do_sh_eleWT_3do_gh': 61512,
        '20231223-1117_fly1_eleWT_5do_sh_eleWT_5do_gh': 55582,
        '20231226-1137_fly2_eleWT_4do_sh_eleWT_4do_gh': 13740,
        '20240105-1007_fly1_eleWT_3do_sh_eleWT_3do_gh': 5051, 
        '20240109-1039_fly1_eleWT_4do_sh_eleWT_4do_gh': 177100,
        '20240322-1001_f1_eleWT_4do_gh': 5474,
        '20240322-1045_f4_eleWT_4do_gh': 54842,
        '20240322-1143_f6_eelWT_4do_gh': 1356,
        '20240322-1146_f7_eleWT_4do_gh': 12190,
        '20240322-1152_f8_eleWT_4do_gh': 7620,
        '20240322-1156_f9_eleWT_4do_gh': 92680
    }

    local_dir = '/Users/julianarhee/Documents/rutalab/projects/courtship/38mm-dyad'
    fname = 'courtship-free-behavior (Responses) - Form Responses 1.csv'
    meta_fpath = os.path.join(local_dir, fname)
    meta = pd.read_csv(meta_fpath)

    if 'ele' in acq:
        match_ = [v for v in cop_ele.keys() if v.startswith(acq)]
        if len(match_)==0:
            print("No match: {}".format(acq))
            cop_ix = np.nan
        else:
            cop_ix = cop_ele[match_[0]]
    else:
        match_ = [v for v in meta['logfile'] if v.startswith(acq)]
        if len(match_)==0: #,  "{} not found".format(acq)
            print("NO match: {}".format(acq))
            cop_ix = np.nan
        else:
            #cop_ix = meta[meta['logfile']==match_[0]]['FlyTracker: copulation index']
            cop_ix = float(meta.loc[meta['logfile']==match_[0], 'FlyTracker: copulation index'])

    return cop_ix

def get_video_cap(acqdir, ftname=None, movie_fmt='avi'):
    if ftname is None:
        vids = util.get_videos(acqdir, vid_type=movie_fmt)
    else:
        vids = util.get_video_by_ft_name(viddir, ftname, vid_type=movie_fmt)

    alt_movie_fmt = 'mp4' if movie_fmt=='avi' else 'avi'

    # try alt movie fmt
    try:
        assert len(vids)>0, "Found no video in directory: {}".format(vids)
        vids = [vids[-1]]
    except AssertionError as e:
        if ftname is None:
            vids = util.get_videos(acqdir, vid_type=alt_movie_fmt)
        else:
            vids = util.get_video_by_ft_name(acqdir, ftname, vid_type=alt_movie_fmt)
        #vids = util.get_videos(acqdir, vid_type=alt_movie_fmt)
        assert len(vids)==1, "Found more than one video in directory: {}".format(vids)  

    vidpath = vids[0]
    print(vidpath)

    cap = cv2.VideoCapture(vidpath)
    return cap


def get_metrics_relative_to_focal_fly(acqdir, mov_is_upstream=False, fps=60, cop_ix=None,
                                      movie_fmt='avi', flyid1=0, flyid2=1,
                                      plot_checks=False,
                                      savedir=None):
    '''
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
    '''
    # check output dir
    if savedir is None:
        print("No save directory provided. Saving to acquisition directory.")
        savedir = acqdir
    # load flyracker data
    if mov_is_upstream:
        subfolder = ''
    else:
        subfolder = '*'
    calib_, trk_, feat_ = util.load_flytracker_data(acqdir, fps=fps, 
                                                    calib_is_upstream=mov_is_upstream,
                                                    subfolder=subfolder,
                                                    filter_ori=True)

    # get video file for plotting/sanity checks
#    vids = util.get_videos(acqdir, vid_type=movie_fmt)
#    alt_movie_fmt = 'mp4' if movie_fmt=='avi' else 'avi'
#    try:
#        assert len(vids)>0, "Found no video in directory: {}".format(vids)
#        vids = [vids[-1]]
#    except AssertionError as e:
#        vids = util.get_videos(acqdir, vid_type=alt_movie_fmt)
#        assert len(vids)==1, "Found more than one video in directory: {}".format(vids)  
#
#    vidpath = vids[0]
#    cap = cv2.VideoCapture(vidpath)
    if mov_is_upstream:
        parentdir, ftname = os.path.split(acqdir)
        viddir = os.path.split(parentdir)[0]
        cap = get_video_cap(viddir, ftname=ftname, movie_fmt=movie_fmt) 
    else:
        cap = get_video_cap(acqdir, movie_fmt=movie_fmt)

    # N frames should equal size of DCL df
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    print(frame_width, frame_height) # array columns x array rows
    # switch ORI
    trk_['ori'] = -1*trk_['ori'] # flip for FT to match DLC and plot with 0, 0 at bottom left
    df_ = do_transformations_on_df(trk_, frame_width, frame_height, 
                                   feat_=feat_, cop_ix=cop_ix,
                                   flyid1=0, flyid2=1)

    # save
    #% save
    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        acq = os.path.split(acqdir)[-1]
        df_fpath = os.path.join(savedir, '{}_df.pkl'.format(acq))
        with open(df_fpath, 'wb') as f: 
            pkl.dump(df_, f)
        print('Saved: {}'.format(df_fpath))

    #% plot - sanity checks
    if plot_checks:
        fly1 = df_[df_['id']==flyid1]
        fly2 = df_[df_['id']==flyid2]
        # check affine transformations for centering and rotating male
        ix = 6500 #5000 #2500 #590
        fig = plot_frame_check_affines(ix, fly1, fly2, cap, frame_width, frame_height)
        fig.text(0.1, 0.95, os.path.split(acqdir)[-1], fontsize=4)

        # check projections for calculating size based on distance and angle
        ix = 100 #213527 #5000 #2500 #590
        for ix in [100, 3000, 5000]:
            fig = plot_frame_target_projection(ix, fly1, fly2, cap, 
                                            xvar='pos_x', yvar='pos_y')
            fig.text(0.1, 0.95, os.path.split(acqdir)[-1], fontsize=4)

    return df_



def load_processed_data(acqdir, savedir=None, load=True):
    '''
    Load processed feat and trk dataframes (pkl files) from savedir.

    Arguments:
        acq_dir -- _description_

    Keyword Arguments:
        savedir -- _description_ (default: {None})

    Returns:
        _description_
    '''
    feat_=None; trk=None;
    if savedir is None:
        savedir = acqdir

    acq = os.path.split(acqdir)[-1]
    df_fpath = os.path.join(savedir, '{}_df.pkl'.format(acq))
    #feat_fpath = os.path.join(savedir, '{}_feat.pkl'.format(acq))
    #trk_fpath = os.path.join(savedir, '{}_trk.pkl'.format(acq))

    if load:
        with open(df_fpath, 'rb') as f:
            df_ = pkl.load(f) 
        print('Loaded: {}'.format(df_fpath))
#        with open(feat_fpath, 'rb') as f:
#            feat_ = pkl.load(f) 
#        print('Loaded: {}'.format(feat_fpath))
#
#        with open(trk_fpath, 'rb') as f:
#            trk = pkl.load(f)
#        print('Loaded: {}'.format(trk_fpath))

    else:
        df_ = os.path.exists(df_fpath)
#        feat_ = os.path.exists(feat_fpath)
#        trk = os.path.exists(trk_fpath)

    return df_ #feat_, trk
    


#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process FlyTracker data for relative metrics.')
    parser.add_argument('--savedir', type=str, help='Directory to save processed data.')    
    parser.add_argument('--movie_fmt', type=str, default='avi', help='Movie format (default: avi).')
    parser.add_argument('--flyid1', type=int, default=0, help='ID of focal fly (default: 0).')  
    parser.add_argument('--flyid2', type=int, default=1, help='ID of target fly (default: 1).') 
    parser.add_argument('--plot_checks', type=bool, default=False, help='Plot checks (default: False).')    
    parser.add_argument('--viddir', type=str, default='/Volumes/Julie/38mm_dyad/courtship-videos/38mm_dyad', help='Root directory of videos (default: /Volumes/Julie/38mm_dyad/courtship-videos/38mm_dyad).')   
    parser.add_argument('--new', type=bool, default=False, help='Create new processed data (default: False).')
    parser.add_argument('--subdir', type=str, default=None, help='subdir of tracked folders, e.g., fly-tracker (default: None).')
    
    args = parser.parse_args()
    # 
    viddir = args.viddir 
    savedir = args.savedir
    movie_fmt = args.movie_fmt
    flyid1 = args.flyid1
    flyid2 = args.flyid2
    subdir = args.subdir
    create_new = args.new

#%% #Hardcoded parameter values for running in interactive mode
    interactive = False

    #viddir = '/Volumes/Giacomo/free_behavior_data'
    #savedir = '/Volumes/Julie/free-behavior-analysis/FlyTracker/38mm_dyad/processed'

    #viddir = '/Volumes/Giacomo/JAABA_classifiers/projector/changing_dot_size_speed'
    #savedir = '/Volumes/Julie/2d-projector-analysis/FlyTracker/processed_mats'

    if interactive:
        viddir = '/Volumes/Juliana/2d-projector'
        savedir = '/Volumes/Juliana/2d-projector-analysis/FlyTracker/processed_mats'
        subdir = 'fly-tracker'
        flyid1 = 0
        flyid2 = 1
        movie_fmt = '.avi'
        create_new=True

#%%
    if subdir is not None:
        found_mats = glob.glob(os.path.join(viddir,  '20*', '*{}*'.format(subdir), '*', '*feat.mat'))
    else:
        found_mats = glob.glob(os.path.join(viddir,  '20*', '*', '*feat.mat'))
    print('Found {} processed videos.'.format(len(found_mats)))

    #%% For each found acquisition (video), calculate relative metrics
    for fp in found_mats:
        if subdir is not None:
            # FT output dir is parent dir
            ftdir = os.path.split(fp)[0]
            # video dir is upstream
            viddir = os.path.split(ftdir)[0]
            # acq name is FT name
            acq = os.path.split(ftdir)[-1]
        else:
            acq = os.path.split(os.path.split(fp.split(viddir+'/')[-1])[0])[0]

        if "BADTRACKING" in fp: # Giacomo added this to skip bad tracking
            continue

        acqdir = os.path.join(viddir, acq)
        print(acq)
        # Try loading data
        if not create_new:
            df_ = load_processed_data(acqdir, load=False, savedir=savedir)
            if df_ is False: 
                create_new=True #assert ft is True, "No feat df found, creating now."
        
        # Create a new processed_mat file by calculating relative metrics
        if create_new:
            if '2d-projector' not in viddir:
                cop_ix = get_copulation_ix(acq)
            else:
                cop_ix = None
            df_ = get_metrics_relative_to_focal_fly(acqdir,
                                        savedir=savedir,
                                        movie_fmt=movie_fmt, 
                                        mov_is_upstream=subdir is not None,
                                        flyid1=flyid1, flyid2=flyid2,
                                        plot_checks=False)
