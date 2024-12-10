
def do_transformations_on_df(trk_, frame_width, frame_height, 
                             feat_=None,
                             cop_ix=None, flyid1=0, flyid2=1):
    if feat_ is None:
        assert 'dist_to_other' in trk_.columns, "No feat df provided. Need dist_to_other."

    # center x- and y-coordinates
    trk_ = util.center_coordinates(trk_, frame_width, frame_height) 

    # separate fly1 and fly2
    fly1 = trk_[trk_['id']==flyid1].copy().reset_index(drop=True)
    fly2 = trk_[trk_['id']==flyid2].copy().reset_index(drop=True)

    # FIRST, do fly1: -------------------------------------------------
    # translate coordinates so that focal fly is at origin
    fly1, fly2 = util.translate_coordinates_to_focal_fly(fly1, fly2)

    # rotate coordinates so that fly1 is facing 0 degrees (East)
    # Assumes fly1 ORI goes from 0 to pi CCW, with y-axis NOT-inverted.
    # if using FlyTracker, trk_['ori'] = -1*trk_['ori']
    fly1, fly2 = util.rotate_coordinates_to_focal_fly(fly1, fly2)

    # add polar conversion
    # FLIP y-axis? TODO check this
    polarcoords = util.cart2pol(fly2['rot_x'], fly2['rot_y']) 
    fly1['targ_pos_radius'] = polarcoords[0]
    fly1['targ_pos_theta'] = polarcoords[1]
    fly1['targ_rel_pos_x'] = fly2['rot_x']
    fly1['targ_rel_pos_y'] = fly2['rot_y']

    # NOW, do fly2: ----------------------------------------------------
    # translate coordinates so that focal fly is at origin
    fly2, fly1 = util.translate_coordinates_to_focal_fly(fly2, fly1)

    # rotate coordinates so that fly1 is facing 0 degrees (East)
    # Assumes fly1 ORI goes from 0 to pi CCW, with y-axis NOT-inverted.
    # if using FlyTracker, trk_['ori'] = -1*trk_['ori']
    fly2, fly1 = util.rotate_coordinates_to_focal_fly(fly2, fly1)

    # add polar conversion
    # FLIP y-axis? TODO check this
    polarcoords = util.cart2pol(fly1['rot_x'], fly1['rot_y']) 
    fly2['targ_pos_radius'] = polarcoords[0]
    fly2['targ_pos_theta'] = polarcoords[1]
    fly2['targ_rel_pos_x'] = fly1['rot_x']
    fly2['targ_rel_pos_y'] = fly1['rot_y']

    #% copulation index - TMP: fix this!
    if cop_ix is None or np.isnan(cop_ix):
        cop_ix = len(fly1)
        copulation = False
    else:
        copulation = True
    cop_ix = int(cop_ix)

    #% Get all sizes and aggregate trk df
    fly1 = get_target_sizes_df(fly1, fly2, xvar='pos_x', yvar='pos_y')
    # Repeat for fly2:
    fly2 = get_target_sizes_df(fly2, fly1, xvar='pos_x', yvar='pos_y')

    # recombine trk df
    trk = pd.concat([fly1.iloc[:cop_ix], fly2.iloc[:cop_ix]], axis=0).reset_index(drop=True)#.sort_index()
    trk['copulation'] = copulation

    # Get relative velocity and aggregate feat df
    if feat_ is not None:
        f_list = []
        for fi, df_ in feat_.groupby('id'):
            df_ = get_relative_velocity(df_, win=1, 
                                value_var='dist_to_other', time_var='sec')
            f_list.append(df_.reset_index(drop=True).iloc[:cop_ix])
        feat = pd.concat(f_list, axis=0).reset_index(drop=True) #.sort_index()
        feat['copulation'] = copulation
        print(trk.iloc[-1].name, feat.iloc[-1].name)
        df = pd.concat([trk, 
                feat.drop(columns=[c for c in feat.columns if c in trk.columns])], axis=1)
        assert df.shape[0]==trk.shape[0], "Bad merge: {}, {}".format(feat.shape, trk.shape)
    else:
        f_list = []
        assert 'dist_to_other' in trk.columns, "No feat df provided. Need dist_to_other."
        for fi, df_ in trk.groupby('id'):
            df_ = get_relative_velocity(df_, win=1, 
                                value_var='dist_to_other', time_var='sec')
            f_list.append(df_.reset_index(drop=True).iloc[:cop_ix])
        df = pd.concat(f_list, axis=0).reset_index(drop=True) #.sort_index()

    #acq = os.path.split(acqdir)[-1]

    return df
