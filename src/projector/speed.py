class Speed:
    @staticmethod
    def get_step_indices(
        dotdf, speed_var="lin_speed_filt", t_start=20, increment=40, n_levels=10
    ):
        """
        Fix DLC tracked dot trajectories with diffspeeds2.csv
        Smooths dot positions, finds indices of steps in velocity. Use these indices to divide trajectories into epochs.
        """
        # speed_var = 'lin_speed_filt'
        # t_start = 20
        # n_epochs = 9
        if speed_var not in dotdf.columns:
            dotdf = smooth_speed_steps(dotdf)

        tmpdf = dotdf.copy()  # loc[motion_start_ix:].copy()
        step_dict = {}
        for i in range(n_levels):
            t_stop = t_start + increment
            curr_chunk = (
                tmpdf[(tmpdf["time"] >= t_start) & (tmpdf["time"] <= t_stop)]
                .copy()
                .interpolate()
            )
            # if i==(n_levels-1):
            find_stepup = i < (n_levels - 1)
            # check in case speed does not actually drop at end:
            if i == (n_levels - 1) and tmpdf.iloc[-20:][speed_var].mean() < 5:
                find_stepup = False
            else:
                find_stepup = True
            tmp_step_ix = get_step_shift_index(
                np.array(curr_chunk[speed_var].values), find_stepup=find_stepup
            )
            step_ix = curr_chunk.iloc[tmp_step_ix].name
            step_dict.update({i: step_ix})
            t_start = t_stop
        return step_dict

    @staticmethod
    def add_speed_epoch(dotdf, step_dict):
        """
        Use step indices found with get_step_indices() to split speed-varying trajectory df into epochs
        """
        last_ix = step_dict[0]
        dotdf.loc[:last_ix, "epoch"] = 0
        step_dict_values = list(step_dict.values())
        for i, v in enumerate(step_dict_values):
            if v == step_dict_values[-1]:
                dotdf.loc[last_ix:, "epoch"] = i + 1
                # flydf.loc[last_ix:, 'epoch'] = i+1
            else:
                next_ix = step_dict_values[i + 1]
                dotdf.loc[last_ix:next_ix, "epoch"] = i + 1
                # flydf.loc[last_ix:next_ix, 'epoch'] = i+1
            last_ix = next_ix
        return dotdf

    @staticmethod
    def add_speed_epochs(dotdf, flydf, acq, filter=True):
        dotdf = smooth_speed_steps(dotdf)
        # get epochs
        if (
            acq
            in "20240214-1045_f1_Dele-wt_5do_sh_prj10_sz12x12_2024-02-14-104540-0000"
        ):
            n_levels = 8
        elif acq in "20240215-1722_fly1_Dmel_sP1-ChR_3do_sh_6x6_2024-02-15-172443-0000":
            n_levels = 9
        else:
            n_levels = 10
        step_dict = get_step_indices(
            dotdf,
            speed_var="lin_speed_filt",
            t_start=20,
            increment=40,
            n_levels=n_levels,
        )

        dotdf = add_speed_epoch(dotdf, step_dict)
        flydf = add_speed_epoch(flydf, step_dict)
        dotdf["acquisition"] = acq
        flydf["acquisition"] = acq
        if filter:
            return dotdf[dotdf["epoch"] < 10], flydf[flydf["epoch"] < 10]
        else:
            return dotdf, flydf

    def smooth_speed_steps(dotdf, win=13, cop_ix=None):
        if cop_ix is None:
            cop_ix = len(dotdf)
        smoothed_x = lpfilter(dotdf['centroid_x'], win)
        smoothed_y = lpfilter(dotdf['centroid_y'], win)

        dot_ctr_sm = np.dstack([smoothed_x, smoothed_y]).squeeze()
        # dotdf['lin_speed_filt'] = smoothed_speed
        dotdf['lin_speed_filt'] = np.concatenate(
                                (np.zeros(1), 
                                np.sqrt(np.sum(np.square(np.diff(dot_ctr_sm[:cop_ix, ], axis=0)), 
                                axis=1)))).round(2)
        dotdf['centroid_x_filt'] = smoothed_x
        dotdf['centroid_y_filt'] = smoothed_y

        return dotdf