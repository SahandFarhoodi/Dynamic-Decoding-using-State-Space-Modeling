"""------------------------------- README ------------------------------- """
"""
Subject: Codes used to generate results presented in paper "Estimating Fluctuations in Neural Representations of 
Uncertain Environments" which is submitted to NeurIPS 2020. 

Author: Anonymous
 
How to use:
    - Dependencies: dependencies through lines 17-26. All of the libraries are very well-known and open source.
    - In order to generate the figures in the paper, first everything up to line 1983 must be executed. After that,
      for each of the following figures execute the corresponding lines:
      
            Paper, Figure 2A                                    Lines  1985-1987
            Paper, Figure 2B                                    Lines  1989-1991
            Paper, Figure 2C                                    Lines  1993-1997
            Paper, Figure 3                                     Lines  1999-2002
            Paper, Figure 4A                                    Lines  2004-2010
            Paper, Figure 4B                                    Lines  2012-2016
            Paper, Figure 4C                                    Lines  2018-2029
            Paper, Figure 4D                                    Lines  2031-2035
            Supplementary Material, Figure 1                    Lines  2042-2053
       
Note 1: Data used for this analysis is not shared, as it is not permitted by people who collected that.

Note 2: A version of this code that works for a simulated data is going to be developed, and it will be shared along 
with the simulated data publicly later.
"""


import os
import numpy as np
import matplotlib    
import matplotlib.pyplot as plt
from collections import Counter
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
import scipy
from sklearn.cluster import KMeans

if 'Apple_PubSub_Socket_Render' in os.environ:
    # on OSX
    matplotlib.use('TkAgg')
elif 'JOB_ID' in os.environ:
    # running in an SCC batch job, no interactive graphics
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
    print('Using TkAgg.')


def find_nearest(arr1, arr2):
    # For each element x of arr2, find the index of the closest element of arr1 to that.

    ans = []
    for x in arr2:
        y = np.argmin(np.abs(arr1 - x))
        ans.append(y)
    ans = np.array(ans)
    return ans


def compute_hist_cov (Y, hist_wind):
    # Compute the covaraite matrix that consists of the history of the response vector (each column one lag)
    # This function is used for fitting models with history-dependent component
    # Y: the response vector
    # hist_wind: size of the history window

    Y_hist = np.zeros(shape=[Y.shape[0] - hist_wind, hist_wind])
    for h in range(hist_wind):
        Y_hist[:, h] = Y[h: h + Y.shape[0] - hist_wind]
    return Y_hist


def morph_trials(morph_lvl):
    # Return trials for the given morph level
    # morph_lvl: morph level

    ind = np.where(VRData[:, 1] == morph_lvl)[0]
    lst = VRData[ind, 20].astype(int)
    cnt = Counter(lst)
    out = list(set([x for x in lst if cnt[x] > 4]))  # only keep complete trials, i.e. with at least 4 data points
    out.sort()
    out = np.array(out)
    return out


def compute_activity_rate(exp_id, morphs, morph_lvl, breaks):
    # For each cell, for each trial with given morph level, return the activity rate as a function of the position
    # morph_lvl: the fixed morph level
    # morphs: all trials with the fixed morph level
    # breaks: number of bins used to discretize the 1-dimensional position
    # exp_id: experiment id

    step = (max_pos - min_pos) / breaks
    activity_rates = np.zeros(shape=[ncells, breaks, ntrials])
    activity_count = np.zeros(shape=[ncells, breaks, ntrials])
    last_moment_activity_rates = np.zeros(shape=[ncells])
    last_moment_activity_count = np.zeros(shape=[ncells])
    for tr in morphs:
        print(tr)
        for i in range(breaks):
            ind = np.where((VRData[:, 3] >= min_pos + i*step) & (VRData[:, 3] < min_pos + (i+1)*step) & (
                            VRData[:, 20] == tr))[0]
            # if there is no data point for a cell and a given bin, use the activity rate for the previous bin
            if len(ind) == 0:
                if i == 0:
                    activity_rates[:, i, tr] = np.zeros(shape=[ncells])
                    activity_count[:, i, tr] = np.zeros(shape=[ncells])
                else:
                    activity_rates[:, i, tr] = last_moment_activity_rates
                    activity_count[:, i, tr] = last_moment_activity_count

            else:
                activity_rates[:, i, tr] = np.mean(F[:, ind], axis=1)
                activity_count[:, i, tr] = len(ind)
                last_moment_activity_rates = activity_rates[:, i, tr]
                last_moment_activity_count = activity_count[:, i, tr]
    np.save(os.getcwd() + '/Data/activity_rates_exp_' + str(exp_id) + '_morph_' + str(morph_lvl) + '.npy',
            activity_rates)
    np.save(os.getcwd() + '/Data/activity_count_exp_' + str(exp_id) + '_morph_' + str(morph_lvl) + '.npy',
            activity_count)


def compute_spline_mat(X):
    # Compute the cubic spline matrix for a 1-dimensional position vector X with regularly distributed knots.
    # X: 1-dimensional input

    min_val = np.min(X)
    max_val = np.max(X)
    knots = np.array([min_val - 10] + list(np.arange(min_val, max_val+50, 50)) + [max_val + 10])
    par = 0.5  # tension parameter
    out = np.zeros(shape=[X.shape[0], knots.shape[0]])
    for i in range(X.shape[0]):
        x = X[i]
        nearest_knot_ind = np.argmax(knots[knots <= x])
        nearest_knot = knots[nearest_knot_ind]
        next1 = knots[nearest_knot_ind+1]
        next2 = knots[nearest_knot_ind+2]
        u = (x - nearest_knot)/(next1 - nearest_knot)
        l = (next2 - next1)/(next1 - nearest_knot)
        A = np.array([np.power(u, 3), np.power(u, 2), u, 1])
        B = np.array([[-par, 2-par/l, par-2, par/l], [2*par, par/l-3, 3-2*par, -par/l], [-par, 0, par, 0],
                      [0, 1, 0, 0]])
        r = A.dot(B)
        r = np.reshape(r, newshape=(1, r.shape[0]))
        out[i, nearest_knot_ind-1: nearest_knot_ind+3] = r
    return out


def compute_spline_Normal_loglike(exp_id, morph_lvl, morphs):
    # For each cell, for each trial with a fixed morph level, fit a spline-Normal model and return the computed
    # log-likelihoods.
    # model: Y ~ Normal(mean = beta_0 + Sum beta_i g_i(x)) where g_i(x)'s are spline basis evaluated at position x
    # morph_lvl: fixed morph level
    # morphs: trials with given morph level
    # exp_id: experiment id

    breaks = 400
    ind = np.where(np.isin(VRData[:, 20], morphs))[0]
    X = VRData[ind, 3]
    Y = F[:, ind]
    step = (max_pos - min_pos) / breaks
    X_disc = []
    Y_disc = []
    for i in range(breaks):
        ind0 = np.where((X >= min_pos + i * step) & (X < min_pos + (i + 1) * step))[0]
        if i == 0:
            X_disc.append(min_pos)
        elif i == breaks - 1:
            X_disc.append(max_pos)
        else:
            X_disc.append(min_pos + (i + 1 / 2) * step)
        if ind0.shape[0] == 0:
            Y_disc.append(Y_disc[-1])
        if ind0.shape[0] > 0:
            Y_disc.append(np.mean(Y[:, ind0], axis=1))
    X_disc = np.array(X_disc)
    Y_disc = np.array(Y_disc)
    spline_mat = compute_spline_mat(X_disc)

    loglike = []
    for cell_id in range(ncells):
        print(cell_id)
        gauss_ident = sm.GLM(Y_disc[:, cell_id], spline_mat, family=sm.families.Gaussian(sm.families.links.identity()))
        gauss_ident_results = gauss_ident.fit()
        b = np.array(gauss_ident_results.params)
        est = spline_mat.dot(b)
        loglike.append(gauss_ident_results.llf)
    loglike = np.array(loglike)
    np.save(os.getcwd() + '/Data/spline_Normal_loglike_exp_' + str(exp_id) + '_morph_' + str(morph_lvl) + '.npy',
            loglike)


def compute_spline_Normal_dist(exp_id):
    # For each cell, fit two spline_Normal models for two original environments and return the L2 distance between
    # estimated activities obtained from these models.
    # exp_id: experiment id

    # For morph = 0
    breaks = 400
    ind = np.where(np.isin(VRData[:, 20], morph_0_trials))[0]
    X = VRData[ind, 3]
    Y = F[:, ind]
    step = (max_pos - min_pos) / breaks
    X_disc = []
    Y0_disc = []
    for i in range(breaks):
        ind0 = np.where((X >= min_pos + i * step) & (X < min_pos + (i + 1) * step))[0]
        if i == 0:
            X_disc.append(min_pos)
        elif i == breaks - 1:
            X_disc.append(max_pos)
        else:
            X_disc.append(min_pos + (i + 1 / 2) * step)
        if ind0.shape[0] == 0:
            Y0_disc.append(Y0_disc[-1])
        if ind0.shape[0] > 0:
            Y0_disc.append(np.mean(Y[:, ind0], axis=1))
    X_disc = np.array(X_disc)
    Y0_disc = np.array(Y0_disc)
    spline_mat0 = compute_spline_mat(X_disc)

    # For morph = 1
    breaks = 400
    ind = np.where(np.isin(VRData[:, 20], morph_1_trials))[0]
    X = VRData[ind, 3]
    Y = F[:, ind]
    step = (max_pos - min_pos) / breaks
    X_disc = []
    Y1_disc = []
    for i in range(breaks):
        ind0 = np.where((X >= min_pos + i * step) & (X < min_pos + (i + 1) * step))[0]
        if i == 0:
            X_disc.append(min_pos)
        elif i == breaks - 1:
            X_disc.append(max_pos)
        else:
            X_disc.append(min_pos + (i + 1 / 2) * step)
        if ind0.shape[0] == 0:
            Y1_disc.append(Y1_disc[-1])
        if ind0.shape[0] > 0:
            Y1_disc.append(np.mean(Y[:, ind0], axis=1))
    X_disc = np.array(X_disc)
    Y1_disc = np.array(Y1_disc)
    spline_mat1 = compute_spline_mat(X_disc)

    # Fitting the models nd computing the L2 distance
    dis = []
    for cell_id in range(ncells):
        print(cell_id)
        gauss_ident = sm.GLM(Y0_disc[:, cell_id], spline_mat0, family=sm.families.Gaussian(sm.families.links.identity()))
        gauss_ident_results = gauss_ident.fit()
        b = np.array(gauss_ident_results.params)
        est0 = spline_mat0.dot(b)
        gauss_ident = sm.GLM(Y1_disc[:, cell_id], spline_mat1, family=sm.families.Gaussian(sm.families.links.identity()))
        gauss_ident_results = gauss_ident.fit()
        b = np.array(gauss_ident_results.params)
        est1 = spline_mat1.dot(b)
        dis.append(np.mean((est0 - est1)**2))
    dis = np.array(dis)
    np.save(os.getcwd() + '/Data/spline_Normal_dist_exp_' + str(exp_id) + '.npy', dis)


def compute_active_trials(exp_id, morphs, morph_lvl):
    # For each cell_id, for each trial with given morph level, use K-means to divide trials into 2 groups using the
    # average activity or place field location. The resulting groups are called "active" and "inactive". However,
    # the inactive group doesn't necessary mean that the cell has no significant activity.
    # exp_id: experiment id
    # morph_lvl: fixed morph level
    # morphs: trials with the fixed morph level

    active_trials = np.zeros(shape=[ncells, len(morphs)])
    activity_trials = np.zeros(shape=[ncells, len(morphs)])
    thrsh = 12000
    for cell_id in range(ncells):
        print('cell_id = {}'.format(cell_id))
        top_num = 5
        top_act = []
        for tr_id in morphs:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            s = np.flip(np.sort(F[cell_id, ind]))
            top_act.append(np.mean(s[:top_num]))
        top_act = np.array(top_act)
        activity_trials[cell_id, :] = top_act
        top_act_org = np.copy(top_act)
        if np.min(top_act) < thrsh:
            top_act[top_act > thrsh] = thrsh
        top_act = np.reshape(top_act, newshape=[top_act.shape[0], 1])
        dots = np.append(np.zeros(shape=top_act.shape), top_act, axis=1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(dots)
        x = np.argmax(dots[:, 1])
        if kmeans.labels_[x] == 0:
            kmeans.labels_ = 1 - kmeans.labels_
        active_trials[cell_id, :] = kmeans.labels_
    np.save(os.getcwd() + '/Data/active_trials_morph_' + str(morph_lvl) + '_exp_' + str(exp_id) + '.npy',
            active_trials)
    np.save(os.getcwd() + '/Data/top_activity_trials_morph_' + str(morph_lvl) + '_exp_' + str(exp_id) + '.npy',
            activity_trials)


def decode_morphs(exp_id, p, mode, visualize, visualize2, history):
    # For each cell fit 8 spline-Gamma models for morph0/morph1 active/inactive with_history/without_history. In
    # addition, fit 2 Gamma models for all active/inactive trials to use later for hypothesis tests on existence of
    # multiple spatial maps. The without_history models are used only for debugging. For each trial compute its
    # log-likelihood and likelihood based on if its active/inactive indicator for both morph0 and morph1 models.
    # Eventually, use the filter and smoother algorithms to compute decoded probability. This function Returns all 8
    # fitted models along with decoding results.
    # exp_id: experiment id
    # p: probability of jumping form one state to the other one in first-order Markov chain
    # mode: shows if we are using shorted version of data (mode = short) or all data (mode = all)
    # visualize: determines if the function must show the decoding results for each cell and each trial or not.
    # visualize2: determines if the function must show the decoding results as a heatmap (for all trials and each
    # individual cell)
    # history: determines if we want to visualize models with history component or not.

    #  each row = [X_tr, Y_tr, p_morphs_filt, p_morph_smooth, p_morph_likelihood]
    p_morph = np.empty(shape=[ncells, ntrials, 6], dtype=object)
    # used later for performing hypothesis tests for existence of multiple spatial maps
    goodness_of_fit = np.empty(shape=[ncells, ntrials, 2], dtype=object)

    gamma_fit_0_act = np.empty(shape=[ncells, 8], dtype=object)  # morph0, active trials, with history
    gamma_fit_0_inact = np.empty(shape=[ncells, 8], dtype=object)  # morph0, inactive trials, with history
    gamma_fit_1_act = np.empty(shape=[ncells, 8], dtype=object)  # morph1, active trials, with history
    gamma_fit_1_inact = np.empty(shape=[ncells, 8], dtype=object)  # morph1, inactive trials, with history
    gamma_fit_0_act_nohist = np.empty(shape=[ncells, 8], dtype=object)  # morph0, active trials, without history
    gamma_fit_0_inact_nohist = np.empty(shape=[ncells, 8], dtype=object)  # morph0, inactive trials, without history
    gamma_fit_1_act_nohist = np.empty(shape=[ncells, 8], dtype=object)  # morph1, active trials, without history
    gamma_fit_1_inact_nohist = np.empty(shape=[ncells, 8], dtype=object)  # morph1, inactive trials, without history
    gamma_fit_0_all = np.empty(shape=[ncells, 8], dtype=object)  # morph0, all trials, with history
    gamma_fit_1_all = np.empty(shape=[ncells, 8], dtype=object)  # morph1, all trials, with history

    trial_ids = range(ntrials)
    cnt = 1
    for cell_id in cells_under_study:
        print('processing the {}-th cell'.format(cnt))
        cnt += 1
        ind = np.where(active_trials_0[cell_id, :] == 1)[0]
        morph0_act = morph_0_trials[ind]
        ind = np.where(active_trials_0[cell_id, :] == 0)[0]
        morph0_inact = morph_0_trials[ind]

        ind = np.where(active_trials_1[cell_id, :] == 1)[0]
        morph1_act = morph_1_trials[ind]
        ind = np.where(active_trials_1[cell_id, :] == 0)[0]
        morph1_inact = morph_1_trials[ind]

        # For morph 0 active trials
        X = np.array(1)
        Y = np.array(1)
        Y_hist = np.array(1)
        first_one = True
        for tr_id in morph0_act:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            Y_hist_tr = compute_hist_cov(Y_tr, hist_wind)
            X_tr = X_tr[hist_wind:]
            Y_tr = Y_tr[hist_wind:]
            if first_one:
                X = X_tr.copy()
                Y = Y_tr.copy()
                Y_hist = Y_hist_tr
                first_one = False
            else:
                X = np.concatenate((X, X_tr), axis=0)
                Y = np.concatenate((Y, Y_tr), axis=0)
                Y_hist = np.concatenate((Y_hist, Y_hist_tr), axis=0)
        X0_act = X.copy()
        Y0_act = Y.copy()
        Y0hist_act= Y_hist.copy()

        # Fititng model with no history for visualization only
        spline_mat0_act = compute_spline_mat(X0_act)
        gamma_0_act = sm.GLM(Y0_act, spline_mat0_act, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_0_act = gamma_0_act.fit()
        mu_0_act_nohist = gamma_res_0_act.mu
        v_0_act_nohist = 1 / gamma_res_0_act.scale
        params_0_act_nohist = gamma_res_0_act.params

        # Fitting model with history
        cov0_act = np.concatenate((spline_mat0_act, Y0hist_act), axis=1)
        gamma_0_act = sm.GLM(Y0_act, cov0_act, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_0_act = gamma_0_act.fit()
        mu_0_act = gamma_res_0_act.mu
        v_0_act = 1 / gamma_res_0_act.scale
        params_0_act = gamma_res_0_act.params

        # For morph 0 inactive trials
        X = np.array(1)
        Y = np.array(1)
        Y_hist = np.array(1)
        first_one = True
        for tr_id in morph0_inact:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            Y_hist_tr = compute_hist_cov(Y_tr, hist_wind)
            X_tr = X_tr[hist_wind:]
            Y_tr = Y_tr[hist_wind:]
            if first_one:
                X = X_tr.copy()
                Y = Y_tr.copy()
                Y_hist = Y_hist_tr
                first_one = False
            else:
                X = np.concatenate((X, X_tr), axis=0)
                Y = np.concatenate((Y, Y_tr), axis=0)
                Y_hist = np.concatenate((Y_hist, Y_hist_tr), axis=0)
        X0_inact = X.copy()
        Y0_inact = Y.copy()
        Y0hist_inact = Y_hist.copy()

        # Fititng model with no history for visualization only
        spline_mat0_inact = compute_spline_mat(X0_inact)
        gamma_0_inact = sm.GLM(Y0_inact, spline_mat0_inact, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_0_inact = gamma_0_inact.fit()
        mu_0_inact_nohist = gamma_res_0_inact.mu
        v_0_inact_nohist = 1 / gamma_res_0_inact.scale
        params_0_inact_nohist = gamma_res_0_inact.params

        # Fitting model with history
        cov0_inact = np.concatenate((spline_mat0_inact, Y0hist_inact), axis=1)
        gamma_0_inact = sm.GLM(Y0_inact, cov0_inact, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_0_inact = gamma_0_inact.fit()
        mu_0_inact = gamma_res_0_inact.mu
        v_0_inact = 1 / gamma_res_0_inact.scale
        params_0_inact = gamma_res_0_inact.params

        # For morph 1 active trials
        X = np.array(1)
        Y = np.array(1)
        Y_hist = np.array(1)
        first_one = True
        for tr_id in morph1_act:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            Y_hist_tr = compute_hist_cov(Y_tr, hist_wind)
            X_tr = X_tr[hist_wind:]
            Y_tr = Y_tr[hist_wind:]
            if first_one:
                X = X_tr.copy()
                Y = Y_tr.copy()
                Y_hist = Y_hist_tr
                first_one = False
            else:
                X = np.concatenate((X, X_tr), axis=0)
                Y = np.concatenate((Y, Y_tr), axis=0)
                Y_hist = np.concatenate((Y_hist, Y_hist_tr), axis=0)
        X1_act = X.copy()
        Y1_act = Y.copy()
        Y1hist_act = Y_hist.copy()

        # Fitting model with no history for visualization only
        spline_mat1_act = compute_spline_mat(X1_act)
        gamma_1_act = sm.GLM(Y1_act, spline_mat1_act, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_1_act = gamma_1_act.fit()
        mu_1_act_nohist = gamma_res_1_act.mu
        v_1_act_nohist = 1 / gamma_res_1_act.scale
        params_1_act_nohist = gamma_res_1_act.params

        # Fitting model with history
        cov1_act = np.concatenate((spline_mat1_act, Y1hist_act), axis=1)
        gamma_1_act = sm.GLM(Y1_act, cov1_act, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_1_act = gamma_1_act.fit()
        mu_1_act = gamma_res_1_act.mu
        v_1_act = 1 / gamma_res_1_act.scale
        params_1_act = gamma_res_1_act.params

        # For morph 1 inactive trials
        X = np.array(1)
        Y = np.array(1)
        Y_hist = np.array(1)
        first_one = True
        for tr_id in morph1_inact:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            Y_hist_tr = compute_hist_cov(Y_tr, hist_wind)
            X_tr = X_tr[hist_wind:]
            Y_tr = Y_tr[hist_wind:]
            if first_one:
                X = X_tr.copy()
                Y = Y_tr.copy()
                Y_hist = Y_hist_tr
                first_one = False
            else:
                X = np.concatenate((X, X_tr), axis=0)
                Y = np.concatenate((Y, Y_tr), axis=0)
                Y_hist = np.concatenate((Y_hist, Y_hist_tr), axis=0)
        X1_inact = X.copy()
        Y1_inact = Y.copy()
        Y1hist_inact = Y_hist.copy()

        # Fititng model with no history for visualization only
        spline_mat1_inact = compute_spline_mat(X1_inact)
        gamma_1_inact = sm.GLM(Y1_inact, spline_mat1_inact, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_1_inact = gamma_1_inact.fit()
        mu_1_inact_nohist = gamma_res_1_inact.mu
        v_1_inact_nohist = 1 / gamma_res_1_inact.scale
        params_1_inact_nohist = gamma_res_1_inact.params

        # Fitting model with history
        cov1_inact = np.concatenate((spline_mat1_inact, Y1hist_inact), axis=1)
        gamma_1_inact = sm.GLM(Y1_inact, cov1_inact, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_1_inact = gamma_1_inact.fit()
        mu_1_inact = gamma_res_1_inact.mu
        v_1_inact = 1 / gamma_res_1_inact.scale
        params_1_inact = gamma_res_1_inact.params

        # Fitting 1-Gamma model to all trials with morph 0 (later used for hypothesis tests of multiple maps)
        X = np.array(1)
        Y = np.array(1)
        Y_hist = np.array(1)
        first_one = True
        for tr_id in morph_0_trials:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            Y_hist_tr = compute_hist_cov(Y_tr, hist_wind)
            X_tr = X_tr[hist_wind:]
            Y_tr = Y_tr[hist_wind:]
            if first_one:
                X = X_tr.copy()
                Y = Y_tr.copy()
                Y_hist = Y_hist_tr
                first_one = False
            else:
                X = np.concatenate((X, X_tr), axis=0)
                Y = np.concatenate((Y, Y_tr), axis=0)
                Y_hist = np.concatenate((Y_hist, Y_hist_tr), axis=0)
        X0_all = X.copy()
        Y0_all = Y.copy()
        Y0hist_all = Y_hist.copy()
        spline_mat0_all = compute_spline_mat(X0_all)
        cov0_all = np.concatenate((spline_mat0_all, Y0hist_all), axis=1)
        gamma_0_all = sm.GLM(Y0_all, cov0_all, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_0_all = gamma_0_all.fit()
        mu_0_all = gamma_res_0_all.mu
        v_0_all = 1 / gamma_res_0_all.scale
        params_0_all = gamma_res_0_all.params

        # Fitting 1-Gamma model to all trials with morph 0 (later used for hypothesis tests of multiple maps)
        X = np.array(1)
        Y = np.array(1)
        Y_hist = np.array(1)
        first_one = True
        for tr_id in morph_1_trials:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            Y_hist_tr = compute_hist_cov(Y_tr, hist_wind)
            X_tr = X_tr[hist_wind:]
            Y_tr = Y_tr[hist_wind:]
            if first_one:
                X = X_tr.copy()
                Y = Y_tr.copy()
                Y_hist = Y_hist_tr
                first_one = False
            else:
                X = np.concatenate((X, X_tr), axis=0)
                Y = np.concatenate((Y, Y_tr), axis=0)
                Y_hist = np.concatenate((Y_hist, Y_hist_tr), axis=0)
        X1_all = X.copy()
        Y1_all = Y.copy()
        Y1hist_all = Y_hist.copy()
        spline_mat1_all = compute_spline_mat(X1_all)
        cov1_all = np.concatenate((spline_mat1_all, Y1hist_all), axis=1)
        gamma_1_all = sm.GLM(Y1_all, cov1_all, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_1_all = gamma_1_all.fit()
        mu_1_all = gamma_res_1_all.mu
        v_1_all = 1 / gamma_res_1_all.scale
        params_1_all = gamma_res_1_all.params

        # For nohist model columns 2 and 4 are repetitions of columns 1 and 3 and are meaningless, for sake of indexing
        gamma_fit_0_act[cell_id, :] = [X0_act, Y0_act, Y0hist_act, spline_mat0_act, cov0_act, mu_0_act, v_0_act,
                                       params_0_act]
        gamma_fit_0_act_nohist[cell_id, :] = [X0_act, Y0_act, Y0_act, spline_mat0_act, spline_mat0_act, mu_0_act_nohist,
                                              v_0_act_nohist, params_0_act_nohist]
        gamma_fit_0_inact[cell_id, :] = [X0_inact, Y0_inact, Y0hist_inact, spline_mat0_inact, cov0_inact,
                                         mu_0_inact, v_0_inact, params_0_inact]
        gamma_fit_0_inact_nohist[cell_id, :] = [X0_inact, Y0_inact, Y0_inact, spline_mat0_inact, spline_mat0_inact,
                                                mu_0_inact_nohist, v_0_inact_nohist, params_0_inact_nohist]
        gamma_fit_1_act[cell_id, :] = [X1_act, Y1_act, Y1hist_act, spline_mat1_act, cov1_act, mu_1_act, v_1_act,
                                       params_1_act]
        gamma_fit_1_act_nohist[cell_id, :] = [X1_act, Y1_act, Y1_act, spline_mat1_act, spline_mat1_act, mu_1_act_nohist,
                                              v_1_act_nohist, params_1_act_nohist]
        gamma_fit_1_inact[cell_id, :] = [X1_inact, Y1_inact, Y1hist_inact, spline_mat1_inact, cov1_inact,
                                         mu_1_inact, v_1_inact, params_1_inact]
        gamma_fit_1_inact_nohist[cell_id, :] = [X1_inact, Y1_inact, Y1_inact, spline_mat1_inact, spline_mat1_inact,
                                                mu_1_inact_nohist, v_1_inact_nohist, params_1_inact_nohist]
        gamma_fit_0_all[cell_id, :] = [X0_all, Y0_all, Y0hist_all, spline_mat0_all, cov0_all, mu_0_all, v_0_all,
                                       params_0_all]
        gamma_fit_1_all[cell_id, :] = [X1_all, Y1_all, Y1hist_all, spline_mat1_all, cov1_all, mu_1_all, v_1_all,
                                       params_1_all]

        # Computing log-likelihoods based on different models (only models with history-dependent component)
        for tr_id in trial_ids:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            plt.show()
            Y_hist_tr = compute_hist_cov(Y_tr, hist_wind)
            X_tr = X_tr[hist_wind:]
            Y_tr = Y_tr[hist_wind:]
            ll_act0 = np.zeros(shape=X_tr.shape[0])
            ll_inact0 = np.zeros(shape=X_tr.shape[0])
            ll_act1 = np.zeros(shape=X_tr.shape[0])
            ll_inact1 = np.zeros(shape=X_tr.shape[0])
            ll_all0 = np.zeros(shape=X_tr.shape[0])
            ll_all1 = np.zeros(shape=X_tr.shape[0])
            for i in range(X_tr.shape[0]):
                # Computing ll_act0 and ll_inact0
                ind0 = find_nearest(X0_act, [X_tr[i]])
                Y_hist_now = np.reshape(Y_hist_tr[i, :], newshape=[1, Y_hist_tr.shape[1]])
                inp = np.concatenate((spline_mat0_act[ind0, :], Y_hist_now), axis=1)
                mu_0_act_now = inp.dot(params_0_act)[0]
                mu_0_act_now = max(.1, mu_0_act_now)
                ll_act0[i] = -scipy.special.loggamma(v_0_act) + v_0_act * np.log(
                    v_0_act * Y_tr[i] / mu_0_act_now) - v_0_act * Y_tr[i] / mu_0_act_now - np.log(Y_tr[i])
                sd_0_act_now = mu_0_act_now/np.sqrt(v_0_act)

                ind0 = find_nearest(X0_inact, [X_tr[i]])
                Y_hist_now = np.reshape(Y_hist_tr[i, :], newshape=[1, Y_hist_tr.shape[1]])
                inp = np.concatenate((spline_mat0_inact[ind0, :], Y_hist_now), axis=1)
                mu_0_inact_now = inp.dot(params_0_inact)[0]
                mu_0_inact_now = max(.1, mu_0_inact_now)
                ll_inact0[i] = -scipy.special.loggamma(v_0_inact) + v_0_inact * np.log(
                    v_0_inact * Y_tr[i] / mu_0_inact_now) - v_0_inact * Y_tr[i] / mu_0_inact_now - np.log(Y_tr[i])
                sd_0_inact_now = mu_0_inact_now / np.sqrt(v_0_inact)

                #  Computing ll_act1 and ll_inact1
                ind0 = find_nearest(X1_act, [X_tr[i]])
                Y_hist_now = np.reshape(Y_hist_tr[i, :], newshape=[1, Y_hist_tr.shape[1]])
                inp = np.concatenate((spline_mat1_act[ind0, :], Y_hist_now), axis=1)
                mu_1_act_now = inp.dot(params_1_act)[0]
                mu_1_act_now = max(.1, mu_1_act_now)
                mu_1_act_now = inp.dot(params_1_act)
                ll_act1[i] = -scipy.special.loggamma(v_1_act) + v_1_act * np.log(
                    v_1_act * Y_tr[i] / mu_1_act_now) - v_1_act * Y_tr[i] / mu_1_act_now - np.log(Y_tr[i])
                sd_1_act_now = mu_1_act_now / np.sqrt(v_1_act)

                ind0 = find_nearest(X1_inact, [X_tr[i]])
                Y_hist_now = np.reshape(Y_hist_tr[i, :], newshape=[1, Y_hist_tr.shape[1]])
                inp = np.concatenate((spline_mat1_inact[ind0, :], Y_hist_now), axis=1)
                mu_1_inact_now = inp.dot(params_1_inact)[0]
                mu_1_inact_now = max(.1, mu_1_inact_now)
                ll_inact1[i] = -scipy.special.loggamma(v_1_inact) + v_1_inact * np.log(
                    v_1_inact * Y_tr[i] / mu_1_inact_now) - v_1_inact * Y_tr[i] / mu_1_inact_now - np.log(Y_tr[i])
                sd_1_inact_now = mu_1_inact_now / np.sqrt(v_1_inact)

                #  Computing ll_all0 and ll_all1
                ind0 = find_nearest(X0_all, [X_tr[i]])
                Y_hist_now = np.reshape(Y_hist_tr[i, :], newshape=[1, Y_hist_tr.shape[1]])
                inp = np.concatenate((spline_mat0_all[ind0, :], Y_hist_now), axis=1)
                mu_0_all_now = inp.dot(params_0_all)[0]
                mu_0_all_now = max(.1, mu_0_all_now)
                ll_all0[i] = -scipy.special.loggamma(v_0_all) + v_0_all * np.log(
                    v_0_all * Y_tr[i] / mu_0_all_now) - v_0_all * Y_tr[i] / mu_0_all_now - np.log(Y_tr[i])
                sd_0_all_now = mu_0_all_now / np.sqrt(v_0_all)

                ind0 = find_nearest(X1_all, [X_tr[i]])
                Y_hist_now = np.reshape(Y_hist_tr[i, :], newshape=[1, Y_hist_tr.shape[1]])
                inp = np.concatenate((spline_mat1_all[ind0, :], Y_hist_now), axis=1)
                mu_1_all_now = inp.dot(params_1_all)[0]
                mu_1_all_now = max(.1, mu_1_all_now)
                ll_all1[i] = -scipy.special.loggamma(v_1_all) + v_1_all * np.log(
                    v_1_all * Y_tr[i] / mu_1_all_now) - v_1_all * Y_tr[i] / mu_1_all_now - np.log(Y_tr[i])
                sd_1_all_now = mu_1_all_now / np.sqrt(v_1_all)
            goodness_of_fit[cell_id, tr_id, 0] = ll_all0
            goodness_of_fit[cell_id, tr_id, 1] = ll_all1

            # For each trial in ambiguous environments, assigning one spatial map for each original environment based
            # on computed log-likelihoods
            ll0 = ll_inact0
            identified_activity0 = 'inactive'
            if np.sum(ll_act0) > np.sum(ll_inact0):
                identified_activity0 = 'active'
                ll0 = ll_act0
            ll1 = ll_inact1
            identified_activity1 = 'inactive'
            if np.sum(ll_act1) > np.sum(ll_inact1):
                ll1 = ll_act1
                identified_activity1 = 'active'

            # Computing and normalizing likelihood from log-likelihood
            L0 = np.exp(ll0)
            L1 = np.exp(ll1)
            denom = L0 + L1
            L0 = L0/denom
            L1 = L1/denom

            ll_morph = np.array([ll0, ll1])
            L_morph = np.array([L0, L1])

            # Running the filtering algorithm to compute the decoded probability
            p_morph_filt = np.zeros(shape=[2, X_tr.shape[0]])
            for i in range(X_tr.shape[0]):
                if i == 0:
                    p_morph_filt[0, i] = L0[i]
                    p_morph_filt[1, i] = L1[i]
                if i > 0:
                    p_morph_filt[0, i] = L0[i] * ((1 - p) * p_morph_filt[0, i - 1] + p * p_morph_filt[1, i - 1])
                    p_morph_filt[1, i] = L1[i] * ((1 - p) * p_morph_filt[1, i - 1] + p * p_morph_filt[0, i - 1])
                p_morph_filt[:, i] = p_morph_filt[:, i] / np.sum(p_morph_filt[:, i])

            # Running the smoother algorithm to compute the decoded probability
            p_morph_smooth = np.zeros(shape=[2, X_tr.shape[0]])
            for i in range(X_tr.shape[0] - 1, -1, -1):
                if i == X_tr.shape[0] - 1:
                    p_morph_smooth[0, i] = p_morph_filt[0, i]
                    p_morph_smooth[1, i] = p_morph_filt[1, i]
                if i < X_tr.shape[0] - 1:
                    p_2step_0 = (1 - p) * p_morph_filt[0, i] + p * p_morph_filt[1, i]
                    p_2step_1 = (1 - p) * p_morph_filt[1, i] + p * p_morph_filt[0, i]
                    p_morph_smooth[0, i] = p_morph_filt[0, i] * (
                                (1 - p) * p_morph_smooth[0, i+1] / p_2step_0 + p * p_morph_smooth[1, i+1] / p_2step_1)
                    p_morph_smooth[1, i] = p_morph_filt[1, i]*(
                                (1 - p) * p_morph_smooth[1, i+1] / p_2step_1 + p * p_morph_smooth[0, i+1] / p_2step_0)
                    p_morph_smooth[:, i] = p_morph_smooth[:, i] / np.sum(p_morph_smooth[:, i])
            p_morph[cell_id, tr_id, :] = [X_tr, Y_tr, p_morph_filt, p_morph_smooth, L_morph, ll_morph]

            if visualize:
                print('cell_id = {}, tri_id = {}'.format(cell_id, tr_id))
                actual_morph = 1
                if tr_id in morph_0_trials:
                    actual_morph = 0
                if tr_id in morph_d25_trials:
                    actual_morph = .25
                if tr_id in morph_d50_trials:
                    actual_morph = .50
                if tr_id in morph_d75_trials:
                    actual_morph = .75
                print('for morph0 identified as {}'.format(identified_activity0))
                print('for morph1 identified as {}'.format(identified_activity1))
                print('Actual morph: {}'.format(actual_morph))
                if np.sum(ll_morph[0, :]) > np.sum(ll_morph[1, :]):
                    print('Decoded result: morph 0')
                else:
                    print('Decoded result: morph 1')
                if True:
                    plt.subplot(6, 2, 1)
                    plt.plot(X_tr, L_morph[0, :], color=orange1, label='morph = 0')
                    plt.plot(X_tr, L_morph[1, :], color=blue1, label='morph = 1')
                    plt.ylabel('likelihood')
                    plt.legend()
                    plt.subplot(6, 2, 2)
                    plt.plot(X_tr, L_morph[0, :], color=orange1)
                    plt.plot(X_tr, L_morph[1, :], color=blue1)

                    plt.subplot(6, 2, 3)
                    plt.plot(X_tr, p_morph_filt[0, :], color=orange1)
                    plt.plot(X_tr, p_morph_filt[1, :], color=blue1)
                    plt.ylabel('filter')
                    plt.subplot(6, 2, 4)
                    plt.plot(X_tr, p_morph_filt[0, :], color=orange1)
                    plt.plot(X_tr, p_morph_filt[1, :], color=blue1)

                    plt.subplot(6, 2, 5)
                    plt.plot(X_tr, p_morph_smooth[0, :], color=orange1)
                    plt.plot(X_tr, p_morph_smooth[1, :], color=blue1)
                    plt.ylabel('smoother')
                    plt.subplot(6, 2, 6)
                    plt.plot(X_tr, p_morph_smooth[0, :], color=orange1)
                    plt.plot(X_tr, p_morph_smooth[1, :], color=blue1)

                    if not history:
                        plt.subplot(6, 2, 7)
                        plt.plot(X0_act, mu_0_act_nohist, '.', markersize=1)
                        plt.plot(X0_act, mu_0_act_nohist + 1.96 * mu_0_act_nohist/np.sqrt(v_0_act_nohist), '.',
                                 markersize=1)
                        plt.plot(X0_act, mu_0_act_nohist - 1.96 * mu_0_act_nohist/np.sqrt(v_0_act_nohist), '.',
                                 markersize=1)
                        plt.plot(X_tr, Y_tr, ',')
                        plt.ylabel('active')

                        plt.subplot(6, 2, 9)
                        plt.plot(X0_inact, mu_0_inact_nohist, '.', markersize=1)
                        plt.plot(X0_inact, mu_0_inact_nohist + 1.96 * mu_0_inact_nohist / np.sqrt(v_0_inact_nohist),
                                 '.', markersize=1)
                        plt.plot(X0_inact, mu_0_inact_nohist - 1.96 * mu_0_inact_nohist / np.sqrt(v_0_inact_nohist),
                                 '.', markersize=1)
                        plt.plot(X_tr, Y_tr, ',')
                        plt.ylabel('inactive')
                        plt.xlabel('morph = 0')

                        plt.subplot(6, 2, 8)
                        plt.plot(X1_act, mu_1_act_nohist, '.', markersize=1)
                        plt.plot(X1_act, mu_1_act_nohist + 1.96 * mu_1_act_nohist / np.sqrt(v_1_act_nohist),
                                 '.', markersize=1)
                        plt.plot(X1_act, mu_1_act_nohist - 1.96 * mu_1_act_nohist / np.sqrt(v_1_act_nohist),
                                 '.', markersize=1)
                        plt.plot(X_tr, Y_tr, ',')

                        plt.subplot(6, 2, 10)
                        plt.plot(X1_inact, mu_1_inact_nohist, '.', markersize=1)
                        plt.plot(X1_inact, mu_1_inact_nohist + 1.96 * mu_1_inact_nohist / np.sqrt(v_1_inact_nohist),
                                 '.', markersize=1)
                        plt.plot(X1_inact, mu_1_inact_nohist - 1.96 * mu_1_inact_nohist / np.sqrt(v_1_inact_nohist),
                                 '.', markersize=1)
                        plt.plot(X_tr, Y_tr, ',')
                        plt.xlabel('morph = 1')
                    else:
                        plt.subplot(6, 2, 7)
                        plt.plot(X0_act, mu_0_act, '.', markersize=1)
                        plt.plot(X0_act, mu_0_act + 1.96 * mu_0_act/np.sqrt(v_0_act), '.', markersize=1)
                        plt.plot(X0_act, mu_0_act - 1.96 * mu_0_act/np.sqrt(v_0_act), '.', markersize=1)
                        plt.plot(X_tr, Y_tr, ',')
                        plt.ylabel('active')

                        plt.subplot(6, 2, 9)
                        plt.plot(X0_inact, mu_0_inact, '.', markersize=1)
                        plt.plot(X0_inact, mu_0_inact + 1.96 * mu_0_inact / np.sqrt(v_0_inact), '.', markersize=1)
                        plt.plot(X0_inact, mu_0_inact - 1.96 * mu_0_inact / np.sqrt(v_0_inact), '.', markersize=1)
                        plt.plot(X_tr, Y_tr, ',')
                        plt.ylabel('inactive')
                        plt.xlabel('morph = 0')

                        plt.subplot(6, 2, 8)
                        plt.plot(X1_act, mu_1_act, '.', markersize=1)
                        plt.plot(X1_act, mu_1_act + 1.96 * mu_1_act / np.sqrt(v_1_act), '.', markersize=1)
                        plt.plot(X1_act, mu_1_act - 1.96 * mu_1_act / np.sqrt(v_1_act), '.', markersize=1)
                        plt.plot(X_tr, Y_tr, ',')

                        plt.subplot(6, 2, 10)
                        plt.plot(X1_inact, mu_1_inact, '.', markersize=1)
                        plt.plot(X1_inact, mu_1_inact + 1.96 * mu_1_inact / np.sqrt(v_1_inact), '.', markersize=1)
                        plt.plot(X1_inact, mu_1_inact - 1.96 * mu_1_inact / np.sqrt(v_1_inact), '.', markersize=1)
                        plt.plot(X_tr, Y_tr, ',')
                        plt.xlabel('morph = 1')

                    plt.subplot(6, 2, 11)
                    plt.plot(X_tr, ll_morph[0, :] - ll_morph[1, :], color='m')
                    plt.plot(X_tr, np.zeros(X_tr.shape), color='gray')
                    plt.ylabel('loglike diff')
                    plt.xlabel('position')

                    plt.subplot(6, 2, 12)
                    plt.plot(X_tr, ll_morph[0, :] - ll_morph[1, :], color='m')
                    plt.plot(X_tr, np.zeros(X_tr.shape), color='gray')
                    plt.ylabel('loglike diff')
                    plt.xlabel('position')
                    plt.show()
        if visualize2:
            show_decode_results_all_trials(exp_id, [cell_id], p_morph)

    np.save(os.getcwd() + '/Data/' + mode + '_p_morph_exp_' + str(exp_id) + '.npy', p_morph)
    np.save(os.getcwd() + '/Data/' + mode + '_goodness_of_fit_exp_' + str(exp_id) + '.npy', goodness_of_fit)
    np.save(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_act_exp_' + str(exp_id) + '.npy', gamma_fit_0_act)
    np.save(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_act_nohist_exp_' + str(exp_id) + '.npy', gamma_fit_0_act_nohist)
    np.save(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_inact_exp_' + str(exp_id) + '.npy', gamma_fit_0_inact)
    np.save(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_inact_nohist_exp_' + str(exp_id) + '.npy', gamma_fit_0_inact_nohist)
    np.save(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_all_exp_' + str(exp_id) + '.npy', gamma_fit_0_all)
    np.save(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_act_exp_' + str(exp_id) + '.npy', gamma_fit_1_act)
    np.save(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_act_nohist_exp_' + str(exp_id) + '.npy', gamma_fit_1_act_nohist)
    np.save(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_inact_exp_' + str(exp_id) + '.npy', gamma_fit_1_inact)
    np.save(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_inact_nohist_exp_' + str(exp_id) + '.npy', gamma_fit_1_inact_nohist)
    np.save(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_all_exp_' + str(exp_id) + '.npy', gamma_fit_1_all)


def decode_morphs_joint(exp_id, p, mode, visualize):
    # For each trial, use the log-likelihood values computed by function "decode_morphs" for individual cells to decode
    # the represented environment based on the full-population activity
    # exp_id: experiment id
    # p: probability of jumping form one state to the other one in a first-order Markov chain
    # mode: shows if we use small subset of data (mode = short) or all data (mode = all)
    # visualize: determines if the function must show the full-population decoding results or not

    # each row = [X_tr, Y_tr, p_morphs_filt, p_morph_smooth, p_morph_likelihood]
    p_morph_joint = np.empty(shape=[ntrials, 6], dtype=object)
    trial_ids = range(ntrials)
    for tr_id in trial_ids:
        ind = np.where(VRData[:, 20] == tr_id)[0]
        X_tr = VRData[ind, 3]
        Y_tr = np.mean(F[:, ind], axis=0)
        X_tr = X_tr[hist_wind:]
        Y_tr = Y_tr[hist_wind:]
        p_morph_filt_joint = np.zeros(shape=[2, X_tr.shape[0]])
        L_morph_joint = np.zeros(shape=[2, X_tr.shape[0]])
        ll_morph_joint = np.zeros(shape=[2, X_tr.shape[0]])
        for i in range(X_tr.shape[0]):
            ll0 = 0
            ll1 = 0
            for cell_id in cells_under_study:
                print('tr_id = {}, cell_id = {}'.format(tr_id, cell_id))
                temp = p_morph[cell_id, tr_id, 5]
                ll0 += temp[0, i]
                ll1 += temp[1, i]
            ll_morph_joint[:, i] = [ll0, ll1]
            # Adding a constant for computational reasons (doesn't affect normalized log-likelihoods)
            tmp = (ll0 + ll1) / 2
            ll0 = ll0 - tmp
            ll1 = ll1 - tmp
            L0 = np.exp(ll0)
            L1 = np.exp(ll1)
            if i == 0:
                p_morph_filt_joint[0, i] = L0
                p_morph_filt_joint[1, i] = L1
            if i > 0:
                p_morph_filt_joint[0, i] = L0 * (
                            (1 - p) * p_morph_filt_joint[0, i - 1] + p * p_morph_filt_joint[1, i - 1])
                p_morph_filt_joint[1, i] = L1 * (
                            (1 - p) * p_morph_filt_joint[1, i - 1] + p * p_morph_filt_joint[0, i - 1])
            p_morph_filt_joint[:, i] = p_morph_filt_joint[:, i] / np.sum(p_morph_filt_joint[:, i])
            L_morph_joint[0, i] = L0
            L_morph_joint[1, i] = L1
            L_morph_joint[:, i] = L_morph_joint[:, i] / np.sum(L_morph_joint[:, i])
        p_morph_smooth_joint = np.zeros(shape=[2, X_tr.shape[0]])
        for i in range(X_tr.shape[0] - 1, -1, -1):
            if i == X_tr.shape[0] - 1:
                p_morph_smooth_joint[0, i] = p_morph_filt_joint[0, i]
                p_morph_smooth_joint[1, i] = p_morph_filt_joint[1, i]
            if i < X_tr.shape[0] - 1:
                p_2step_0 = (1 - p) * p_morph_filt_joint[0, i] + p * p_morph_filt_joint[1, i]
                p_2step_1 = (1 - p) * p_morph_filt_joint[1, i] + p * p_morph_filt_joint[0, i]
                p_morph_smooth_joint[0, i] = p_morph_filt_joint[0, i] * (
                        (1 - p) * p_morph_smooth_joint[0, i + 1] / p_2step_0 + p * p_morph_smooth_joint[
                    1, i + 1] / p_2step_1)
                p_morph_smooth_joint[1, i] = p_morph_filt_joint[1, i] * (
                        (1 - p) * p_morph_smooth_joint[1, i + 1] / p_2step_1 + p * p_morph_smooth_joint[
                    0, i + 1] / p_2step_0)

                p_morph_smooth_joint[:, i] = p_morph_smooth_joint[:, i] / np.sum(p_morph_smooth_joint[:, i])
        p_morph_joint[tr_id, :] = [X_tr, Y_tr, p_morph_filt_joint, p_morph_smooth_joint, L_morph_joint, ll_morph_joint]
        if visualize == True:
            plt.subplot(4, 1, 1)
            plt.plot(X_tr, L_morph_joint[0, :], color=orange1, label='morph = 0')
            plt.plot(X_tr, L_morph_joint[1, :], color=blue1, label='morph = 1')
            plt.ylabel('likelihood')
            plt.legend()

            plt.subplot(4, 1, 2)
            plt.plot(X_tr, p_morph_filt_joint[0, :], color=orange1)
            plt.plot(X_tr, p_morph_filt_joint[1, :], color=blue1)
            plt.ylabel('filter')

            plt.subplot(4, 1, 3)
            plt.plot(X_tr, p_morph_smooth_joint[0, :], color=orange1)
            plt.plot(X_tr, p_morph_smooth_joint[1, :], color=blue1)
            plt.ylabel('smoother')

            plt.subplot(4, 1, 4)
            plt.plot(X_tr, Y_tr, ',')
            plt.show()

    np.save(os.getcwd() + '/Data/' + mode + '_p_morph_joint_exp_' + str(exp_id) + '.npy', p_morph_joint)


def decode_morphs_joint_selected(exp_id, p, mode, visualize, visualize2, visualize3, visualize4, selected_trial,
                                 diff_trials, selected_cells):
    # For each trial, use the log-likelihood values computed by function "decode_morphs" for individual cells to decode
    # the represented environment based on activity of a subset of cells under study, called "selected cells".
    # exp_id: experiment id
    # p: probability of jumping form one state to the other one in a first-order Markov chain.
    # mode: shows if we use shorted version of data (mode = short) or all data (mode = all)
    # selected_cells: cells that we use to compute the joint decoding probability
    # visualize: determines if the function must show the decoded results for each trial or not.
    # visualize2: determines if the function must show a heatmap of decoded probabilities for all trials or not.
    # visualize3: determines if the function must show a heatmap for contribution of different cells in the decoding
    # results during a single trial called "selected trial".
    # selected_trial: the trial used if the visualize3 is TRUE.
    # visualize4: determines if the function must show a heatmap for contribution of different cells in the decoding
    # results for multiple trials given in "diff_trial". Cells will be sorted based on the location of the maximum
    # contribution for the first trial in vector "diff_trials".
    # diff_trial: the vector of trial used if visualize4 is TRUE.

    #  Note: Both visualize3 and visualize4 shouldn't be TRUE simultaneously
    for cell_id in selected_cells:
        if cell_id not in cells_under_study:
            print('WARNING: THE SELECTED CELL IS NOT IN UNDER STUDY CELLS!')

    # each row = [X_tr, Y_tr, p_morphs_filt, p_morph_smooth, p_morph_likelihood]
    p_morph_joint_selected = np.empty(shape=[ntrials, 6], dtype=object)
    trial_ids = range(ntrials)
    if visualize4:
        tr_id = diff_trials[0]
        ind = np.where(VRData[:, 20] == tr_id)[0]
        X_tr = VRData[ind, 3]
        Y_tr = np.mean(F[:, ind], axis=0)
        X_tr = X_tr[hist_wind:]
        Y_tr = Y_tr[hist_wind:]
        p_morph_filt_joint = np.zeros(shape=[2, X_tr.shape[0]])
        L_morph_joint = np.zeros(shape=[2, X_tr.shape[0]])
        ll_morph_joint = np.zeros(shape=[2, X_tr.shape[0]])
        log_ll = np.zeros(shape=[len(selected_cells), X_tr.shape[0]])
        for i in range(X_tr.shape[0]):
            cnt = 0
            for cell_id in selected_cells:
                temp = p_morph[cell_id, tr_id, 5]
                log_ll[cnt, i] = temp[0, i] - temp[1, i]
                cnt += 1
        min_log_ll = np.argmin(log_ll, axis=1)
        selected_cells = selected_cells[np.argsort(min_log_ll)]

    for tr_id in trial_ids:
        ind = np.where(VRData[:, 20] == tr_id)[0]
        X_tr = VRData[ind, 3]
        Y_tr = np.mean(F[:, ind], axis=0)
        X_tr = X_tr[hist_wind:]
        Y_tr = Y_tr[hist_wind:]
        p_morph_filt_joint = np.zeros(shape=[2, X_tr.shape[0]])
        L_morph_joint = np.zeros(shape=[2, X_tr.shape[0]])
        ll_morph_joint = np.zeros(shape=[2, X_tr.shape[0]])
        # Constructing log-likelhood difference for using in visualize3 or visualize4 later
        if visualize3 and tr_id == selected_trial:
            log_ll = np.zeros(shape=[len(selected_cells), X_tr.shape[0]])
            for i in range(X_tr.shape[0]):
                cnt = 0
                for cell_id in selected_cells:
                    temp = p_morph[cell_id, tr_id, 5]
                    log_ll[cnt, i] = temp[0, i] - temp[1, i]
                    cnt += 1
        if visualize4 and tr_id in diff_trials:
            log_ll = np.zeros(shape=[len(selected_cells), X_tr.shape[0]])
            for i in range(X_tr.shape[0]):
                cnt = 0
                for cell_id in selected_cells:
                    temp = p_morph[cell_id, tr_id, 5]
                    log_ll[cnt, i] = temp[0, i] - temp[1, i]
                    cnt += 1

        # Running filter and smoother algorithms to compute the population decoded probabilities
        for i in range(X_tr.shape[0]):
            ll0 = 0
            ll1 = 0
            for cell_id in selected_cells:
                temp = p_morph[cell_id, tr_id, 5]
                ll0 += temp[0, i]
                ll1 += temp[1, i]
            ll_morph_joint[:, i] = [ll0, ll1]
            # Adding a constant for computational reasons (doesn't affect normalized log-likelihoods)
            tmp = (ll0 + ll1) / 2
            ll0 = ll0 - tmp
            ll1 = ll1 - tmp
            L0 = np.exp(ll0)
            L1 = np.exp(ll1)
            if i == 0:
                p_morph_filt_joint[0, i] = L0
                p_morph_filt_joint[1, i] = L1
            if i > 0:
                p_morph_filt_joint[0, i] = L0 * (
                            (1 - p) * p_morph_filt_joint[0, i - 1] + p * p_morph_filt_joint[1, i - 1])
                p_morph_filt_joint[1, i] = L1 * (
                            (1 - p) * p_morph_filt_joint[1, i - 1] + p * p_morph_filt_joint[0, i - 1])
            p_morph_filt_joint[:, i] = p_morph_filt_joint[:, i] / np.sum(p_morph_filt_joint[:, i])
            L_morph_joint[0, i] = L0
            L_morph_joint[1, i] = L1
            L_morph_joint[:, i] = L_morph_joint[:, i] / np.sum(L_morph_joint[:, i])  # L_morph_joint is Normalized
        p_morph_smooth_joint = np.zeros(shape=[2, X_tr.shape[0]])
        for i in range(X_tr.shape[0] - 1, -1, -1):
            if i == X_tr.shape[0] - 1:
                p_morph_smooth_joint[0, i] = p_morph_filt_joint[0, i]
                p_morph_smooth_joint[1, i] = p_morph_filt_joint[1, i]
            if i < X_tr.shape[0] - 1:
                # State transition model: to other state with prob. p
                p_2step_0 = (1 - p) * p_morph_filt_joint[0, i] + p * p_morph_filt_joint[1, i]
                p_2step_1 = (1 - p) * p_morph_filt_joint[1, i] + p * p_morph_filt_joint[0, i]
                p_morph_smooth_joint[0, i] = p_morph_filt_joint[0, i] * (
                        (1 - p) * p_morph_smooth_joint[0, i + 1] / p_2step_0 + p * p_morph_smooth_joint[
                    1, i + 1] / p_2step_1)
                p_morph_smooth_joint[1, i] = p_morph_filt_joint[1, i] * (
                        (1 - p) * p_morph_smooth_joint[1, i + 1] / p_2step_1 + p * p_morph_smooth_joint[
                    0, i + 1] / p_2step_0)

                p_morph_smooth_joint[:, i] = p_morph_smooth_joint[:, i] / np.sum(p_morph_smooth_joint[:, i])
        p_morph_joint_selected[tr_id, :] = [X_tr, Y_tr, p_morph_filt_joint, p_morph_smooth_joint, L_morph_joint, ll_morph_joint]  # ll is normalized, L is normalized

        if visualize:
            plt.subplot(4, 1, 1)
            plt.plot(X_tr, L_morph_joint[0, :], color=orange1, label='morph = 0')
            plt.plot(X_tr, L_morph_joint[1, :], color=blue1, label='morph = 1')
            plt.ylabel('likelihood')
            plt.legend()

            plt.subplot(4, 1, 2)
            plt.plot(X_tr, p_morph_filt_joint[0, :], color=orange1)
            plt.plot(X_tr, p_morph_filt_joint[1, :], color=blue1)
            plt.ylabel('filter')

            plt.subplot(4, 1, 3)
            plt.plot(X_tr, p_morph_smooth_joint[0, :], color=orange1)
            plt.plot(X_tr, p_morph_smooth_joint[1, :], color=blue1)
            plt.ylabel('smoother')

            plt.subplot(4, 1, 4)
            plt.plot(X_tr, Y_tr, ',')
            plt.show()

        if visualize3 and tr_id == selected_trial:
            plt.subplot(2, 1, 1)
            plt.plot(X_tr, p_morph_smooth_joint[0, :], linewidth=3, color='purple')
            plt.xlim(np.min(X_tr), np.max(X_tr))
            plt.ylim([-.1, 1.1])
            plt.yticks([])
            plt.xticks([])
            plt.subplot(2, 1, 2)
            plt.imshow(log_ll, cmap='coolwarm', aspect='auto', vmin=-.5, vmax=.5)
            plt.yticks([])
            plt.xticks(np.arange(0, log_ll.shape[1], int(log_ll.shape[1] / 3)), [])
            plt.show()

        if visualize4 and tr_id in diff_trials:
            plt.subplot(2, 1, 1)
            plt.plot(X_tr, p_morph_smooth_joint[0, :], linewidth=3, color='purple')
            plt.xlim(np.min(X_tr), np.max(X_tr))
            plt.ylim([-.1, 1.1])
            plt.yticks([])
            if tr_id == diff_trials[0]:
                plt.yticks([0, 1], [])
            plt.xticks([])
            plt.subplot(2, 1, 2)
            plt.imshow(log_ll, cmap='coolwarm', aspect='auto', vmin=-2, vmax=2)
            plt.yticks([])
            plt.xticks([])
            if tr_id in [0, 74]:
                plt.xticks(np.arange(0, log_ll.shape[1], int(log_ll.shape[1] / 3)), [])
            plt.show()

    if visualize2:
        all_morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
        l = len(all_morphs)
        X_range = np.arange(np.min(VRData[:, 3]), np.max(VRData[:, 3]), 0.5)
        avgs = []
        for j in range(l):
            trial_ids = all_morphs[j]
            P0 = np.zeros(shape=[trial_ids.shape[0], X_range.shape[0]])
            for i in range(trial_ids.shape[0]):
                tr_id = trial_ids[i]
                ind = find_nearest(p_morph_joint_selected[tr_id, 0], X_range)
                pp = p_morph_joint_selected[tr_id, 3]
                P0[i, :] = pp[0, ind]
            plt.subplot(l, 1, j + 1)
            plt.imshow(P0, cmap='coolwarm', aspect='auto')
            plt.xticks([])
            plt.yticks([])
            if j == l - 1:
                plt.xticks(np.arange(0, P0.shape[1], int(P0.shape[1] / 3)), [])
            temp = np.mean(P0, axis=1)
            avgs = avgs + list(temp)
        plt.show()
        avgs = np.array(avgs)
        X_avg = []
        for i in range(len(all_morphs)):
            morph = all_morphs[i]
            X_avg = X_avg + list(np.arange(30*i, 30*(i+1)-.9, 30/len(morph)))
        X_avg = np.array(X_avg)
        avgs = gaussian_filter(avgs, sigma=.5)
        plt.plot(X_avg, avgs, linewidth=3, color='purple')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    np.save(os.getcwd() + '/Data/' + mode + '_p_morph_joint_selected_exp_' + str(exp_id) + '.npy', p_morph_joint_selected)


def compute_cells_dec_perf(exp_id, cells_under_study, mode):
    # Compute the decoding performance of each cell for trials that belongs to original environments
    # exp_id: experiment id
    # cells_under_study: cells for which we are computing the decoding performance
    # mode: indicates if we are working with a small subset of cells (mode = short) or all cells (mode = all)

    cells_dec_perf = np.zeros(shape=[ncells, 5])  # Each row: [cell_id, avg_ll0, hit_rate0, avg_ll1, hit_rate1]
    for cell_id in cells_under_study:
        cells_dec_perf[cell_id, 0] = cell_id
        # Computing performance for morph = 0
        avg_ll = 0
        hit = 0
        for tr_id in morph_0_trials:
            lls = p_morph[cell_id, tr_id, 5]
            ll0s = lls[0, :]
            ll1s = lls[1, :]
            avg_ll += np.mean(ll0s) - np.mean(ll1s)
            if np.mean(ll0s) > np.mean(ll1s):
                hit += 1
        avg_ll = avg_ll / morph_0_trials.shape[0]
        cells_dec_perf[cell_id, 1] = avg_ll
        cells_dec_perf[cell_id, 2] = hit / morph_0_trials.shape[0]

        # Computing performance for morph = 1
        avg_ll = 0
        hit = 0
        for tr_id in morph_1_trials:
            lls = p_morph[cell_id, tr_id, 5]
            ll0s = lls[0, :]
            ll1s = lls[1, :]
            avg_ll += np.mean(ll1s) - np.mean(ll0s)
            if np.mean(ll1s) > np.mean(ll0s):
                hit += 1
        avg_ll = avg_ll / morph_1_trials.shape[0]
        cells_dec_perf[cell_id, 3] = avg_ll
        cells_dec_perf[cell_id, 4] = hit / morph_1_trials.shape[0]
    np.save(os.getcwd() + '/Data/' + mode + '_cells_dec_perf_exp_' + str(exp_id) + '.npy', cells_dec_perf)


def show_decode_results_all_trials(exp_id, cell_ids, p_morph):
    # For each cell in cell_ids show the decoding probability for all trials as a function of position split by morph
    # levels and active/inactive. Also show the top-activity of morph0 and morph1 trials for each cell.
    # exp_id: experiment id
    # cell_ids: the set of cells that we want to show this for.
    # p_morph: contains the decoding results for all cells and trials.

    X_range = np.arange(np.min(VRData[:, 3]), np.max(VRData[:, 3]), 0.5)
    all_morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
    active_trials = [active_trials_0, active_trials_d25, active_trials_d50, active_trials_d75, active_trials_1]
    for cell_id in cell_ids:
        print('cell id = {}'.format(cell_id))
        plt.subplot(1, 2, 1)
        plt.plot(np.sort(top_activity_trials_0[cell_id, :]), '.')
        plt.title('morph = 0')
        plt.subplot(1, 2, 2)
        plt.plot(np.sort(top_activity_trials_1[cell_id, :]), '.')
        plt.title('morph = 1')
        plt.show()
        for j in range(5):
            tr_ids = all_morphs[j]
            act_tr = active_trials[j]
            tr_ids = np.concatenate((tr_ids[act_tr[cell_id, :] == 1], tr_ids[act_tr[cell_id, :] == 0]), axis=0)
            P0 = np.zeros(shape=[tr_ids.shape[0], X_range.shape[0]])
            for i in range(tr_ids.shape[0]):
                tr_id = tr_ids[i]
                ind = find_nearest(p_morph[cell_id, tr_id, 0], X_range)
                pp = p_morph[cell_id, tr_id, 3]
                P0[i, :] = pp[0, ind]
            sh = np.concatenate((act_tr[cell_id, act_tr[cell_id, :] == 1], act_tr[cell_id, act_tr[cell_id, :] == 0]),
                                axis=0)
            sh = np.reshape(sh, newshape=[sh.shape[0], 1])
            plt.subplot2grid((5, 10), (j, 0), colspan=1)
            plt.imshow(sh, aspect='auto', extent=[0, 0.5, sh.shape[0], 0])
            plt.ylabel('Trials \n (m={})'.format(j*.25))
            if j == 0:
                plt.title('active/ \n inactive')
            plt.subplot2grid((5, 10), (j, 1), colspan=9)
            plt.imshow(P0, cmap='coolwarm', aspect='auto')
            plt.colorbar()
            if j == 4:
                plt.xlabel('position')
            if j == 0:
                plt.title('P(morph = 0)')
        plt.show()


def show_multiple_maps_neurips(cell_ids):
    # For individual cells, demonstrate the existence of multiple parallel spatial maps within original environments.
    # For visual aids, trials are sorted based on the location of their place fields, or their average activity rate.
    # cell_ids: cells that we want to show this for.

    for cell_id in cell_ids:
        activity_rates_morph_0[cell_id, :, :] = activity_rates_morph_0[cell_id, :, :] / \
                                                np.mean(activity_rates_morph_0[cell_id, :, :])
        activity_rates_morph_1[cell_id, :, :] = activity_rates_morph_1[cell_id, :, :] / \
                                                np.mean(activity_rates_morph_1[cell_id, :, :])
    for j in range(len(cell_ids)):
        cell_id = cell_ids[j]
        activities = [activity_rates_morph_0, activity_rates_morph_1]
        morphs = [morph_0_trials, morph_1_trials]
        print(cell_id)
        for i in range(len(morphs)):
            curr_activity = activities[i]
            m = morphs[i]
            plt.subplot(len(morphs), 1, i+1)
            mat = curr_activity[cell_id, :, m].copy()
            mat1 = np.array(range(len(m)))
            mat1 = np.reshape(mat1, newshape=[mat1.shape[0], 1])
            mat2 = np.max(mat, axis=1)
            if cell_id in [56, 202, 25]:
                mat2 = np.argmax(mat, axis=1)
            mat2 = np.reshape(mat2, newshape=[mat2.shape[0], 1])
            mat3 = np.concatenate((mat1, mat2), axis=1)
            ind = np.argsort(mat3[:, 1])
            mat3 = mat3[ind, :]
            plotted_mat = mat[mat3[:, 0].astype(int), :]
            maxx = 1
            plotted_mat = plotted_mat/maxx
            plt.imshow(plotted_mat, aspect='auto', vmin=0, vmax=6)
            plt.colorbar()
            if i < len(morphs)-1:
                plt.xticks([])
            if i == len(morphs)-1:
                plt.xticks(np.arange(0, plotted_mat.shape[1], int(plotted_mat.shape[1]/3)), [])
            if j > 0:
                plt.yticks([])
            if j == 0:
                plt.yticks(np.arange(0, plotted_mat.shape[0], int(plotted_mat.shape[0]/3)), [])
        plt.show()


def show_population_maps_neurips(exp_id, tr_ids, cell_ids):
    # Show the population-wide representation of space during trials of original environments
    # cell_ids: cells that construct the population
    # tr_ids: trials that we want to show this for

    for j in range(len(tr_ids)):
        tr_id = tr_ids[j]
        if tr_id in morph_0_trials:
            activity = activity_rates_morph_0
        if tr_id in morph_d25_trials:
            activity = activity_rates_morph_d25
        if tr_id in morph_d50_trials:
            activity = activity_rates_morph_d50
        if tr_id in morph_d75_trials:
            activity = activity_rates_morph_d75
        if tr_id in morph_1_trials:
            activity = activity_rates_morph_1

        for cell_id in cell_ids:
            activity[cell_id, :, :] = activity[cell_id, :, :]/np.mean(activity[cell_id, :, :])

        # sort cells based on place fields for the first trial in tr_ids
        mat = activity[cell_ids, :, tr_id]
        mat1 = np.array(range(mat.shape[0]))
        mat1 = np.reshape(mat1, newshape=[mat1.shape[0], 1])
        mat2 = np.argmax(mat, axis=1)
        mat2 = np.reshape(mat2, newshape=[mat2.shape[0], 1])
        mat3 = np.concatenate((mat1, mat2), axis=1)
        if j == 0:
            ind = np.argsort(mat3[:, 1])
            mat3 = mat3[ind, :]
            index = mat3[:, 0].astype(int)
        plotted_mat = mat[index, :]
        plt.imshow(plotted_mat, aspect='auto', vmin=0, vmax=6)
        if j > 0:
            plt.yticks([])
        if j == 0:
            plt.yticks(np.arange(0, plotted_mat.shape[0], int(plotted_mat.shape[0] / 3)-1), [])
        plt.xticks(np.arange(0, plotted_mat.shape[1], int(plotted_mat.shape[1] / 3)), [])
        plt.show()


def show_fluctuations_neurips(exp_id, cell_ids, plot_cells):
    # For each cell and each trial, compute the similarity fraction based on the equation given in paper NeurIPS 2020.
    # tr_ids: set of trials that we want to compute the similarity fraction for.
    # exp_id = experiment id
    # cell_ids: set of cells under study (no special usage in this function, except for making the code more readable).
    # plot_cells: cells that we want to show the similarity fractions for.

    arr0 = activity_rates_morph_0[cell_ids, :, :]
    arr0 = arr0[:, :, morph_0_trials]
    arr1 = activity_rates_morph_1[cell_ids, :, :]
    arr1 = arr1[:, :, morph_1_trials]

    # Computing similarity based on distance for one cell and all trials
    morphs = [morph_d25_trials, morph_d50_trials, morph_d75_trials]
    l = len(morphs)
    for cell_id in plot_cells:
        i = np.where(cell_ids == cell_id)[0]
        i = i[0]
        print('cell_id = {}'.format(cell_id))
        for j in range(l):
            curr_morph = morphs[j]
            plt.subplot(l, 1, j + 1)
            for tr_id in curr_morph:
                if tr_id in morph_0_trials:
                    curr_act = activity_rates_morph_0[cell_ids, :, :]
                    curr_act = curr_act[:, :, tr_id]
                if tr_id in morph_d25_trials:
                    curr_act = activity_rates_morph_d25[cell_ids, :, :]
                    curr_act = curr_act[:, :, tr_id]
                if tr_id in morph_d50_trials:
                    curr_act = activity_rates_morph_d50[cell_ids, :, :]
                    curr_act = curr_act[:, :, tr_id]
                if tr_id in morph_d75_trials:
                    curr_act = activity_rates_morph_d75[cell_ids, :, :]
                    curr_act = curr_act[:, :, tr_id]
                if tr_id in morph_1_trials:
                    curr_act = activity_rates_morph_1[cell_ids, :, :]
                    curr_act = curr_act[:, :, tr_id]
                diff0 = np.zeros(shape=[arr0.shape[1], ])
                for tr0 in range(arr0.shape[2]):
                    diff = np.abs(arr0[i, :, tr0] - curr_act[i, :])
                    diff0 += diff
                diff1 = np.zeros(shape=[arr1.shape[1], ])
                for tr1 in range(arr1.shape[2]):
                    diff = np.abs(arr1[i, :, tr1] - curr_act[i, :])
                    diff1 += diff
                avg_diff0 = gaussian_filter(diff0 / arr0.shape[2], sigma=2)
                avg_diff1 = gaussian_filter(diff1 / arr1.shape[2], sigma=2)
                plt.plot(avg_diff0 / (avg_diff0 + avg_diff1), color=(j/4, 0, 1-j/4), alpha=.1)
                plt.ylim([.1, .9])
                plt.yticks([])
                if i == 16:
                    plt.yticks([.25, .5, .75], [])
                if j == l-1:
                    plt.xticks(np.arange(0, len(avg_diff0), int(len(avg_diff0) / 3)), [])
                else:
                    plt.xticks([])
        plt.show()


def show_decoding_results_neurips(exp_id, cell_ids):
    # For each cell in cell_ids show the decoded probabilities for all trials along position split by morph levels
    # and active/inactive. Also show the top-activity of morph0 and morph1 trials for each cell (mainly for debugging).
    # exp_id: experiment id
    # cell_ids: the set of cells that we want to show this for.

    X_range = np.arange(np.min(VRData[:, 3]), np.max(VRData[:, 3]), 0.5)
    all_morphs = [morph_d25_trials, morph_d50_trials, morph_d75_trials]
    active_trials = [active_trials_d25, active_trials_d50, active_trials_d75]
    l = len(all_morphs)
    for cell_id in cell_ids:
        print('cell id = {}'.format(cell_id))
        for j in range(l):
            tr_ids = all_morphs[j]
            act_tr = active_trials[j]
            tr_ids = np.concatenate((tr_ids[act_tr[cell_id, :] == 1], tr_ids[act_tr[cell_id, :] == 0]), axis=0)
            P0 = np.zeros(shape=[tr_ids.shape[0], X_range.shape[0]])
            for i in range(tr_ids.shape[0]):
                tr_id = tr_ids[i]
                ind = find_nearest(p_morph[cell_id, tr_id, 0], X_range)
                pp = p_morph[cell_id, tr_id, 3]
                P0[i, :] = pp[0, ind]
            plt.subplot(l, 1, j+1)
            plt.imshow(P0, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
            plt.yticks([])
            if j == l-1:
                plt.xticks(np.arange(0, X_range.shape[0], int(X_range.shape[0] / 3)), [])
            else:
                plt.xticks([])
        plt.show()


def perform_hypothesis_multiple_maps(exp_id, cell_ids):
    # For each original environment, for each cell, perform hypothesis tests for existence of multiple maps.
    # H_0: K_j = 1          H_1: K_j = 2
    # cell_ids: cells that we want to perform this hypothesis test for.

    # For morph = 0
    p_vals0 = []
    for cell_id in cell_ids:
        ll_red = 0
        ll_full = 0
        for tr_id in morph_0_trials:
            ll_1gam = goodness_of_fit[cell_id, tr_id, 0]
            ll_morph = p_morph[cell_id, tr_id, 5]
            ll_2gam = ll_morph[0, :]
            ll_red += np.sum(ll_1gam)
            ll_full += np.sum(ll_2gam)
        dof = gamma_fit_0_act[cell_id, 4].shape[1]
        chi2_stat = -2 * (np.sum(ll_red) - np.sum(ll_full))
        p_value = 1 - scipy.stats.chi2.cdf(chi2_stat, dof)
        p_vals0.append(p_value)
    plt.subplot(2, 1, 1)
    plt.plot(np.sort(p_vals0))
    plt.plot([0, len(p_vals0)], [.05, .05], color='grey')
    plt.show()

    # For morph = 1
    p_vals1 = []
    for cell_id in cell_ids:
        ll_red = 0
        ll_full = 0
        for tr_id in morph_1_trials:
            ll_1gam = goodness_of_fit[cell_id, tr_id, 1]
            ll_morph = p_morph[cell_id, tr_id, 5]
            ll_2gam = ll_morph[1, :]
            ll_red += np.sum(ll_1gam)
            ll_full += np.sum(ll_2gam)
        dof = gamma_fit_1_act[cell_id, 4].shape[1]
        chi2_stat = -2 * (np.sum(ll_red) - np.sum(ll_full))
        p_value = 1 - scipy.stats.chi2.cdf(chi2_stat, dof)
        p_vals1.append(p_value)
    plt.subplot(2, 1, 2)
    plt.plot(np.sort(p_vals1))
    plt.plot([0, len(p_vals1)], [.05, .05], color='grey')
    plt.show()


def compute_trans_prob(exp_id, p_range, cell_ids, trial_ids):
    # For each cell, first fit 2-Gamma models for each original environment. Then for each p, decode the represented
    # environment during ambiguous environments and compute the likelihood. Visualize these as a function of p
    # and compute its MLE for p and its sd, and use a Normal-approximation Wald test.
    # H_0: p = 0          H_1: p > 0
    # exp_id: experiment id
    # p_range: A vector containing the range of values of p
    # cell_ids: cells that we want to perform hypothesis test for.
    # trial_ids: trials used for ambiguous environments

    cnt = 1
    p_ll = np.zeros(shape=[ncells, ntrials, p_range.shape[0]])
    for cell_id in cell_ids:
        print('processing the {}-th cell'.format(cnt))
        cnt += 1
        ind = np.where(active_trials_0[cell_id, :] == 1)[0]
        morph0_act = morph_0_trials[ind]
        ind = np.where(active_trials_0[cell_id, :] == 0)[0]
        morph0_inact = morph_0_trials[ind]
        ind = np.where(active_trials_1[cell_id, :] == 1)[0]
        morph1_act = morph_1_trials[ind]
        ind = np.where(active_trials_1[cell_id, :] == 0)[0]
        morph1_inact = morph_1_trials[ind]

        # For morph 0 active trials
        X = np.array(1)
        Y = np.array(1)
        Y_hist = np.array(1)
        first_one = True
        for tr_id in morph0_act:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            Y_hist_tr = compute_hist_cov(Y_tr, hist_wind)
            X_tr = X_tr[hist_wind:]
            Y_tr = Y_tr[hist_wind:]
            if first_one:
                X = X_tr.copy()
                Y = Y_tr.copy()
                Y_hist = Y_hist_tr
                first_one = False
            else:
                X = np.concatenate((X, X_tr), axis=0)
                Y = np.concatenate((Y, Y_tr), axis=0)
                Y_hist = np.concatenate((Y_hist, Y_hist_tr), axis=0)
        X0_act = X.copy()
        Y0_act = Y.copy()
        Y0hist_act= Y_hist.copy()
        spline_mat0_act = compute_spline_mat(X0_act)
        cov0_act = np.concatenate((spline_mat0_act, Y0hist_act), axis=1)
        gamma_0_act = sm.GLM(Y0_act, cov0_act, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_0_act = gamma_0_act.fit()
        mu_0_act = gamma_res_0_act.mu
        v_0_act = 1 / gamma_res_0_act.scale
        params_0_act = gamma_res_0_act.params

        # For morph 0 inactive trials
        X = np.array(1)
        Y = np.array(1)
        Y_hist = np.array(1)
        first_one = True
        for tr_id in morph0_inact:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            Y_hist_tr = compute_hist_cov(Y_tr, hist_wind)
            X_tr = X_tr[hist_wind:]
            Y_tr = Y_tr[hist_wind:]
            if first_one:
                X = X_tr.copy()
                Y = Y_tr.copy()
                Y_hist = Y_hist_tr
                first_one = False
            else:
                X = np.concatenate((X, X_tr), axis=0)
                Y = np.concatenate((Y, Y_tr), axis=0)
                Y_hist = np.concatenate((Y_hist, Y_hist_tr), axis=0)
        X0_inact = X.copy()
        Y0_inact = Y.copy()
        Y0hist_inact = Y_hist.copy()
        spline_mat0_inact = compute_spline_mat(X0_inact)
        cov0_inact = np.concatenate((spline_mat0_inact, Y0hist_inact), axis=1)
        gamma_0_inact = sm.GLM(Y0_inact, cov0_inact, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_0_inact = gamma_0_inact.fit()
        mu_0_inact = gamma_res_0_inact.mu
        v_0_inact = 1 / gamma_res_0_inact.scale
        params_0_inact = gamma_res_0_inact.params

        # For morph 1 active trials
        X = np.array(1)
        Y = np.array(1)
        Y_hist = np.array(1)
        first_one = True
        for tr_id in morph1_act:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            Y_hist_tr = compute_hist_cov(Y_tr, hist_wind)
            X_tr = X_tr[hist_wind:]
            Y_tr = Y_tr[hist_wind:]
            if first_one:
                X = X_tr.copy()
                Y = Y_tr.copy()
                Y_hist = Y_hist_tr
                first_one = False
            else:
                X = np.concatenate((X, X_tr), axis=0)
                Y = np.concatenate((Y, Y_tr), axis=0)
                Y_hist = np.concatenate((Y_hist, Y_hist_tr), axis=0)
        X1_act = X.copy()
        Y1_act = Y.copy()
        Y1hist_act = Y_hist.copy()
        spline_mat1_act = compute_spline_mat(X1_act)
        cov1_act = np.concatenate((spline_mat1_act, Y1hist_act), axis=1)
        gamma_1_act = sm.GLM(Y1_act, cov1_act, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_1_act = gamma_1_act.fit()
        mu_1_act = gamma_res_1_act.mu
        v_1_act = 1 / gamma_res_1_act.scale
        params_1_act = gamma_res_1_act.params

        # For morph 1 inactive trials
        X = np.array(1)
        Y = np.array(1)
        Y_hist = np.array(1)
        first_one = True
        for tr_id in morph1_inact:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            Y_hist_tr = compute_hist_cov(Y_tr, hist_wind)
            X_tr = X_tr[hist_wind:]
            Y_tr = Y_tr[hist_wind:]
            if first_one:
                X = X_tr.copy()
                Y = Y_tr.copy()
                Y_hist = Y_hist_tr
                first_one = False
            else:
                X = np.concatenate((X, X_tr), axis=0)
                Y = np.concatenate((Y, Y_tr), axis=0)
                Y_hist = np.concatenate((Y_hist, Y_hist_tr), axis=0)
        X1_inact = X.copy()
        Y1_inact = Y.copy()
        Y1hist_inact = Y_hist.copy()
        spline_mat1_inact = compute_spline_mat(X1_inact)
        cov1_inact = np.concatenate((spline_mat1_inact, Y1hist_inact), axis=1)
        gamma_1_inact = sm.GLM(Y1_inact, cov1_inact, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_1_inact = gamma_1_inact.fit()
        mu_1_inact = gamma_res_1_inact.mu
        v_1_inact = 1 / gamma_res_1_inact.scale
        params_1_inact = gamma_res_1_inact.params

        for tr_id in trial_ids:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            plt.show()
            Y_hist_tr = compute_hist_cov(Y_tr, hist_wind)
            X_tr = X_tr[hist_wind:]
            Y_tr = Y_tr[hist_wind:]

            ll_act0 = np.zeros(shape=X_tr.shape[0])
            ll_inact0 = np.zeros(shape=X_tr.shape[0])
            ll_act1 = np.zeros(shape=X_tr.shape[0])
            ll_inact1 = np.zeros(shape=X_tr.shape[0])
            ll_all0 = np.zeros(shape=X_tr.shape[0])
            ll_all1 = np.zeros(shape=X_tr.shape[0])
            for i in range(X_tr.shape[0]):
                # Computing log-likelihoods for morph = 0
                ind0 = find_nearest(X0_act, [X_tr[i]])
                Y_hist_now = np.reshape(Y_hist_tr[i, :], newshape=[1, Y_hist_tr.shape[1]])
                inp = np.concatenate((spline_mat0_act[ind0, :], Y_hist_now), axis=1)
                mu_0_act_now = inp.dot(params_0_act)[0]
                mu_0_act_now = max(.1, mu_0_act_now)  # rarely estimated mu is negative which is not desired
                ll_act0[i] = -scipy.special.loggamma(v_0_act) + v_0_act * np.log(
                    v_0_act * Y_tr[i] / mu_0_act_now) - v_0_act * Y_tr[i] / mu_0_act_now - np.log(Y_tr[i])
                sd_0_act_now = mu_0_act_now/np.sqrt(v_0_act)

                ind0 = find_nearest(X0_inact, [X_tr[i]])
                Y_hist_now = np.reshape(Y_hist_tr[i, :], newshape=[1, Y_hist_tr.shape[1]])
                inp = np.concatenate((spline_mat0_inact[ind0, :], Y_hist_now), axis=1)
                mu_0_inact_now = inp.dot(params_0_inact)[0]
                mu_0_inact_now = max(.1, mu_0_inact_now)  # rarely estimated mu is negative which is not desired
                ll_inact0[i] = -scipy.special.loggamma(v_0_inact) + v_0_inact * np.log(
                    v_0_inact * Y_tr[i] / mu_0_inact_now) - v_0_inact * Y_tr[i] / mu_0_inact_now - np.log(Y_tr[i])
                sd_0_inact_now = mu_0_inact_now / np.sqrt(v_0_inact)

                # Computing log-likelihoods for morph = 1
                ind0 = find_nearest(X1_act, [X_tr[i]])
                Y_hist_now = np.reshape(Y_hist_tr[i, :], newshape=[1, Y_hist_tr.shape[1]])
                inp = np.concatenate((spline_mat1_act[ind0, :], Y_hist_now), axis=1)
                mu_1_act_now = inp.dot(params_1_act)[0]
                mu_1_act_now = max(.1, mu_1_act_now)  # rarely estimated mu is negative which is not desired
                mu_1_act_now = inp.dot(params_1_act)
                ll_act1[i] = -scipy.special.loggamma(v_1_act) + v_1_act * np.log(
                    v_1_act * Y_tr[i] / mu_1_act_now) - v_1_act * Y_tr[i] / mu_1_act_now - np.log(Y_tr[i])
                sd_1_act_now = mu_1_act_now / np.sqrt(v_1_act)

                ind0 = find_nearest(X1_inact, [X_tr[i]])
                Y_hist_now = np.reshape(Y_hist_tr[i, :], newshape=[1, Y_hist_tr.shape[1]])
                inp = np.concatenate((spline_mat1_inact[ind0, :], Y_hist_now), axis=1)
                mu_1_inact_now = inp.dot(params_1_inact)[0]
                mu_1_inact_now = max(.1, mu_1_inact_now)  # rarely estimated mu is negative which is not desired
                ll_inact1[i] = -scipy.special.loggamma(v_1_inact) + v_1_inact * np.log(
                    v_1_inact * Y_tr[i] / mu_1_inact_now) - v_1_inact * Y_tr[i] / mu_1_inact_now - np.log(Y_tr[i])
                sd_1_inact_now = mu_1_inact_now / np.sqrt(v_1_inact)

            # Assigning active/inactive as what we did in function "decode_morph"
            ll0 = ll_inact0
            identified_activity0 = 'inactive'
            if np.sum(ll_act0) > np.sum(ll_inact0):
                identified_activity0 = 'active'
                ll0 = ll_act0
            ll1 = ll_inact1
            identified_activity1 = 'inactive'
            if np.sum(ll_act1) > np.sum(ll_inact1):
                ll1 = ll_act1
                identified_activity1 = 'active'

            # Computing normalized likelihood from log-likelihood
            L0 = np.exp(ll0)
            L1 = np.exp(ll1)
            denom = L0 + L1
            L0 = L0/denom
            L1 = L1/denom

            ll_morph = np.array([ll0, ll1])
            L_morph = np.array([L0, L1])
            # Running the filtering and smoother algorithms
            for j in range(p_range.shape[0]):
                p = p_range[j]

                p_morph_filt = np.zeros(shape=[2, X_tr.shape[0]])
                for i in range(X_tr.shape[0]):
                    if i == 0:
                        p_morph_filt[0, i] = L0[i]
                        p_morph_filt[1, i] = L1[i]
                    if i > 0:
                        p_morph_filt[0, i] = L0[i] * ((1 - p) * p_morph_filt[0, i - 1] + p * p_morph_filt[1, i - 1])
                        p_morph_filt[1, i] = L1[i] * ((1 - p) * p_morph_filt[1, i - 1] + p * p_morph_filt[0, i - 1])
                    p_morph_filt[:, i] = p_morph_filt[:, i] / np.sum(p_morph_filt[:, i])
                p_morph_smooth = np.zeros(shape=[2, X_tr.shape[0]])
                for i in range(X_tr.shape[0] - 1, -1, -1):
                    if i == X_tr.shape[0] - 1:
                        p_morph_smooth[0, i] = p_morph_filt[0, i]
                        p_morph_smooth[1, i] = p_morph_filt[1, i]
                    if i < X_tr.shape[0] - 1:
                        p_2step_0 = (1 - p) * p_morph_filt[0, i] + p * p_morph_filt[1, i]
                        p_2step_1 = (1 - p) * p_morph_filt[1, i] + p * p_morph_filt[0, i]
                        p_morph_smooth[0, i] = p_morph_filt[0, i] * (
                                    (1 - p) * p_morph_smooth[0, i+1] / p_2step_0 + p * p_morph_smooth[1, i+1] / p_2step_1)
                        p_morph_smooth[1, i] = p_morph_filt[1, i]*(
                                    (1 - p) * p_morph_smooth[1, i+1] / p_2step_1 + p * p_morph_smooth[0, i+1] / p_2step_0)

                        p_morph_smooth[:, i] = p_morph_smooth[:, i] / np.sum(p_morph_smooth[:, i])
                new_p_ll = np.sum(np.log(p_morph_smooth[0, :]*L0 + p_morph_smooth[1, :]*L1))
                p_ll[cell_id, tr_id, j] = new_p_ll
    np.save(os.getcwd() + '/Data/' + mode + '_trans_prob_ll_exp_' + str(exp_id) + '.npy', p_ll)


def show_trans_prob(exp_id, cell_ids, trial_ids):
    # Showing the result of function "compute_trans_prob" to visualize the posterior distributions of p, computing the
    # p-values and perform the hypothesis test
    # H_0: p = 0          H_1: p > 0
    # exp_id: experiment id
    # cell_ids: cells that we want to perform hypothesis test for.
    # trial_ids: trials used for ambiguous environments

    for cell_id in cell_ids:
        p_ll_agg = np.sum(trans_prob_ll[cell_id, trial_ids, :], axis=0)
        p_L_agg = p_ll_agg - np.mean(p_ll_agg)
        p_L_agg = np.exp(p_L_agg)
        p_L_agg = p_L_agg / np.sum(p_L_agg)
        ind = np.argmax(p_ll_agg)
        print('cell-id= {}, best p = {}'.format(cell_id, p_range[ind]))
        p_mean = np.sum(p_range * p_L_agg)
        p_sd = np.sqrt(np.sum((p_range - p_mean)**2 * p_L_agg))
        print('\t \t \t p_value = {}'.format(1 - norm.cdf(np.abs(0.35 - p_range[ind])/p_sd)))
        plt.plot(p_range, p_L_agg)
        plt.ylim([-0.01, 0.08])
        plt.xticks([])
        plt.xticks([0, 0.33, 0.66, 1], [])
        plt.yticks([])
        if cell_id == cell_ids[0]:
            plt.yticks([0, .035, .07], [])
        plt.show()


''' ------------------------------- ENCODING ------------------------------- '''

# Defining some colors for debugging plots
blue1 = (75/255, 139/255, 190/255)
orange1 = (255/255, 212/255, 59/255)
mr1 = 'b'
mr2 = blue1
mr3 = 'purple'
mr4 = 'pink'
mr5 = 'r'

# Which experiment to work with?
# exp_id = 3: A Rare session
# exp_id = 4: A Frequent session
exp_id = 3

# Reading the deconvolved activity rates aligned with movement data
F = np.load(os.getcwd() + '/Data/suite2p' + str(exp_id) + '/plane0/F.npy')  # cells by timepoints
Fneu = np.load(os.getcwd() + '/Data/suite2p' + str(exp_id) + '/plane0/Fneu.npy')  # cells by timepoints
iscell = np.load(os.getcwd() + '/Data/suite2p' + str(exp_id) + '/plane0/iscell.npy')  # cells by 2
stat = np.load(os.getcwd() + '/Data/suite2p' + str(exp_id) + '/plane0/stat.npy', allow_pickle=True) # cells by 1
spks = np.load(os.getcwd() + '/Data/suite2p' + str(exp_id) + '/plane0/spks.npy')  # cells by timepoints
ops = np.load(os.getcwd() + '/Data/suite2p' + str(exp_id) + '/plane0/ops.npy', allow_pickle=True)  # dictionary
VRData = np.load(os.getcwd() + '/Data/VRData' + str(exp_id) + '.npy')  # timepoints by 20 features

# Deleting 8th predictor to make data sets for different experiments consistent and of the same size
if exp_id in [3, 4]:
    VRData = np.delete(VRData, 8, axis=1)

# Deleting first multiple seconds for negative position
ind = np.where(VRData[:, 3] < -60)[0]
starting_time = max(ind)+1
F = F[:, starting_time:]
Fneu = Fneu[:, starting_time:]
spks = spks[:, starting_time:]
VRData = VRData[starting_time:, :]

# Detecting cells with negative activity rate and removing them from dataset
F_min = np.min(F, axis=1)
ind = np.where(F_min < 0)[0]
F = np.delete(F, ind, axis=0)
Fneu = np.delete(Fneu, ind, axis=0)
spks = np.delete(spks, ind, axis=0)

# Deleting data for the last trial which is an incomplete trial
ind = np.where(VRData[:, 20] == np.max(VRData[:, 20]))[0]
VRData = np.delete(VRData, ind, 0)
F = np.delete(F, ind, 1)
Fneu = np.delete(Fneu, ind, 1)
spks = np.delete(spks, ind, 1)
S = spks.T

#  Computing basic statistics of the data
ncells = F.shape[0]  # number of cells
ntimepoints = F.shape[1]  # number of time points
ntrials = len(set(VRData[:, 20]))  # number of trials
min_pos = np.floor(min(VRData[:, 3]))  # smallest position
max_pos = np.ceil(max(VRData[:, 3]))  # largest position

# Extracting trials of each morph level
morph_0_trials = morph_trials(0)  # 1-dim vector
morph_d25_trials = morph_trials(0.25)
morph_d50_trials = morph_trials(0.5)
morph_d75_trials = morph_trials(0.75)
morph_1_trials = morph_trials(1)

# For each morph level, constructing a matrix  that contains activity rates of all cells for all trials with given morph
# level as a function of position
compute_activity_rate(exp_id, morphs=morph_0_trials, morph_lvl=0, breaks=200)
compute_activity_rate(exp_id, morphs=morph_d25_trials, morph_lvl=0.25, breaks=200)
compute_activity_rate(exp_id, morphs=morph_d50_trials, morph_lvl=0.5, breaks=200)
compute_activity_rate(exp_id, morphs=morph_d75_trials, morph_lvl=0.75, breaks=200)
compute_activity_rate(exp_id, morphs=morph_1_trials, morph_lvl=1, breaks=200)
activity_rates_morph_0 = np.load(os.getcwd() + '/Data/activity_rates_exp_' + str(exp_id) + '_morph_0.npy')
activity_count_morph_0 = np.load(os.getcwd() + '/Data/activity_count_exp_' + str(exp_id) + '_morph_0.npy')
activity_rates_morph_d25 = np.load(os.getcwd() + '/Data/activity_rates_exp_' + str(exp_id) + '_morph_0.25.npy')
activity_count_morph_d25 = np.load(os.getcwd() + '/Data/activity_count_exp_' + str(exp_id) + '_morph_0.25.npy')
activity_rates_morph_d50 = np.load(os.getcwd() + '/Data/activity_rates_exp_' + str(exp_id) + '_morph_0.5.npy')
activity_count_morph_d50 = np.load(os.getcwd() + '/Data/activity_count_exp_' + str(exp_id) + '_morph_0.5.npy')
activity_rates_morph_d75 = np.load(os.getcwd() + '/Data/activity_rates_exp_' + str(exp_id) + '_morph_0.75.npy')
activity_count_morph_d75 = np.load(os.getcwd() + '/Data/activity_count_exp_' + str(exp_id) + '_morph_0.75.npy')
activity_rates_morph_1 = np.load(os.getcwd() + '/Data/activity_rates_exp_' + str(exp_id) + '_morph_1.npy')
activity_count_morph_1 = np.load(os.getcwd() + '/Data/activity_count_exp_' + str(exp_id) + '_morph_1.npy')

# For each cell, dividing the trials of each morph level into 2 groups based on the activity pattern using K-means
# These two modes are calمed "active" and "inactive" in the code (inactive trials are not necessarily inactive!)
compute_active_trials(exp_id, morph_0_trials, 0)
compute_active_trials(exp_id, morph_d25_trials, 0.25)
compute_active_trials(exp_id, morph_d50_trials, 0.5)
compute_active_trials(exp_id, morph_d75_trials, 0.75)
compute_active_trials(exp_id, morph_1_trials, 1)
active_trials_0 = np.load(os.getcwd() + '/Data/active_trials_morph_0_exp_' + str(exp_id) + '.npy')
active_trials_d25 = np.load(os.getcwd() + '/Data/active_trials_morph_0.25_exp_' + str(exp_id) + '.npy')
active_trials_d50 = np.load(os.getcwd() + '/Data/active_trials_morph_0.5_exp_' + str(exp_id) + '.npy')
active_trials_d75 = np.load(os.getcwd() + '/Data/active_trials_morph_0.75_exp_' + str(exp_id) + '.npy')
active_trials_1 = np.load(os.getcwd() + '/Data/active_trials_morph_1_exp_' + str(exp_id) + '.npy')
top_activity_trials_0 = np.load(os.getcwd() + '/Data/top_activity_trials_morph_0_exp_' + str(exp_id) + '.npy')
top_activity_trials_d25 = np.load(os.getcwd() + '/Data/top_activity_trials_morph_0.25_exp_' + str(exp_id) + '.npy')
top_activity_trials_d50 = np.load(os.getcwd() + '/Data/top_activity_trials_morph_0.5_exp_' + str(exp_id) + '.npy')
top_activity_trials_d75 = np.load(os.getcwd() + '/Data/top_activity_trials_morph_0.75_exp_' + str(exp_id) + '.npy')
top_activity_trials_1 = np.load(os.getcwd() + '/Data/top_activity_trials_morph_1_exp_' + str(exp_id) + '.npy')
# For each cell, putting active/inactive indicator of all trials in a single matrix
active_trials_all = np.zeros(shape=[ncells, ntrials])
active_trials_all[:, morph_0_trials] = active_trials_0
active_trials_all[:, morph_d25_trials] = active_trials_d25
active_trials_all[:, morph_d50_trials] = active_trials_d50
active_trials_all[:, morph_d75_trials] = active_trials_d75
active_trials_all[:, morph_1_trials] = active_trials_1

# For each cell, for each morph level, computing the log-likelihood of fitted spline-Normal model
compute_spline_Normal_loglike(exp_id, morph_lvl=0, morphs=morph_0_trials)
compute_spline_Normal_loglike(exp_id, morph_lvl=0.25, morphs=morph_d25_trials)
compute_spline_Normal_loglike(exp_id, morph_lvl=0.5, morphs=morph_d50_trials)
compute_spline_Normal_loglike(exp_id, morph_lvl=0.75, morphs=morph_d75_trials)
compute_spline_Normal_loglike(exp_id, morph_lvl=1, morphs=morph_1_trials)
spline_Normal_loglike_morph_0 = np.load(os.getcwd() + '/Data/spline_Normal_loglike_exp_' + str(exp_id) +
                                        '_morph_0.npy')
spline_Normal_loglike_morph_d25 = np.load(os.getcwd() + '/Data/spline_Normal_loglike_exp_' + str(exp_id) +
                                          '_morph_0.25.npy')
spline_Normal_loglike_morph_d50 = np.load(os.getcwd() + '/Data/spline_Normal_loglike_exp_' + str(exp_id) +
                                          '_morph_0.5.npy')
spline_Normal_loglike_morph_d75 = np.load(os.getcwd() + '/Data/spline_Normal_loglike_exp_' + str(exp_id) +
                                          '_morph_0.75.npy')
spline_Normal_loglike_morph_1 = np.load(os.getcwd() + '/Data/spline_Normal_loglike_exp_' + str(exp_id) + '_morph_1.npy')

# For each cell, computing the difference between spline-Normal models fitted for morph=0 and morph=1
compute_spline_Normal_dist(exp_id)
spline_Normal_dist = np.load(os.getcwd() + '/Data/spline_Normal_dist_exp_' + str(exp_id) + '.npy')

# Computing cells that differentiate between morph=0 and morph=1 trials the most based on spline-Normal models
l = np.reshape(spline_Normal_dist, newshape=[ncells, 1])
s = np.reshape(np.arange(0, ncells, 1).T, newshape=[ncells, 1])
imp_diff_cells = np.append(s, l, axis=1)
imp_diff_cells = imp_diff_cells[imp_diff_cells[:, 1].argsort()]
imp_diff_cells = np.flip(imp_diff_cells, axis=0)

# Computing cells that give the largest log-likelihood for morph=0 based on spline-Normal models
l = np.reshape(spline_Normal_loglike_morph_0, newshape=[ncells, 1])
s = np.reshape(np.arange(0, ncells, 1).T, newshape=[ncells, 1])
imp_env0_cells = np.append(s, l, axis=1)
imp_env0_cells = imp_env0_cells[imp_env0_cells[:, 1].argsort()]
imp_env0_cells = np.flip(imp_env0_cells, axis=0)

# Computing cells that gives the largest log-likelihood for morph 0 based on spline-Normal models
l = np.reshape(spline_Normal_loglike_morph_1, newshape=[ncells, 1])
s = np.reshape(np.arange(0, ncells, 1).T, newshape=[ncells, 1])
imp_env1_cells = np.append(s, l, axis=1)
imp_env1_cells = imp_env1_cells[imp_env1_cells[:, 1].argsort()]
imp_env1_cells = np.flip(imp_env1_cells, axis=0)


''' ------------------------------- DECODING ------------------------------- '''

# In first-order Markov chain, p is the probability of going from one state to another one at each iteration. It is
# called q in the NeurIPS 2020 paper.
p = .35
hist_wind = 10  # Length of the history window for history dependent part of spline-Gamma models
# mode = 'short'  # Working with only a small subset of the cells (for debugging and basic visualizations)
mode = 'all'  # Working with all of the cells

if mode == 'short':
    num = 100  # number of cells that we work with
    cells_under_study = imp_diff_cells[:num, 0].astype(int)  # choosing cells to work with
if mode == 'all':
    num = ncells  # number of cells that we work with
    cells_under_study = range(ncells)  # choosing cells to work with

# For each cell, running the filter and smoother algorithm for all trials w.r.t spline-Gamma models
decode_morphs(exp_id, p, mode=mode, visualize=False, visualize2=False, history=False)
p_morph = np.load(os.getcwd() + '/Data/' + mode + '_p_morph_exp_' + str(exp_id) + '.npy', allow_pickle=True)
goodness_of_fit = np.load(os.getcwd() + '/Data/' + mode + '_goodness_of_fit_exp_' + str(exp_id) + '.npy',
                          allow_pickle=True)
gamma_fit_0_act = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_act_exp_' + str(exp_id) + '.npy',
                          allow_pickle=True)
gamma_fit_0_act_nohist = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_act_nohist_exp_' + str(exp_id) + '.npy',
                                 allow_pickle=True)
gamma_fit_0_inact = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_inact_exp_' + str(exp_id) + '.npy',
                            allow_pickle=True)
gamma_fit_0_inact_nohist = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_inact_nohist_exp_' + str(exp_id) +
                                   '.npy', allow_pickle=True)
gamma_fit_0_all = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_all_exp_' + str(exp_id) + '.npy',
                          allow_pickle=True)
gamma_fit_1_act = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_act_exp_' + str(exp_id) + '.npy',
                          allow_pickle=True)
gamma_fit_1_act_nohist = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_act_nohist_exp_' + str(exp_id) +
                                 '.npy', allow_pickle=True)
gamma_fit_1_inact = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_inact_exp_' + str(exp_id) + '.npy',
                            allow_pickle=True)
gamma_fit_1_inact_nohist = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_inact_nohist_exp_' + str(exp_id)+
                                   '.npy', allow_pickle=True)
gamma_fit_1_all = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_all_exp_' + str(exp_id) + '.npy',
                          allow_pickle=True)

# For the whole population,  running the filter and smootehr algorithm
decode_morphs_joint(exp_id, p, mode=mode, visualize=False)
p_morph_joint = np.load(os.getcwd() + '/Data/' + mode + '_p_morph_joint_exp_' + str(exp_id) + '.npy',
                        allow_pickle=True)

# Computing goodness-of-fit for individual cells, in terms of their success in decoding the trials from original
# environments correctly.
compute_cells_dec_perf(exp_id, cells_under_study, mode)
cells_dec_perf = np.load(os.getcwd() + '/Data/' + mode + '_cells_dec_perf_exp_' + str(exp_id) + '.npy',
                         allow_pickle=True)

# Sorting cells under study based on their decoding performance in original environments
# Sort1. Based on percentage of correct decisions in detecting the original environment during trials from original
# environments
temp = cells_dec_perf[cells_under_study, :]
temp = np.flip(temp[np.argsort(temp[:, 2] + temp[:, 4]), :], axis=0)
sorted_hit_cells = temp[:, 0].astype(int)
# Sort2. Based on average log-liklihood difference between the correct original environment and the other original
# environment
temp = cells_dec_perf[cells_under_study, :]
temp = np.flip(temp[np.argsort(temp[:, 1] + temp[:, 3]), :], axis=0)
sorted_avg_ll_cells = temp[:, 0].astype(int)

'''------------------------------- VISUALIZATION & PLOTS -------------------------------'''

# Showing that multiple parallel spatial maps exist for individual cells (Fig. 2A)
cell_ids = [220, 75, 56, 13]
show_multiple_maps_neurips(cell_ids)

# Showing the population-wide representation of environment for some trials (Fig. 2B)
tr_ids = [5, 39, 96, 1]
show_population_maps_neurips(exp_id, tr_ids, imp_diff_cells[:300, 0].astype(int))

# Showing the fluctuations in the represented environment using "similarity fraction" defined in NeurIPS 2020 paper
# (Fig. 2C)
cell_ids = imp_diff_cells[:num, 0].astype(int)
plot_cells = [220, 75, 56, 13]
show_fluctuations_neurips(exp_id, cell_ids, plot_cells)

# Showing the output of smoother algorithm for individual cells (Fig. 3)
cell_ids = imp_diff_cells[:num, 0].astype(int)
cell_ids = [220, 75, 56, 13]
show_decoding_results_neurips(exp_id, cell_ids)

# Showing the population decoding results for two groups of size 3 (Fig. 4A)
population_cells = [75, 56, 13]
decode_morphs_joint_selected(exp_id, p, mode=mode, visualize=False, visualize2=True, visualize3=False,
                             visualize4=False, selected_trial=30, diff_trials=[30], selected_cells=population_cells)
population_cells = [75, 56, 220]
decode_morphs_joint_selected(exp_id, p, mode=mode,   visualize=False, visualize2=True, visualize3=False,
                             visualize4=False, selected_trial=30, diff_trials=[30], selected_cells=population_cells)

# Showing the full-population decoding result for all trials (Fig. 4B)
# This must be ran once for exp_id=3 (left panel) and once for exp_id=4 (right panel)
population_cells = imp_diff_cells[:num, 0].astype(int)
decode_morphs_joint_selected(exp_id, p, mode=mode, visualize=False, visualize2=True, visualize3=False,
                             visualize4=False, selected_trial=30, diff_trials=[30], selected_cells=population_cells)

# Showing the population decoding result for two separate groups of size seven, and in addition for their combination.
# This demonstrate how the population code is dominated by different groups at different times (Fig. 4C)
tr_id = 30
group1 = [153, 202, 175, 559, 805, 376, 240]
decode_morphs_joint_selected(exp_id, p, mode=mode, visualize=False, visualize2=True, visualize3=True,
                             visualize4=False, selected_trial=tr_id, selected_cells=group1)
group2 = [75, 12, 16, 3, 220, 35, 4]
decode_morphs_joint_selected(exp_id, p, mode=mode, visualize=False, visualize2=True, visualize3=True,
                             visualize4=False, selected_trial=tr_id, selected_cells=group2)
group3 = group1 + group2
decode_morphs_joint_selected(exp_id, p, mode=mode, visualize=False, visualize2=True, visualize3=True,
                             visualize4=False, selected_trial=tr_id, selected_cells=group3)

# Showing the contribution of all cells in population code at different time points for six different trials (Fig. 4D)
population_cells = imp_diff_cells[:num, 0].astype(int)
diff_trs = [30, 11, 74, 0, 61, 100]
decode_morphs_joint_selected(exp_id, p, mode=mode, visualize=False, visualize2=True, visualize3=False,
                             visualize4=True, selected_trial=30, diff_trials=diff_trs, selected_cells=population_cells)

# For each cell, performing hypothesis test for existence of multiple spatial maps
# H_0: K_j = 1          H_1: K_j > 2
cell_ids = imp_diff_cells[:num, 0].astype(int)
perform_hypothesis_multiple_maps(exp_id, cell_ids)

# For each cell, performing the hypothesis test for state transition model
# H_0: p = 0          H_1: p > 0
p_range = np.arange(0, 1, .005)
cell_ids = imp_diff_cells[:num, 0].astype(int)
trial_ids = np.array(list(morph_d25_trials) + list(morph_d50_trials) + list(morph_d75_trials))
compute_trans_prob(exp_id, p_range, cell_ids, trial_ids)
trans_prob_ll = np.load(os.getcwd() + '/Data/' + mode + '_trans_prob_ll_exp_' + str(exp_id) + '.npy')

# Showing that Normal approximation for Wald tests are justified by looking at the shape of posterior distributions
# (Supplementary Material, Fig. 1)
cell_ids = [75, 13, 12, 3]
show_trans_prob(exp_id, cell_ids, trial_ids)
