import os
import sys
import numpy as np
from numpy import genfromtxt
import matplotlib
if 'Apple_PubSub_Socket_Render' in os.environ:
    # on OSX
    matplotlib.use('TkAgg')
elif 'JOB_ID' in os.environ:
    # running in an SCC batch job, no interactive graphics
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
    print('Using TkAgg.')
import matplotlib.pyplot as plt
import statistics as st
import math
#
from collections import Counter
# import cv2
# import preprocessing as pr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import interp1d
from scipy import spatial
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
import scipy
import time
from scipy.stats import chi2
from sklearn.cluster import KMeans


def find_nearest(arr1, arr2):
    #  for each element x of arr2, finds the closest element of arr1 to that
    ans = []
    for x in arr2:
        y = np.argmin(np.abs(arr1 - x))
        ans.append(y)
    ans = np.array(ans)
    return ans
def compute_hist_cov (Y, hist_wind):
    # Computes the covariate matrix that is for history-dependent models from a response vector
    # Y: the response vector
    # hist_wind: size of the history window
    Y_hist = np.zeros(shape=[Y.shape[0] - hist_wind, hist_wind])
    for h in range(hist_wind):
        Y_hist[:, h] = Y[h: h + Y.shape[0] - hist_wind]
    return Y_hist
def morph_trials(morph_lvl):
    #  Returns trial ids for the given morph level
    ind = np.where(VRData[:, 1] == morph_lvl)[0]
    lst = VRData[ind, 20].astype(int)
    cnt = Counter(lst)
    out = list(set([x for x in lst if cnt[x] > 4]))  # only keep trials with at least 4 data points (rows)
    out.sort()
    out = np.array(out)
    return out
def compute_activity_rate(exp_id, morphs, morph_lvl, breaks):
    #  Returns a 3d array that for a fixed morph, gives the activity rate of each cell for each trial at each position
    #  morph_level: the fixed morph
    #  morphs: all trials in the fixed morph
    #  breas: number of bins for discretization of the position
    #  exp_id: experiment id

    step = (max_pos - min_pos) / breaks
    activity_rates = np.zeros(shape=[ncells, breaks, ntrials])
    activity_count = np.zeros(shape=[ncells, breaks, ntrials])
    # To use it if data is not available for a position bin
    last_moment_activity_rates = np.zeros(shape=[ncells])
    last_moment_activity_count = np.zeros(shape=[ncells])
    for tr in morphs:
        print(tr)
        for i in range(breaks):
            ind = np.where((VRData[:, 3] >= min_pos + i*step) & (VRData[:, 3] < min_pos + (i+1)*step) & (
                            VRData[:, 20] == tr))[0]

            if len(ind) == 0:  # if there is no data point for a cell of the output, we set rate to be 0
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
    np.save(os.getcwd() + '/Data/activity_rates_exp_' + str(exp_id) + '_morph_' + str(morph_lvl) + '.npy', activity_rates)
    np.save(os.getcwd() + '/Data/activity_count_exp_' + str(exp_id) + '_morph_' + str(morph_lvl) + '.npy', activity_count)
def fit_gamma_alt(Y, X, v, num, alpha, initial_c):
    #  Fit a Gamma model as Y - Gamma(c + X*beta) -- both beta and c are estimated
    #  Y: response vector
    #  X: Covariate matrix
    #  v: the parameter for variance of Gamma (we don't estimate this, consider its given)
    #  num: # of iterations in using IRLS
    #  alpha: learning rate of IRLS
    #  initial_c: the initial value for c
    b_thr = 0.001
    b_log = []
    c_log = []
    ll_log = []
    b_old = np.zeros(shape=[X.shape[1], ])
    b_new = np.zeros(shape=[X.shape[1], ])
    c = initial_c
    ll = 0
    for i in range(num):
        b_old = b_new
        nu = X.dot(b_old)
        mu = np.exp(nu) + c
        if i % 2 == 0:  # update the value of beta
            W = Y / mu
            U = v * X.T.dot(Y / mu - 1)
            I = -1 * v * (X.T*W).dot(X)
            b_new = b_old - alpha * np.linalg.inv(I).dot(U)
            b_log.append(b_new)
            ll = v * np.sum(-1 * Y / mu - np.log(mu))
            ll_log.append(ll)
            if ll == np.max(ll_log):
                b_out = b_new
                c_out = c
                ll_out = ll
            if np.max(np.abs(b_new - b_old)) < b_thr:
                break
        if i % 2 != 0:  # update the value of c
            U = v * np.sum(Y / np.power(mu, 2) - 1 / mu)
            I = v * np.sum(-2 * Y / np.power(mu, 3) + 1 / np.power(mu, 2))
            c = c - 1*U / I  # The learning rate can affect goodness of fit significanlty
            # c = c + U
            c_log.append(c)
    # plt.plot(ll_log)
    # plt.title('not fixed c')
    # plt.show()
    return [b_out, c_out, ll_out, b_log, c_log, ll_log]
def fit_gamma_fixed_c(Y, X, v, c, num, alpha):
    #  Fit a Gamma model as Y - Gamma(c + X*beta) -- in Gamma GLM traditionally there is no c, c is not estimated
    #  Y: response vector
    #  X: Covariate matrix
    #  v: the parameter for variance of Gamma (we don't estimate this, consider its given)
    #  num: # of iterations in using IRLS
    #  alpha: learning rate of IRLS
    b_thr = 0.001
    log_b = []
    b_old = np.zeros(shape=[X.shape[1], ])
    b_new = np.zeros(shape=[X.shape[1], ])
    ll = 0
    ll_log = []
    b_log = []
    for i in range(num):
        b_old = b_new
        nu = X.dot(b_old)
        mu = np.exp(nu) + c
        W = Y / mu
        U = v * X.T.dot(Y / mu - 1)
        I = -1 * v * (X.T*W).dot(X)
        b_new = b_old - alpha * np.linalg.inv(I).dot(U)
        log_b.append(b_new)
        ll = v * np.sum(-1 * Y / mu - np.log(mu))
        ll_log.append(ll)
        if ll == np.max(ll_log):
            b_out = b_new
            ll_out = ll
        b_log.append(b_new)
        if np.max(np.abs(b_new - b_old)) < b_thr:
            break
    # plt.plot(ll_log)
    # plt.title('fixed c')
    # plt.show()
    return [b_out, c, ll_out, b_log, ll_log]
def cos(a, b):
    # Compute the cosine between vectors a and b -- they must be of the same size
    return 1 - spatial.distance.cosine(a, b)

def compute_Gamma_dev_cell(exp_id, morph_lvl, morphs):
    #  For all trials in a fixed morph level (0 or 1), fit a gamma model to the activity of each cell versus
    #  position and compute its deviance
    #  exp_id: experiment id
    #  morph_lvl: morph level
    #  morphs: trials with the fixed morph_level
    dev_cell = []
    c_logg = []
    for cell_id in range(ncells):
        print(cell_id)
        ind = np.where(np.isin(VRData[:, 20], morphs))[0]
        X = []
        X.append(VRData[ind, 3])
        X.append(np.power(VRData[ind, 3], 2))
        X = np.array(X)
        X = X.T
        X = sm.add_constant(X, prepend=True)
        Y = F[cell_id, ind]
        [b, c, ll, b_log, c_log, ll_log] = fit_gamma_alt(Y, X, v=1, num=100, alpha=1, initial_c=1000)
        [b_new, c_new, ll_new, b_log_new, c_log_new, ll_log_new] = fit_gamma_alt(Y, X, v=1, num=100, alpha=1, initial_c=0)
        if ll_new > ll:  #  try two models with initial c = 0 and 1000and take the better model
            [b, c, ll, b_log, c_log, ll_log] = [b_new, c_new, ll_new, b_log_new, c_log_new, ll_log_new]
        c_logg.append(c)
        mu = c + np.exp(X.dot(b))
        dev = 2*np.sum(-1*np.log(Y/mu) + (Y-mu)/mu)
        dev_bar = 2*np.mean(-1*np.log(Y/mu) + (Y-mu)/mu)
        v = (6 + 2*dev_bar)/(dev_bar*(6 + dev_bar)) # Look at v and variance of estimation later
        dev_cell.append([cell_id, dev])
    dev_cell = np.array(dev_cell)
    dev_cell = np.flip(dev_cell[dev_cell[:, 1].argsort()], axis=0)
    np.save(os.getcwd() + '/Data/dev_cell_Gamma_exp_' + str(exp_id) + '_morph_' + str(morph_lvl) + '.npy', dev_cell)
    return c_logg

def compute_Gamma_dev_indic(exp_id):
    #  For all trials in a morph levels 0 and 1, fit two gamma models to estimate the activity of each cell versus
    #  position. In one model the covariates is all positions from both morphs, in the other one we have an indicator
    #  variable that differentiate input for moprh0 and morph1 (see below). Deviance for each model and the improvement
    #  in the deviance for the complex model is returned.
    #  exp_id: experiment id

    dev_cell = []
    for cell_id in range(ncells):
        print(cell_id)
        ind = np.where(np.isin(VRData[:, 20], morph_0_trials + morph_1_trials) & np.isin(VRData[:, 1], [0, 1]))[0]
        Y = F[cell_id, ind]
        pos = np.array([VRData[ind, 3]])
        pos2 = np.power(pos, 2)
        indic = np.array([VRData[ind, 1]])

        # Fitting model for design matrix without indicator
        X_noind = np.append(pos, pos2, axis=0)
        X_noind = X_noind.T
        X_noind = sm.add_constant(X_noind, prepend=True)
        [b_noind, c_noind, ll_noind, b_log, c_log, ll_log] = fit_gamma_alt(Y, X_noind, v=1, num=100, alpha=1, initial_c=1000)
        # print(c_noind)
        # plt.plot(c_log)
        # plt.show()
        # plt.plot(ll_log)
        # plt.show()
        [b_noind, c_noind, ll_noind, b_log, ll_log] = fit_gamma_fixed_c(Y, X_noind, v=1, c=c_noind, num=100, alpha=1)

        # Fitting model for design matrix with indicator
        X = np.append(pos, pos2, axis=0)
        pos_indic = pos * indic
        pos2_indic = pos2 * indic
        X = np.append(X, indic, axis=0)
        X = np.append(X, pos_indic, axis=0)
        X = np.append(X, pos2_indic, axis=0)
        X = X.T
        X = sm.add_constant(X, prepend=True)
        [b_ind, c_ind, ll_ind, b_log, ll_log] = fit_gamma_fixed_c(Y, X, v=1, c=c_noind, num=100, alpha=1)
        mu = c_ind + np.exp(X.dot(b_ind))
        dev_ind = 2*np.sum(-1*np.log(Y/mu) + (Y-mu)/mu)
        mu = c_noind + np.exp(X_noind.dot(b_noind))
        dev_noind = 2*np.sum(-1*np.log(Y/mu) + (Y-mu)/mu)
        # print(dev_ind)
        # print(dev_noind)
        # print(ll_noind)
        # print(ll_ind)
        # print(ll_ind - ll_noind)
        # print()
        dev_cell.append([cell_id, dev_noind, dev_ind, dev_noind - dev_ind])
        # print('c_ind = {}, c_noind = {}, dev_ind = {}, dev_noind = {}'.format(c_ind, c_noind, dev_ind, dev_noind))

    dev_cell = np.array(dev_cell)
    # print(dev_cell[:10, :])

    dev_cell = np.flip(dev_cell[dev_cell[:, 3].argsort()], axis=0)
    # print(dev_cell[:10, :])
    np.save(os.getcwd() + '/Data/dev_cell_Gamma_indic_exp_' + str(exp_id) + '.npy', dev_cell)

def compute_spline_Normal_loglike(exp_id, morph_lvl, morphs):
    #  For each cell, for all tirals with a fixed morph level, fit a Normal GLM where the input comes from a spline
    #  fitted on position. Returns loglikelihood for all cells.
    #  Y ~ Normal(mean = sum beta_0 + sum beta_i g_i(x)) where g_i(x)'s are spline basis evaluated at position x
    #  morph_lvl: fixed morph level
    #  morphs: trials with given morph level
    #  exp_id: experiment id
    print('min= {}, max = {}'.format(min_pos, max_pos))
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
    # Y_disc = Y_disc[:Y_disc.shape[0]-1, :]
    # X_disc = X_disc[:X_disc.shape[0]-1]
    # spline_mat = spline_mat[:spline_mat.shape[0]-1, :]

    loglike = []
    # print(np.isnan(X_disc))
    # print(np.isnan(Y_disc))
    for cell_id in range(ncells):
        print(cell_id)
        gauss_ident = sm.GLM(Y_disc[:, cell_id], spline_mat, family=sm.families.Gaussian(sm.families.links.identity()))
        gauss_ident_results = gauss_ident.fit()
        b = np.array(gauss_ident_results.params)
        est = spline_mat.dot(b)
        loglike.append(gauss_ident_results.llf)
    loglike = np.array(loglike)
    np.save(os.getcwd() + '/Data/spline_Normal_loglike_exp_' + str(exp_id) + '_morph_' + str(morph_lvl) + '.npy', loglike)

def compute_spline_Normal_dist(exp_id):
    #  For each cell, for all trials with morph = 0, fit a Normal GLM where the input comes from a spline
    #  fitted on position. Do the same thing for morph = 1. Return the L2 distance between estimated activities obtained
    #  from these two models for every cell.
    #  Y ~ Normal(mean = sum beta_0 + sum beta_i g_i(x)) where g_i(x)'s are spline basis evaluated at position x
    #  exp_id: experiment id

    # MORPH = 0
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

    # MORPH = 1
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

    # COMPUTE DISTANCE
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

def visualize_spline_fit(morph_lvl, morphs, cell_id):
    #  For a fixed morph level and a fixed cell and all trials in that fixed morph level, fit a spline to activity vs.
    #  position. Show this along with the actual (position, activity) points and a mean activity plot.
    #  morph_lvl: fixed morph level
    #  morphs: trials with the fixed morph level
    #  cell_id: id of the fixed cell
    breaks = 400
    knots = np.linspace(0, breaks - 1, num=10, endpoint=True).astype(int)
    ind = np.where(np.isin(VRData[:, 20], morphs))[0]
    X = VRData[ind, 3]
    Y = F[:, ind]
    # if np.any(np.isnan(Y)):
    #     print('Y has nan')
    # if np.any(np.isnan(X)):
    #     print('X has nan')
    step = (max_pos - min_pos) / breaks
    X_disc = []
    Y_disc = []
    last_Y_value = 0
    for i in range(breaks):
        ind0 = np.where((X >= min_pos + i * step) & (X < min_pos + (i + 1) * step))[0]
        if i == 0:
            X_disc.append(min_pos)
        elif i == breaks - 1:
            X_disc.append(max_pos)
        else:
            X_disc.append(min_pos + (i + 1 / 2) * step)
        if len(ind0) == 0:
            Y_disc.append(last_Y_value)
        else:
            last_Y_value = np.mean(Y[:, ind0], axis=1)
            Y_disc.append(last_Y_value)

    X_disc = np.array(X_disc)
    Y_disc = np.array(Y_disc)
    f = interp1d(X_disc[knots], Y_disc[knots, cell_id], kind='cubic')
    Y_hat = f(X)
    plt.plot(X, Y[cell_id, :], '.', label='actual activity')
    plt.plot(X_disc, Y_disc[:, cell_id], '.', label='mean activity')
    plt.plot(X, Y_hat, '.', label='spline')
    plt.legend()
    plt.title('cell id = {}, morph = {}'.format(cell_id, morph_lvl))
    plt.show()
    # RSS = np.mean((Y[cell_id, :] - Y_hat)**2)
    # print('RSS = {}'.format(RSS))

def compute_loglike_diff_Gamma(exp_id):
    #  For each cell and each trial, fit two 2-polynomial Gamma models, one for morph = 0 and one for morph = 1 (after
    #  excluding the trial under study from the data and compute log-likelihood of the trial under the study for both
    #  models. Return these log-likelihoods and their difference for each cell and each trial.
    #  exp_id: experiment id
    loglike_diff = []
    for cell_id in range(ncells):
        print('***************:{}'.format(cell_id))
        l = []
        for tr_id in range(ntrials):
            # Fit model for morph0
            morphs = morph_0_trials
            if np.isin(tr_id, morph_0_trials):
                morphs = morph_0_trials[morph_0_trials != tr_id]
            ind = np.where(np.isin(VRData[:, 20], morphs))[0]
            X = []
            X.append(VRData[ind, 3])
            X.append(np.power(VRData[ind, 3], 2))
            X = np.array(X)
            X = X.T
            X = sm.add_constant(X, prepend=True)
            Y = F[cell_id, ind]
            v = 1
            [b, c, ll, b_log, c_log, ll_log] = fit_gamma_alt(Y, X, v, num=100, alpha=1, initial_c=1000)

            # Computing likelihood for tr_id using morph0 model
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X = []
            X.append(VRData[ind, 3])
            X.append(np.power(VRData[ind, 3], 2))
            X = np.array(X)
            X = X.T
            X = sm.add_constant(X, prepend=True)
            Y = F[cell_id, ind]
            mu = c + np.exp(X.dot(b))
            ll_env0 = v * np.mean(-1 * Y / mu - np.log(mu))  # usually in deviance this must be sum, I have normalization here

            # Fit model for morph1
            morphs = morph_1_trials
            if np.isin(tr_id, morph_1_trials):
                morphs = morph_1_trials[morph_1_trials != tr_id]
            ind = np.where(np.isin(VRData[:, 20], morphs))[0]
            X = []
            X.append(VRData[ind, 3])
            X.append(np.power(VRData[ind, 3], 2))
            X = np.array(X)
            X = X.T
            X = sm.add_constant(X, prepend=True)
            Y = F[cell_id, ind]
            v = 1
            [b, c, ll, b_log, ll_log] = fit_gamma_fixed_c(Y, X, v, c=c, num=100, alpha=1)

            # Computing likelihood for tr_id using morph1 model
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X = []
            X.append(VRData[ind, 3])
            X.append(np.power(VRData[ind, 3], 2))
            X = np.array(X)
            X = X.T
            X = sm.add_constant(X, prepend=True)
            Y = F[cell_id, ind]
            mu = c + np.exp(X.dot(b))
            ll_env1 = v * np.mean(-1 * Y / mu - np.log(mu))  # usually in deviance this must be sum, I have normalization here

            # identify the morph lvl of tr_id
            morph_lvl = 0
            if np.isin(tr_id, morph_d25_trials):
                morph_lvl = 0.25
            if np.isin(tr_id, morph_d50_trials):
                morph_lvl = 0.50
            if np.isin(tr_id, morph_d75_trials):
                morph_lvl = 0.75
            if np.isin(tr_id, morph_1_trials):
                morph_lvl = 1
            # saving everything
            l.append([tr_id, ll_env0, ll_env1, ll_env0 - ll_env1, morph_lvl])
        loglike_diff.append(l)
    loglike_diff = np.array(loglike_diff)
    loglike_diff = loglike_diff[:, loglike_diff[0, :, 4].argsort(), :]
    np.save(os.getcwd() + '/Data/loglike_diff_Gamma' + str(exp_id) + '.npy', loglike_diff)

def compute_loglike_diff_spline_Normal(exp_id):
    #  For each cell and each trial, fit two 2 Normal models with inputs from spline fits, one for morph = 0 and one
    #  for morph = 1 (I didn't exclude the trial under study from the training data) and compute log-likelihood (RSS
    #  here) of the trial under the study for both models. Return these log-likelihoods and their difference for each
    #  cell and each trial.
    #  exp_id: experiment id
    diff = []
    for tr_id in range(ntrials):
        print('trial id is {}'.format(tr_id))

        # Fit model for morph0
        breaks = 400
        morphs = [x for x in morph_0_trials if x != tr_id]

        ind = np.where(np.isin(VRData[:, 20], morphs))[0]
        X = VRData[ind, 3]
        Y = F[:, ind]
        step = (max_pos - min_pos) / breaks
        X_disc = []
        Y0_disc = []
        for i in range(breaks):
            ind0 = np.where((X >= min_pos + i * step) & (X < min_pos + (i + 1) * step))[0]
            if ind0.size == 0:
                print('screaaaaaam: tr_id = {}'.format(tr_id))
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

        # Fit model for morph1
        breaks = 400
        morphs = [x for x in morph_1_trials if x != tr_id]
        ind = np.where(np.isin(VRData[:, 20], morphs))[0]
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
                Y1_disc.append(Y0_disc[-1])
            if ind0.shape[0] > 0:
                Y1_disc.append(np.mean(Y[:, ind0], axis=1))
        X_disc = np.array(X_disc)
        Y1_disc = np.array(Y1_disc)
        spline_mat1 = compute_spline_mat(X_disc)

        # Comparing loglikelihood for morph0 and morph1 models
        morph_lvl = 0
        if np.isin(tr_id, morph_d25_trials):
            morph_lvl = 0.25
        if np.isin(tr_id, morph_d50_trials):
            morph_lvl = 0.50
        if np.isin(tr_id, morph_d75_trials):
            morph_lvl = 0.75
        if np.isin(tr_id, morph_1_trials):
            morph_lvl = 1
        ind_test = np.where(VRData[:, 20] == tr_id)[0]
        X_test = VRData[ind_test, 3]
        Y_test = F[:, ind_test]
        # print(Y_test.shape)
        Y_test_disc = []
        for i in range(breaks):
            ind0 = np.where((X_test >= min_pos + i * step) & (X_test < min_pos + (i + 1) * step))[0]
            if ind0.size == 0:
                Y_test_disc.append([-1]*ncells)
                continue
            Y_test_disc.append(np.mean(Y_test[:, ind0], axis=1))
        Y_test_disc = np.array(Y_test_disc)
        # print(Y_test_disc.shape)

        r = []
        for cell_id in range(ncells):
            gauss_ident = sm.GLM(Y0_disc[:, cell_id], spline_mat0, family=sm.families.Gaussian(sm.families.links.identity()))
            gauss_ident_results = gauss_ident.fit()
            b = np.array(gauss_ident_results.params)
            est0 = spline_mat0.dot(b)

            gauss_ident = sm.GLM(Y1_disc[:, cell_id], spline_mat1, family=sm.families.Gaussian(sm.families.links.identity()))
            gauss_ident_results = gauss_ident.fit()
            b = np.array(gauss_ident_results.params)
            est1 = spline_mat1.dot(b)
            ind0 = np.where(Y_test_disc[:, 0] != -1)[0]

            RSS0 = np.mean((Y_test_disc[ind0, cell_id] - est0[ind0]) ** 2)
            RSS1 = np.mean((Y_test_disc[ind0, cell_id] - est1[ind0]) ** 2)
            # saving everything
            r.append([tr_id, RSS0, RSS1, RSS0 - RSS1, morph_lvl])
        diff.append(r)
    diff = np.array(diff)
    diff = diff[diff[:, 0, 4].argsort(), :, :]
    np.save(os.getcwd() + '/Data/loglike_diff_spline_Normal_exp_' + str(exp_id) + '.npy', diff)

def compute_loglike_diff_spline_Gamma(exp_id, cells):
    #  For each cell in all_cells and each trial, fit four Gamma with identity links where input is output of spline. These four
    #  models are for morph0/morph1 active/inactive data (and the trial under study is not excluded). log-likelihood
    #  is computed for each trial (time to time) and a final log-likelihood for each morph is computed (as a weighted
    #  average of active/inactive log-likelihoods). Return these log-likelihoods and their difference for each
    #  cell and each trial.
    #  exp_id: experiment id
    #  cells: indicate all_cells
    diff = []
    for cell_id in cells:
        print('cell id = {}'.format(cell_id))
        print('active trials for morph 0 \n {}'.format(active_trials_0[cell_id, :]))
        print('active trials for morph 1 \n {}'.format(active_trials_1[cell_id, :]))
        # x = input('pause!')
        r = []
        for tr_id in range(ntrials):
            print('trial id is {}'.format(tr_id))
            not_crossable_morph0_act = False
            not_crossable_morph0_inact = False
            not_crossable_morph1_act = False
            not_crossable_morph1_inact = False

            #  Fit model for morph0
            breaks = 400
            ind = np.where(active_trials_0[cell_id, :] == 1)[0]
            morph0_act = morph_0_trials[ind]
            ind = np.where(active_trials_0[cell_id, :] == 0)[0]
            morph0_inact = morph_0_trials[ind]

            #  Model for active trials
            morphs = [x for x in morph0_act if x != tr_id]
            morph0_act_wei = len(morphs)
            if len(morphs) == 0:
                not_crossable_morph0_act = True
            else:
                ind =  np.where(np.isin(VRData[:, 20], morphs))[0]
                X = VRData[ind, 3]
                Y = F[cell_id, ind]
                arg_sort = np.argsort(X)
                X0_disc_act = X[arg_sort]
                Y0_disc_act = Y[arg_sort]
                spline_mat0_act = compute_spline_mat(X0_disc_act)

            #  Model for inactive trials
            morphs = [x for x in morph0_inact if x != tr_id]
            morph0_inact_wei = len(morphs)
            if len(morphs) == 0:
                not_crossable_morph0_inact = True
            else:
                ind = np.where(np.isin(VRData[:, 20], morphs))[0]
                X = VRData[ind, 3]
                Y = F[cell_id, ind]
                arg_sort = np.argsort(X)
                X0_disc_inact = X[arg_sort]
                Y0_disc_inact = Y[arg_sort]
                spline_mat0_inact = compute_spline_mat(X0_disc_inact)

            #  Fit model for morph1
            breaks = 400
            ind = np.where(active_trials_1[cell_id, :] == 1)[0]
            morph1_act = morph_1_trials[ind]
            ind = np.where(active_trials_1[cell_id, :] == 0)[0]
            morph1_inact = morph_1_trials[ind]

            #  Model for active trials
            morphs = [x for x in morph1_act if x != tr_id]
            morph1_act_wei = len(morphs)
            if len(morphs) == 0:
                not_crossable_morph1_act = True
            else:
                ind = np.where(np.isin(VRData[:, 20], morphs))[0]
                X = VRData[ind, 3]
                Y = F[cell_id, ind]
                arg_sort = np.argsort(X)
                X1_disc_act = X[arg_sort]
                Y1_disc_act = Y[arg_sort]
                spline_mat1_act = compute_spline_mat(X1_disc_act)

            #  Model for inactive trials
            morphs = [x for x in morph1_inact if x != tr_id]
            morph1_inact_wei = len(morphs)
            if len(morphs) == 0:
                not_crossable_morph1_inact = True
            else:
                ind = np.where(np.isin(VRData[:, 20], morphs))[0]
                X = VRData[ind, 3]
                Y = F[cell_id, ind]
                arg_sort = np.argsort(X)
                X1_disc_inact = X[arg_sort]
                Y1_disc_inact = Y[arg_sort]
                spline_mat1_inact = compute_spline_mat(X1_disc_inact)

            #  Fitting models for moph0/morph1 active/inactive models
            morph_lvl = 0
            if np.isin(tr_id, morph_d25_trials):
                morph_lvl = 0.25
            if np.isin(tr_id, morph_d50_trials):
                morph_lvl = 0.50
            if np.isin(tr_id, morph_d75_trials):
                morph_lvl = 0.75
            if np.isin(tr_id, morph_1_trials):
                morph_lvl = 1
            ind_test = np.where(VRData[:, 20] == tr_id)[0]
            X_test = VRData[ind_test, 3]
            time_test = VRData[ind_test, 0]
            time_test = time_test - np.min(time_test)
            Y_test = F[cell_id, ind_test]

            if not not_crossable_morph0_act:
                gamma_0_act = sm.GLM(Y0_disc_act, spline_mat0_act, family=sm.families.Gamma(sm.families.links.identity))
                gamma_res_0_act = gamma_0_act.fit()
                mu_0_act = gamma_res_0_act.mu
                v_0_act = 1/gamma_res_0_act.scale
                sd_0_act = mu_0_act / np.sqrt(v_0_act)

            if not not_crossable_morph0_inact:
                gamma_0_inact = sm.GLM(Y0_disc_inact, spline_mat0_inact, family=sm.families.Gamma(sm.families.links.identity))
                gamma_res_0_inact = gamma_0_inact.fit()
                mu_0_inact = gamma_res_0_inact.mu
                v_0_inact = 1/gamma_res_0_inact.scale
                sd_0_inact = mu_0_inact / np.sqrt(v_0_inact)

            if not not_crossable_morph1_act:
                gamma_1_act = sm.GLM(Y1_disc_act, spline_mat1_act, family=sm.families.Gamma(sm.families.links.identity))
                gamma_res_1_act = gamma_1_act.fit()
                mu_1_act = gamma_res_1_act.mu
                v_1_act = 1/gamma_res_1_act.scale
                sd_1_act = mu_1_act / np.sqrt(v_1_act)

            if not not_crossable_morph1_inact:
                gamma_1_inact = sm.GLM(Y1_disc_inact, spline_mat1_inact, family=sm.families.Gamma(sm.families.links.identity))
                gamma_res_1_inact = gamma_1_inact.fit()
                mu_1_inact = gamma_res_1_inact.mu
                v_1_inact = 1/gamma_res_1_inact.scale
                sd_1_inact = mu_1_inact / np.sqrt(v_1_inact)

            #  Computing log-liklihood for morph0/morph1 active/inactive models
            #  Also computing 1 log-liklihood for each morph which is a linear combination of active and inactive models
            #  Maybe this can be changed with identifying the trial as one of active/inactive groups and look at the
            #  related log-likelihood
            ll_act0 = 0
            if not not_crossable_morph0_act:
                ind0 = find_nearest(X0_disc_act, X_test)
                ll_act0 = -np.log(scipy.special.gamma(v_0_act)) + v_0_act*np.log(v_0_act*Y_test/mu_0_act[ind0]) - v_0_act*Y_test/mu_0_act[ind0] - np.log(Y_test)
            ll_inact0 = 0
            if not not_crossable_morph0_inact:
                ind0 = find_nearest(X0_disc_inact, X_test)
                ll_inact0 = -np.log(scipy.special.gamma(v_0_inact)) + v_0_inact*np.log(v_0_inact*Y_test/mu_0_inact[ind0]) - v_0_inact*Y_test/mu_0_inact[ind0] - np.log(Y_test)
            ll0 = (ll_act0 * morph0_act_wei + ll_inact0 * morph0_inact_wei)/(morph0_act_wei + morph0_inact_wei)

            ll_act1 = 0
            if not not_crossable_morph1_act:
                ind0 = find_nearest(X1_disc_act, X_test)
                ll_act1 = -np.log(scipy.special.gamma(v_1_act)) + v_1_act*np.log(v_1_act*Y_test/mu_1_act[ind0]) - v_1_act*Y_test/mu_1_act[ind0] - np.log(Y_test)
            ll_inact1 = 0
            if not not_crossable_morph1_inact:
                ind0 = find_nearest(X1_disc_inact, X_test)
                ll_inact1 = -np.log(scipy.special.gamma(v_1_inact)) + v_1_inact*np.log(v_1_inact*Y_test/mu_1_inact[ind0]) - v_1_inact*Y_test/mu_1_inact[ind0] - np.log(Y_test)
            ll1 = (ll_act1 * morph1_act_wei + ll_inact1 * morph1_inact_wei)/(morph1_act_wei + morph1_inact_wei)
            '''
            if tr_id == 28:
            # if True:
                print('ll_act0 mean = {}'.format(np.mean(ll_act0)))
                print('ll_inact0 mean = {}'.format(np.mean(ll_inact0)))
                print('ll0 mean = {}'.format(np.mean(ll0)))

                plt.subplot(2, 2, 1)
                # print(time_test.shape)
                # print(ll_act0.shape)
                # print(ll_inact0.shape)
                # print(ll0.shape)
                # sys.exit()
                plt.plot(X_test, ll_act0, label='ll_act0')
                plt.plot(X_test, ll_inact0, label='ll_inact0')
                plt.plot(X_test, ll0, label='ll0')
                plt.legend()
                plt.subplot(2, 2, 3)
                plt.plot(X0_disc_act, mu_0_act, label='active')
                plt.plot(X0_disc_act, mu_0_act + 2*sd_0_act)
                plt.plot(X0_disc_act, mu_0_act - 2*sd_0_act)
                plt.plot(X0_disc_inact, mu_0_inact, label='inactive')
                plt.plot(X0_disc_inact, mu_0_inact + 2 * sd_0_inact)
                plt.plot(X0_disc_inact, mu_0_inact - 2 * sd_0_inact)
                plt.plot(X_test, Y_test, '.', label='actual flourescence levcel')

                print('ll_act1 mean = {}'.format(np.mean(ll_act1)))
                print('ll_inact1 mean = {}'.format(np.mean(ll_inact1)))
                print('ll1 mean = {}'.format(np.mean(ll1)))

                plt.subplot(2, 2, 2)
                plt.plot(X_test, ll_act1, label='ll_act1')
                plt.plot(X_test, ll_inact1, label='ll_inact1')
                plt.plot(X_test, ll1, label='ll1')
                plt.legend()
                plt.subplot(2, 2, 4)
                plt.plot(X1_disc_act, mu_1_act, label='active')
                plt.plot(X1_disc_act, mu_1_act + 2 * sd_1_act)
                plt.plot(X1_disc_act, mu_1_act - 2 * sd_1_act)
                plt.plot(X1_disc_inact, mu_1_inact, label='inactive')
                plt.plot(X1_disc_inact, mu_1_inact + 2 * sd_1_inact)
                plt.plot(X1_disc_inact, mu_1_inact - 2 * sd_1_inact)
                plt.plot(X_test, Y_test, '.', label='actual flourescence levcel')
                plt.show()
            '''
            ll0 = np.mean(ll0)
            ll1 = np.mean(ll1)
            r.append([tr_id, ll0, ll1, ll0 - ll1, morph_lvl])
        diff.append(r)
    diff = np.array(diff)
    diff = diff[:, diff[0, :, 4].argsort(), :]
    np.save(os.getcwd() + '/Data/loglike_diff_spline_Gamma_exp_' + str(exp_id) + '.npy', diff)

# Check this prior to using:
def compute_PP_dev_cell(exp_id, morph_lvl, morphs):
    #  For all cells, use the data for all trials with a fixed morph level to fit a poly-2 Poisson GLM model and return
    #  deviances. Ofc the spiking activity data is used to fit this model.
    #  exp_id: experiment id
    #  morph_lvl: morph level
    #  morphs: all trials with the given morph level
    dev_cell = []
    for cell_id in range(ncells):
        print(cell_id)
        ind = np.where(np.isin(VRData[:, 20], morphs))[0]
        X = []
        X.append(VRData[ind, 3])
        X.append(np.power(VRData[ind, 3], 2))
        X = np.array(X)
        X = X.T
        X = sm.add_constant(X, prepend=True)
        Y = S[ind, cell_id]
        ind1 = np.where(Y > 100)
        Y = np.zeros(shape=Y.shape)
        Y[ind1] = 1
        model = smf.GLM(Y, X, family=sm.families.Poisson())
        results = model.fit()
        dev_cell.append([cell_id, results.deviance])
    dev_cell = np.array(dev_cell)
    dev_cell = np.flip(dev_cell[dev_cell[:, 1].argsort()], axis=0)
    np.save(os.getcwd() + '/Data/dev_cell_PP_exp_' + str(exp_id) + '_morph_' + str(morph_lvl) + '.npy', dev_cell) #
# Check this prior to using:
def compare_pp_gamma():
    #  For a cell_id and a morph level defined below, fit two models: poly-2 Poisson GLM and Gamma with alternating c
    #  (see fit_gamma_alt function). Then plot the estimated values of these two models and the qctual data points.
    #  Ofc Poisson GLM is fitted using spiking activity and Gamma model is fitted using the Fluorescence activity.
    cell_id = 0
    morph_lvl = 0
    ind = np.where(np.isin(VRData[:, 20], morph_0_trials))[0]
    X = []
    X.append(VRData[ind, 3])
    X.append(np.power(VRData[ind, 3], 2))
    X = np.array(X)
    X = X.T
    X = sm.add_constant(X, prepend=True)
    Y_p = S[ind, cell_id]
    # sys.exit()

    ind1 = np.where(Y_p > 550)
    Y_p = np.zeros(shape=Y_p.shape)
    Y_p[ind1] = 1
    model_p = smf.GLM(Y_p, X, family=sm.families.Poisson())
    results_p = model_p.fit()
    print(results_p.summary())

    # print('*********************************************')

    Y_g = F[cell_id, ind]
    [b, c, ll, b_log, c_log, ll_log] = fit_gamma_alt(Y_g, X, v=1, num=100, alpha=1, initial_c=1000)
    # print(b)
    # print(c)

    mu_p = np.exp(X.dot(results_p.params))
    mu_g = c + np.exp(X.dot(b))
    plt.subplot(2, 1, 1)
    plt.plot(VRData[ind, 3], Y_p, '.', label='actual')
    plt.plot(VRData[ind, 3], mu_p, '.', label='estimation')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(VRData[ind, 3], Y_g, '.', label='actual')
    plt.plot(VRData[ind, 3], mu_g, '.', label='estimation')
    plt.legend()
    plt.show()

def compare_glm_models(cell_id, morphs, acc_rate, acc_count):
    #  Compare three different Gamma models and the real activity rates for data of the fixed cell_id and all trials
    #  of the same morph level.
    #  The three Gamma models are as follows:
    #  2-poly Gamma with no term c and log link (function sm.GLM function)
    #  2-poly Gamma with fixed c = 1800 (function fit_gamma_fixed_c)
    #  2-poly Gamma with alternating c (function fit_gamma_alt)
    #  cell_id: the fixed cell
    #  morphs: trials that are used
    #  acc_rate: activity rate of all cells and trials
    #  acc_count: spiking count of all cells and trials
    ind = np.where(np.isin(VRData[:, 20], morphs))[0]
    X = []
    X.append(VRData[ind, 3])
    X.append(np.power(VRData[ind, 3], 2))
    X = np.array(X)
    X = X.T
    X = sm.add_constant(X, prepend=True)
    Y = F[cell_id, ind]
    v=1
    model = sm.GLM(Y, X, family=sm.families.Gamma(sm.families.links.log))
    results_gamma = model.fit()
    [b_alt, c, ll_new, b_log, ll_log] = fit_gamma_fixed_c(Y, X, v, c=1800, num=100, alpha=1)
    print('c is {}'.format(c))
    [b_alt2, c2, ll_new2, b_log2, c_log, ll_log2] = fit_gamma_alt(Y, X, v, num=100, alpha=1, initial_c=1000)
    print('value that I computed for b with fixed c is {}'.format(b_alt))
    print('value that I computed for b without fixed c is {}'.format(b_alt2))
    print(results_gamma.summary())
    mu_alt = np.exp(X.dot(b_alt)) + c
    mu_alt2 = np.exp(X.dot(b_alt2)) + c2
    mu_irls = np.exp(X.dot(results_gamma.params))
    print('likelihood for fixed c is{}'.format(v * np.sum(-1 * Y / mu_alt - np.log(mu_alt))))
    print('likelihood for not fixed c is{}'.format(v * np.sum(-1 * Y / mu_alt2 - np.log(mu_alt2))))
    print('likelihood for Gamma-IRLS is{}'.format(v * np.sum(-1 * Y / mu_irls - np.log(mu_irls))))
    print('value computed for c is {}'.format(c2))
    plt.subplot(2, 1, 1)
    plt.plot(VRData[ind, 3], Y, '.', label='true values')
    plt.plot(VRData[ind, 3], mu_alt, '.', label='gamma - fixed c')
    plt.plot(VRData[ind, 3], mu_alt2, '.', label='gamma - not fixed c')
    plt.plot(VRData[ind, 3], mu_irls, '.', label='gamma - irls')
    plt.legend()
    plt.title('Fitted models')
    plt.subplot(2, 1, 2)
    print(acc_rate.shape)
    plt.plot(np.mean(acc_rate[cell_id, :, :], axis=1), label='mean activity rate')
    plt.plot(np.sum(acc_count[cell_id, :, :], axis=1), label='count of data points')
    plt.title('Mean activity')
    plt.legend()
    plt.show()

def compare_gamma_for_diff_morphs(cell_id):
    # For a fixed cell, compares the 2-poly alt gamma fits for all data from morph0 and morph 1
    # This should give a sense of how different is the behaviour of a cell for morph0 and morph1, and how much our
    # fitted models show this.
    # cell_id: the fixed cell

    # MORPH = 0
    ind = np.where(np.isin(VRData[:, 20], morph_0_trials))[0]
    X = []
    X.append(VRData[ind, 3])
    X.append(np.power(VRData[ind, 3], 2))
    X = np.array(X)
    X = X.T
    X = sm.add_constant(X, prepend=True)
    Y = F[cell_id, ind]
    # Best one among different learning rates is chosen (different learning rates can result in very different solutions)
    [b1, c1, ll1, b_log1, c_log1, ll_log1] = fit_gamma_alt(Y, X, v=1, num=500, alpha=1, initial_c=1000)
    [b2, c2, ll2, b_log2, c_log2, ll_log2] = fit_gamma_alt(Y, X, v=1, num=500, alpha=.1, initial_c=1000)
    [b3, c3, ll3, b_log3, c_log3, ll_log3] = fit_gamma_alt(Y, X, v=1, num=500, alpha=10, initial_c=1000)
    if ll1 >= max(ll2, ll3):
        [b, c, ll, b_log, c_log, ll_log] = [b1, c1, ll1, b_log1, c_log1, ll_log1]
    if ll2 >= max(ll1, ll3):
        [b, c, ll, b_log, c_log, ll_log] = [b2, c2, ll2, b_log2, c_log2, ll_log2]
    if ll3 >= max(ll1, ll2):
        [b, c, ll, b_log, c_log, ll_log] = [b3, c3, ll3, b_log3, c_log3, ll_log3]
    print('Morph = 0, c = {}, ll = {}'.format(c, ll))
    mu = c + np.exp(X.dot(b))
    dev = 2 * np.sum(-1 * np.log(Y / mu) + (Y - mu) / mu)
    breaks = 200
    step = (max_pos - min_pos) / breaks
    mean = []
    for i in range(breaks):
        ind0 = np.where(
            (VRData[:, 3] >= min_pos + i * step) & (VRData[:, 3] < min_pos + (i + 1) * step) & (VRData[:, 1] == 0))[0]
        mean.append(np.mean(F[cell_id, ind0]))
    plt.subplot(2, 1, 1)
    plt.plot(VRData[ind, 3], Y, '.', label='true values')
    plt.plot(np.arange(min_pos, max_pos - step, step), mean, label='mean activity rate')
    plt.plot(VRData[ind, 3], mu, '.', label='estimated values')
    plt.legend()

    # MORPH = 1
    ind = np.where(np.isin(VRData[:, 20], morph_1_trials))[0]
    X = []
    X.append(VRData[ind, 3])
    X.append(np.power(VRData[ind, 3], 2))
    X = np.array(X)
    X = X.T
    X = sm.add_constant(X, prepend=True)
    Y = F[cell_id, ind]
    # Best one among different learning rates is chosen (different learning rates can result in very different solutions)
    [b1, c1, ll1, b_log1, c_log1, ll_log1] = fit_gamma_alt(Y, X, v=1, num=500, alpha=1, initial_c=1000)
    [b2, c2, ll2, b_log2, c_log2, ll_log2] = fit_gamma_alt(Y, X, v=1, num=500, alpha=.1, initial_c=1000)
    [b3, c3, ll3, b_log3, c_log3, ll_log3] = fit_gamma_alt(Y, X, v=1, num=500, alpha=10, initial_c=1000)
    if ll1 >= max(ll2, ll3):
        [b, c, ll, b_log, c_log, ll_log] = [b1, c1, ll1, b_log1, c_log1, ll_log1]
    if ll2 >= max(ll1, ll3):
        [b, c, ll, b_log, c_log, ll_log] = [b2, c2, ll2, b_log2, c_log2, ll_log2]
    if ll3 >= max(ll1, ll2):
        [b, c, ll, b_log, c_log, ll_log] = [b3, c3, ll3, b_log3, c_log3, ll_log3]
    print('Morph = 1, c = {}, ll = {}'.format(c, ll))
    mu = c + np.exp(X.dot(b))
    dev = 2 * np.sum(-1 * np.log(Y / mu) + (Y - mu) / mu)
    breaks = 200
    step = (max_pos - min_pos) / breaks
    mean = []
    for i in range(breaks):
        ind0 = np.where(
            (VRData[:, 3] >= min_pos + i * step) & (VRData[:, 3] < min_pos + (i + 1) * step) & (VRData[:, 1] == 1))[0]
        mean.append(np.mean(F[cell_id, ind0]))
    plt.subplot(2, 1, 2)
    plt.plot(VRData[ind, 3], Y, '.', label='true values')
    plt.plot(np.arange(min_pos, max_pos - step, step), mean, label='mean activity rate')
    plt.plot(VRData[ind, 3], mu, '.', label='estimated values')
    plt.legend()
    plt.show()

def show_activity_per_morphs(cell_id):
    #  Show imagesc plots of activity of a fixed cell for all morph levels plus a plot that shows the mean activity
    #  (over position) of trials with different morph levels.
    #  cell_id: id of the fixed cell
    min0 = np.min(activity_rates_morph_0[cell_id, :, morph_0_trials])
    mind25 = np.min(activity_rates_morph_d25[cell_id, :, morph_d25_trials])
    mind50 = np.min(activity_rates_morph_d50[cell_id, :, morph_d50_trials])
    mind75 = np.min(activity_rates_morph_d75[cell_id, :, morph_d75_trials])
    min1 = np.min(activity_rates_morph_1[cell_id, :, morph_1_trials])
    vmin = min(min0, mind25, mind50, mind75, min1)
    max0 = np.max(activity_rates_morph_0[cell_id, :, morph_0_trials])
    maxd25 = np.max(activity_rates_morph_d25[cell_id, :, morph_d25_trials])
    maxd50 = np.max(activity_rates_morph_d50[cell_id, :, morph_d50_trials])
    maxd75 = np.max(activity_rates_morph_d75[cell_id, :, morph_d75_trials])
    max1 = np.max(activity_rates_morph_1[cell_id, :, morph_1_trials])
    vmax = min(max0, maxd25, maxd50, maxd75, max1)

    plt.subplot(6, 1, 1)
    plt.imshow(activity_rates_morph_0[cell_id, :, morph_0_trials], aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.ylabel('m=0')
    plt.title('Activity Rate Images')

    plt.subplot(6, 1, 2)
    plt.imshow(activity_rates_morph_d25[cell_id, :, morph_d25_trials], aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.ylabel('m=.25')

    plt.subplot(6, 1, 3)
    plt.imshow(activity_rates_morph_d50[cell_id, :, morph_d50_trials], aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.ylabel('m=.5')

    plt.subplot(6, 1, 4)
    plt.imshow(activity_rates_morph_d75[cell_id, :, morph_d75_trials], aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.ylabel('m=.75')

    plt.subplot(6, 1, 5)
    plt.imshow(activity_rates_morph_1[cell_id, :, morph_1_trials], aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.ylabel('m=1')
    plt.xlabel('position')

    plt.subplot(6, 1, 6)
    plt.plot(np.mean(activity_rates_morph_0[cell_id, :, morph_0_trials], axis=0), label='morph = 0')
    plt.plot(np.mean(activity_rates_morph_d25[cell_id, :, morph_d25_trials], axis=0), label='morph = 0.25')
    plt.plot(np.mean(activity_rates_morph_d50[cell_id, :, morph_d50_trials], axis=0), label='morph = 0.5')
    plt.plot(np.mean(activity_rates_morph_d75[cell_id, :, morph_d75_trials], axis=0), label='morph = 0.75')
    plt.plot(np.mean(activity_rates_morph_1[cell_id, :, morph_1_trials], axis=0), label='morph = 1')
    plt.legend(loc='upper left')
    plt.show()

def compute_spline_mat(X):
    #  Computes the cubic spline matrix for a "1-dimensional" input X with knots described as below. This has been used
    #  for cases where X represents the position, and may need a more precise examination before using for other X's.
    #  X: 1-dimensional input
    min_val = np.min(X)
    max_val = np.max(X)
    knots = np.array([min_val - 10] + list(np.arange(min_val, max_val+50, 50)) + [max_val + 10])

    par = 0.5
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
        B = np.array([[-par, 2-par/l, par-2, par/l], [2*par, par/l-3, 3-2*par, -par/l], [-par, 0, par, 0], [0, 1, 0, 0]])
        r = A.dot(B)
        r = np.reshape(r, newshape=(1, r.shape[0]))
        out[i, nearest_knot_ind-1: nearest_knot_ind+3] = r
    # print('knots: {}'.format(knots))
    # print('shape of out:{}'.format(out.shape))
    # print('shape of knots:{}'.format(knots.shape))
    # out2 = out.dot(knots)
    # plt.plot(X)
    # plt.show()
    # plt.plot(out2)
    # plt.show()

    return out

def compute_active_trials(exp_id, morphs, morph_lvl):
    #  For each cell_id, for each trial in morphs compute a single value by averaging over the top 15 activity points.
    #  Then use K-means to divide trials into 2 groups: active and inactive
    #  exp_id: experiment id
    #  mophs: trials with the fixed morph level
    #  morph_lvl: fixed morph level
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
        top_act_org = np.copy(top_act)  # storing original version of activity somewhere
        if np.min(top_act) < thrsh:
            top_act[top_act > thrsh] = thrsh
        top_act = np.reshape(top_act, newshape=[top_act.shape[0], 1])
        dots = np.append(np.zeros(shape=top_act.shape), top_act, axis=1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(dots)
        x = np.argmax(dots[:, 1])
        if kmeans.labels_[x] == 0:
            kmeans.labels_ = 1 - kmeans.labels_
        active_trials[cell_id, :] = kmeans.labels_
        '''
        cols = [blue1, orange1]
        print(np.sum(active_trials[cell_id, :]))
        plt.plot(np.sort(top_act_org), '.')
        plt.show()
        for i in range(len(morphs)):
            tr_id = morphs[i]
            ind = np.where(VRData[:, 20] == tr_id)[0]
            plt.plot(VRData[ind, 3], F[cell_id, ind], color=cols[active_trials[cell_id, i].astype(int)])
        plt.show()
        '''
    np.save(os.getcwd() + '/Data/active_trials_morph_' + str(morph_lvl) + '_exp_' + str(exp_id) + '.npy', active_trials)
    np.save(os.getcwd() + '/Data/top_activity_trials_morph_' + str(morph_lvl) + '_exp_' + str(exp_id) + '.npy', activity_trials)

def show_active_inactive_trials(cell_id, morph_lvl, morphs, active_trials):
    # For a fixed cell and morph level, shows the acitivity of that cell during trials of that morph level, and further,
    # shows which trials are considered active and which are considered inactive
    # cell_id: fixed cell
    # morph_lvl: fixed morph level
    # mophs: trials with that fixed morph level
    # active_trials: indicates trials of that morph level that are considered active
    plt.subplot(2, 2, 1)
    for i in range(len(morphs)):
        tr_id = morphs[i]
        ind = np.where(VRData[:, 20] == tr_id)[0]
        plt.plot(VRData[ind, 3], F[cell_id, ind])
    plt.title('cell_id = {}, morph = {}'.format(cell_id, morph_lvl))
    plt.subplot(2, 2, 2)
    for i in range(len(morphs)):
        tr_id = morphs[i]
        ind = np.where(VRData[:, 20] == tr_id)[0]
        if active_trials[cell_id, i] == 1:
            plt.plot(VRData[ind, 3], F[cell_id, ind], 'r')
        else:
            plt.plot(VRData[ind, 3], F[cell_id, ind], 'b')
    plt.subplot(2, 2, 3)
    for i in range(len(morphs)):
        tr_id = morphs[i]
        ind = np.where(VRData[:, 20] == tr_id)[0]
        if active_trials[cell_id, i] == 1:
            plt.plot(VRData[ind, 3], F[cell_id, ind], 'r')
    plt.subplot(2, 2, 4)
    for i in range(len(morphs)):
        tr_id = morphs[i]
        ind = np.where(VRData[:, 20] == tr_id)[0]
        if active_trials[cell_id, i] == 0:
            plt.plot(VRData[ind, 3], F[cell_id, ind], 'b')
    plt.show()

def decode_morphs(exp_id, p, mode, visualize, visualize2, history):
    #  For each cell fit 8 spline-gamma models for morph0/morph1 active/inactive with_history/without_history.
    #  Second for each trial, compute its log-likelihood and likelihood based on its active/nonactive state point to
    #  point for both morph0 and morph1. Then use filter and smoother algorithms to compute prob. of decoding this trial
    #  as morph0 and morph1.
    #  The set of all cells and trials are defined below, num is a global variable.
    #  Visualize all of this for each cell and each trial if the variable visualize is y (yes).
    #  Return fitted values of all 8 models along with decoding results (log-likelihood, likelihood, filter, smoother)
    #  for every trial and cell in addition to position and actual activity.
    #  exp_id: experiment id
    #  p: probability of jumping form one state to the other one in filter and smoother algorithms.
    #  mode: shows if we use shorted version of data (mode = short) or all data (mode = all)
    #  visualize: determines if the function must show plots for each trial or not
    #  visualize2: determines if the function must show imshow for all trials for each cell or not
    #  history: determines if we want to visualize models with history component or not

    p_morph = np.empty(shape=[ncells, ntrials, 6], dtype=object)
    #  each row = [X_tr, Y_tr, p_morphs_filt, p_morph_smooth, p_morph_likelihood]
    goodness_of_fit = np.empty(shape=[ncells, ntrials, 2], dtype=object)

    gamma_fit_0_act = np.empty(shape=[ncells, 8], dtype=object)  # each row = [X, Y, Yhist, splinemat, mu, v, params]
    gamma_fit_0_inact = np.empty(shape=[ncells, 8], dtype=object)
    gamma_fit_1_act = np.empty(shape=[ncells, 8], dtype=object)
    gamma_fit_1_inact = np.empty(shape=[ncells, 8], dtype=object)
    gamma_fit_0_act_nohist = np.empty(shape=[ncells, 8], dtype=object)
    gamma_fit_0_inact_nohist = np.empty(shape=[ncells, 8], dtype=object)
    gamma_fit_1_act_nohist = np.empty(shape=[ncells, 8], dtype=object)
    gamma_fit_1_inact_nohist = np.empty(shape=[ncells, 8], dtype=object)
    gamma_fit_0_all = np.empty(shape=[ncells, 8], dtype=object)
    gamma_fit_1_all = np.empty(shape=[ncells, 8], dtype=object)

    trial_ids = range(ntrials)
    # trial_ids = morph_0_trials
    cnt = 1
    # cells_under_study = [75, 559, 220, 72, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for cell_id in cells_under_study:
        # cell_id = 75
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
        # dof = gamma_res_0_act.df_resid
        # chi2_stat = gamma_res_0_act.deviance/gamma_res_0_act.scale
        # p_value = 1 - scipy.stats.chi2.cdf(chi2_stat, dof)
        # print(p_value)

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
        # dof = gamma_res_0_inact.df_resid
        # chi2_stat = gamma_res_0_inact.deviance/gamma_res_0_inact.scale
        # p_value = 1 - scipy.stats.chi2.cdf(chi2_stat, dof)
        # print(p_value)

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

        # Fititng model with no history for visualization only
        spline_mat1_act = compute_spline_mat(X1_act)
        gamma_1_act = sm.GLM(Y1_act, spline_mat1_act, family=sm.families.Gamma(sm.families.links.identity))
        gamma_res_1_act = gamma_1_act.fit()
        mu_1_act_nohist = gamma_res_1_act.mu
        v_1_act_nohist = 1 / gamma_res_1_act.scale
        params_1_act_nohist = gamma_res_1_act.params
        # dof = gamma_res_1_act.df_resid
        # chi2_stat = gamma_res_1_act.deviance/gamma_res_1_act.scale
        # p_value = 1 - scipy.stats.chi2.cdf(chi2_stat, dof)
        # print(p_value)

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
        dof = gamma_res_1_inact.df_resid
        chi2_stat = gamma_res_1_inact.deviance / gamma_res_1_inact.scale
        p_value = 1 - scipy.stats.chi2.cdf(chi2_stat, dof)
        # print(gamma_res_1_inact.summary())
        # print(gamma_res_1_inact.scale)
        # print(chi2_stat)
        # print(p_value)

        # Fitting 1 Gamma model to all trials with morph 0 (for goodness-of-fit purposes)
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

        # Fitting 1 Gamma model to all trials with morph 1 (for goodness-of-fit purposes)
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

        print('morph 0 active trials = {}'.format(morph0_act.shape[0]))
        print('morph 0 inactive trials = {}'.format(morph0_inact.shape[0]))
        print('morph 1 active trials = {}'.format(morph1_act.shape[0]))
        print('morph 1 inactive trials = {}'.format(morph1_inact.shape[0]))

        # Computing Goodness-of-fit for multiple spatial maps: 2-Gamma vs. 1-Gamma
        for tr_id in trial_ids:
            # tr_id = 38
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
                # Maybe you should compute likelihood directly instead of finding it for the nearest point!

                #  Computing ll_act0 and ll_inact0
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
                # print('mu_0_inact_now = {}'.format(mu_0_inact_now))
                # print('v_0_inact = {}'.format(v_0_inact))
                # print('sd_0_inact_now = {}'.format(sd_0_inact_now))
                # print('Y_tr = {}'.format(Y_tr[i]))
                # print('ll_inact0[i] = {}'.format(ll_inact0[i]))
                # print('**************************')

                #  Computing ll_act1 and ll_inact1
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

                #  Computing ll_all0 and ll_all1
                ind0 = find_nearest(X0_all, [X_tr[i]])
                Y_hist_now = np.reshape(Y_hist_tr[i, :], newshape=[1, Y_hist_tr.shape[1]])
                inp = np.concatenate((spline_mat0_all[ind0, :], Y_hist_now), axis=1)
                mu_0_all_now = inp.dot(params_0_all)[0]
                mu_0_all_now = max(.1, mu_0_all_now)  # rarely estimated mu is negative which is not desired
                ll_all0[i] = -scipy.special.loggamma(v_0_all) + v_0_all * np.log(
                    v_0_all * Y_tr[i] / mu_0_all_now) - v_0_all * Y_tr[i] / mu_0_all_now - np.log(Y_tr[i])
                sd_0_all_now = mu_0_all_now / np.sqrt(v_0_all)

                ind0 = find_nearest(X1_all, [X_tr[i]])
                Y_hist_now = np.reshape(Y_hist_tr[i, :], newshape=[1, Y_hist_tr.shape[1]])
                inp = np.concatenate((spline_mat1_all[ind0, :], Y_hist_now), axis=1)
                mu_1_all_now = inp.dot(params_1_all)[0]
                mu_1_all_now = max(.1, mu_1_all_now)  # rarely estimated mu is negative which is not desired
                ll_all1[i] = -scipy.special.loggamma(v_1_all) + v_1_all * np.log(
                    v_1_all * Y_tr[i] / mu_1_all_now) - v_1_all * Y_tr[i] / mu_1_all_now - np.log(Y_tr[i])
                sd_1_all_now = mu_1_all_now / np.sqrt(v_1_all)

            # Assigning active/inactive to current trial based on activity of this trial among all trials with the
            # same morph level
            '''
            identified_activity0 = 'inactive'
            identified_activity1 = 'inactive'
            ll0 = ll_inact0
            ll1 = ll_inact1
            if active_trials_all[cell_id, tr_id] == 1:  #  Then this is an active trial for this cell_id
                ll0 = ll_act0
                ll1 = ll_act1
                identified_activity0 = 'active'
                identified_activity1 = 'active'
            '''

            # Assigning active/inactive based on activity of trials for morph0 and morph1 models
            # '''
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
            # '''

            goodness_of_fit[cell_id, tr_id, 0] = ll_all0
            goodness_of_fit[cell_id, tr_id, 1] = ll_all1
            '''
            if tr_id in morph_0_trials:
                # plt.plot(ll0, label='2-Gamma')
                # plt.plot(ll_all0, label='1-Gamma')
                # plt.legend()
                # plt.show()
                dof = cov0_all.shape[1]
                chi2_stat = -2*(np.sum(ll_all0) - np.sum(ll0))
                p_value = 1-scipy.stats.chi2.cdf(chi2_stat, dof)
                # print('tr_id = {}, ll0={:.2f}, ll_all0={:.2f}, p-value={:.2f}'.format(tr_id, np.sum(ll0), np.sum(ll_all0), p_value))
                # print('morph0, dof = {}, chi2_stat={:.2f}'.format(dof, chi2_stat))
                pvals.append(p_value)
            if tr_id in morph_1_trials:
                # plt.plot(ll1, label='2-Gamma')
                # plt.plot(ll_all1, label='1-Gamma')
                # plt.legend()
                # plt.show()
                dof = cov1_all.shape[1]
                chi2_stat = -2 * (np.sum(ll_all1) - np.sum(ll1))
                p_value = 1-scipy.stats.chi2.cdf(chi2_stat, dof)
                # print('morph1, dof = {}, chi2_stat={:.2f}'.format(dof, chi2_stat))
                # print('tr_id = {}, ll0={:.2f}, ll_all0={:.2f}, p-value={:.2f}'.format(tr_id, np.sum(ll0), np.sum(ll_all0), p_value))
                pvals.append(p_value)
            '''

            # Disabling points with small log-likelihood difference w.r.t the largest observed log-likelihood difference
            # This is a mean of putting more emphasize on high-activity regions (as likelihood change is
            # more dramatic in these areas), but it is still very ad-hoc
            # '''
            # thrsh = max(abs(ll0 - ll1)) / 10
            thrsh = 0
            for i in range(ll0.shape[0]):
                if abs(ll0[i] - ll1[i]) <= thrsh:
                    ll_min = min(ll0[i], ll1[i])
                    ll1[i] = ll_min
                    ll0[i] = ll_min

            L0 = np.exp(ll0)
            L1 = np.exp(ll1)
            # Normalizing Likelihood
            denom = L0 + L1
            L0 = L0/denom
            L1 = L1/denom

            ll_morph = np.array([ll0, ll1])  #  No normalization
            L_morph = np.array([L0, L1])  # Already normalized

            #  Running the filtering algorithm to compute probability of morph0 and morph1
            p_morph_filt = np.zeros(shape=[2, X_tr.shape[0]])
            for i in range(X_tr.shape[0]):
                if i == 0:
                    p_morph_filt[0, i] = L0[i]
                    p_morph_filt[1, i] = L1[i]
                if i > 0:
                    #  State transition model: to other state with prob. p
                    p_morph_filt[0, i] = L0[i] * ((1 - p) * p_morph_filt[0, i - 1] + p * p_morph_filt[1, i - 1])
                    p_morph_filt[1, i] = L1[i] * ((1 - p) * p_morph_filt[1, i - 1] + p * p_morph_filt[0, i - 1])
                p_morph_filt[:, i] = p_morph_filt[:, i] / np.sum(p_morph_filt[:, i])

            #  Running the smoother algorithm to compute probability of morph0 and morph1
            p_morph_smooth = np.zeros(shape=[2, X_tr.shape[0]])
            for i in range(X_tr.shape[0] - 1, -1, -1):
                if i == X_tr.shape[0] - 1:
                    p_morph_smooth[0, i] = p_morph_filt[0, i]
                    p_morph_smooth[1, i] = p_morph_filt[1, i]
                if i < X_tr.shape[0] - 1:
                    #  State transition model: to other state with prob. p
                    p_2step_0 = (1 - p) * p_morph_filt[0, i] + p * p_morph_filt[1, i]
                    p_2step_1 = (1 - p) * p_morph_filt[1, i] + p * p_morph_filt[0, i]
                    p_morph_smooth[0, i] = p_morph_filt[0, i] * (
                                (1 - p) * p_morph_smooth[0, i+1] / p_2step_0 + p * p_morph_smooth[1, i+1] / p_2step_1)
                    p_morph_smooth[1, i] = p_morph_filt[1, i]*(
                                (1 - p) * p_morph_smooth[1, i+1] / p_2step_1 + p * p_morph_smooth[0, i+1] / p_2step_0)

                    p_morph_smooth[:, i] = p_morph_smooth[:, i] / np.sum(p_morph_smooth[:, i])

            print('cell_id = {}, tri_id = {}'.format(cell_id, tr_id))
            p_morph[cell_id, tr_id, :] = [X_tr, Y_tr, p_morph_filt, p_morph_smooth, L_morph, ll_morph]

            if visualize:
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
                        plt.plot(X0_act, mu_0_act_nohist + 1.96 * mu_0_act_nohist/np.sqrt(v_0_act_nohist), '.', markersize=1)
                        plt.plot(X0_act, mu_0_act_nohist - 1.96 * mu_0_act_nohist/np.sqrt(v_0_act_nohist), '.', markersize=1)
                        plt.plot(X_tr, Y_tr, ',')
                        plt.ylabel('active')

                        plt.subplot(6, 2, 9)
                        plt.plot(X0_inact, mu_0_inact_nohist, '.', markersize=1)
                        plt.plot(X0_inact, mu_0_inact_nohist + 1.96 * mu_0_inact_nohist / np.sqrt(v_0_inact_nohist), '.', markersize=1)
                        plt.plot(X0_inact, mu_0_inact_nohist - 1.96 * mu_0_inact_nohist / np.sqrt(v_0_inact_nohist), '.', markersize=1)
                        plt.plot(X_tr, Y_tr, ',')
                        plt.ylabel('inactive')
                        plt.xlabel('morph = 0')

                        plt.subplot(6, 2, 8)
                        plt.plot(X1_act, mu_1_act_nohist, '.', markersize=1)
                        plt.plot(X1_act, mu_1_act_nohist + 1.96 * mu_1_act_nohist / np.sqrt(v_1_act_nohist), '.', markersize=1)
                        plt.plot(X1_act, mu_1_act_nohist - 1.96 * mu_1_act_nohist / np.sqrt(v_1_act_nohist), '.', markersize=1)
                        plt.plot(X_tr, Y_tr, ',')

                        plt.subplot(6, 2, 10)
                        plt.plot(X1_inact, mu_1_inact_nohist, '.', markersize=1)
                        plt.plot(X1_inact, mu_1_inact_nohist + 1.96 * mu_1_inact_nohist / np.sqrt(v_1_inact_nohist), '.', markersize=1)
                        plt.plot(X1_inact, mu_1_inact_nohist - 1.96 * mu_1_inact_nohist / np.sqrt(v_1_inact_nohist), '.', markersize=1)
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
    #  First each trial, use the log-likelihood values computed for cells before to compute a joint log-likelihood
    #  and use that to decode the time-to-time morph of the trial using filter and smoother algorithms.
    #  Return fitted values of all 8 models along with decoding results (log-likelihood, likelihood, filter, smoother)
    #  for every trial in addition to position and actual activity (of all cells) for that trial.
    #  exp_id: experiment id
    #  p: probability of jumping form one state to the other one in filter and smoother algorithms.
    #  mode: shows if we use shorted version of data (mode = short) or all data (mode = all)
    #  visualize: determines if the function must show plots or not

    p_morph_joint = np.empty(shape=[ntrials, 6], dtype=object)
    #  each row = [X_tr, Y_tr, p_morphs_filt, p_morph_smooth, p_morph_likelihood]
    trial_ids = range(ntrials)
    for tr_id in trial_ids:
        ind = np.where(VRData[:, 20] == tr_id)[0]
        X_tr = VRData[ind, 3]
        Y_tr = np.mean(F[:, ind], axis=0)
        # We have excluded first hist_window data points for fitting models (since we have history-dependent component)
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
                temp = p_morph[cell_id, tr_id, 5]  # this is log-likelihood computed previously
                ll0 += temp[0, i]
                ll1 += temp[1, i]
            ll_morph_joint[:, i] = [ll0, ll1]
            tmp = (ll0 + ll1) / 2
            ll0 = ll0 - tmp
            ll1 = ll1 - tmp
            L0 = np.exp(ll0)  # a constant times true L0 (for computational reasons)
            L1 = np.exp(ll1)  # a constant times true L0 (for computational reasons)
            if i == 0:
                p_morph_filt_joint[0, i] = L0
                p_morph_filt_joint[1, i] = L1
            if i > 0:
                #  State transition model: to other state with prob. p
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
        p_morph_joint[tr_id, :] = [X_tr, Y_tr, p_morph_filt_joint, p_morph_smooth_joint, L_morph_joint, ll_morph_joint]  # ll is normalized, L is normalized
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

def decode_morphs_joint_selected(exp_id, p, mode, visualize, visualize2, visualize3, visualize4, selected_trial, diff_trials, selected_cells):
    #  First each trial, use the log-likelihood values computed for cells before to compute a joint log-likelihood
    #  and use that to decode the time-to-time morph of the trial using filter and smoother algorithms.
    #  Return fitted values of all 8 models along with decoding results (log-likelihood, likelihood, filter, smoother)
    #  for every trial in addition to position and actual activity (of all cells) for that trial.
    #  exp_id: experiment id
    #  p: probability of jumping form one state to the other one in filter and smoother algorithms.
    #  mode: shows if we use shorted version of data (mode = short) or all data (mode = all)
    #  visualize: determines if the function must show plots or not for each trial
    #  visualize2: determines if the function must show an imshow for all trials together
    #  visualize3: determines if the function must show an imshow for contribution of different cells
    #  visualize4: determines if the function must show an imshow for contribution of different cells, sorted by the
    #  location of contribution (use for large populations)
    #  selected_trial: the trial that we want to see vote of different cells for
    #  diff_trial: the trial that we want to see vote of different cells for at population level and compare them after
    #  sorting cells based on the first trial of this list
    #  selected_cells: cells that we use to compute the joint decoding probability (this must be a subset of
    #  cells_under_study!
    #  Note: Be careful if you use both visualize3 and visualize4 -- prefer to not do so
    for cell_id in selected_cells:
        if cell_id not in cells_under_study:
            print('WARNING: THE SELECTED CELL IS NOT IN UNDER STUDY CELLS!')

    p_morph_joint_selected = np.empty(shape=[ntrials, 6], dtype=object)
    #  each row = [X_tr, Y_tr, p_morphs_filt, p_morph_smooth, p_morph_likelihood]
    trial_ids = range(ntrials)
    sorted_cells = []
    if visualize4:
        tr_id = diff_trials[0]
        ind = np.where(VRData[:, 20] == tr_id)[0]
        X_tr = VRData[ind, 3]
        Y_tr = np.mean(F[:, ind], axis=0)
        # We have excluded first hist_window data points for fitting models (since we have history-dependent component)
        X_tr = X_tr[hist_wind:]
        Y_tr = Y_tr[hist_wind:]
        p_morph_filt_joint = np.zeros(shape=[2, X_tr.shape[0]])
        L_morph_joint = np.zeros(shape=[2, X_tr.shape[0]])
        ll_morph_joint = np.zeros(shape=[2, X_tr.shape[0]])
        log_ll = np.zeros(shape=[len(selected_cells), X_tr.shape[0]])
        print('shape of log_ll = {}'.format(log_ll.shape))
        for i in range(X_tr.shape[0]):
            cnt = 0
            for cell_id in selected_cells:
                temp = p_morph[cell_id, tr_id, 5]  # this is log-likelihood computed previously
                log_ll[cnt, i] = temp[0, i] - temp[1, i]
                cnt += 1
        min_log_ll = np.argmin(log_ll, axis=1)
        selected_cells = selected_cells[np.argsort(min_log_ll)]

    for tr_id in trial_ids:
        print('tr_id = {}'.format(tr_id))
        ind = np.where(VRData[:, 20] == tr_id)[0]
        X_tr = VRData[ind, 3]
        Y_tr = np.mean(F[:, ind], axis=0)
        # We have excluded first hist_window data points for fitting models (since we have history-dependent component)
        X_tr = X_tr[hist_wind:]
        Y_tr = Y_tr[hist_wind:]
        p_morph_filt_joint = np.zeros(shape=[2, X_tr.shape[0]])
        L_morph_joint = np.zeros(shape=[2, X_tr.shape[0]])
        ll_morph_joint = np.zeros(shape=[2, X_tr.shape[0]])
        if visualize3 and tr_id == selected_trial:
            log_ll = np.zeros(shape=[len(selected_cells), X_tr.shape[0]])
            print('shape of log_ll = {}'.format(log_ll.shape))
            for i in range(X_tr.shape[0]):
                cnt = 0
                for cell_id in selected_cells:
                    temp = p_morph[cell_id, tr_id, 5]  # this is log-likelihood computed previously
                    log_ll[cnt, i] = temp[0, i] - temp[1, i]
                    cnt += 1
        if visualize4 and tr_id in diff_trials:
            log_ll = np.zeros(shape=[len(selected_cells), X_tr.shape[0]])
            print('shape of log_ll = {}'.format(log_ll.shape))
            for i in range(X_tr.shape[0]):
                cnt = 0
                for cell_id in selected_cells:
                    temp = p_morph[cell_id, tr_id, 5]  # this is log-likelihood computed previously
                    log_ll[cnt, i] = temp[0, i] - temp[1, i]
                    cnt += 1

        for i in range(X_tr.shape[0]):
            ll0 = 0
            ll1 = 0
            for cell_id in selected_cells:
                # print('tr_id = {}, cell_id = {}'.format(tr_id, cell_id))
                temp = p_morph[cell_id, tr_id, 5]  # this is log-likelihood computed previously
                ll0 += temp[0, i]
                ll1 += temp[1, i]
            ll_morph_joint[:, i] = [ll0, ll1]
            tmp = (ll0 + ll1) / 2
            ll0 = ll0 - tmp
            ll1 = ll1 - tmp
            L0 = np.exp(ll0)  # a constant times true L0 (for computational reasons)
            L1 = np.exp(ll1)  # a constant times true L0 (for computational reasons)
            if i == 0:
                p_morph_filt_joint[0, i] = L0
                p_morph_filt_joint[1, i] = L1
            if i > 0:
                #  State transition model: to other state with prob. p
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

        if visualize3 and tr_id == selected_trial:
            plt.subplot(2, 1, 1)
            plt.plot(X_tr, p_morph_smooth_joint[0, :], linewidth=3, color='purple')
            plt.xlim(np.min(X_tr), np.max(X_tr))
            plt.ylim([-.1, 1.1])
            # plt.yticks([0, 1], [])
            plt.yticks([])
            plt.xticks([])
            print('haha')
            plt.subplot(2, 1, 2)
            plt.imshow(log_ll, cmap='coolwarm', aspect='auto', vmin=-.5, vmax=.5)
            plt.yticks([])
            plt.xticks(np.arange(0, log_ll.shape[1], int(log_ll.shape[1] / 3)), [])
            # plt.colorbar()
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
            # plt.colorbar()
            plt.show()

    if visualize2:
        all_morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
        # all_morphs = [morph_d25_trials, morph_d50_trials, morph_d75_trials]
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
        # brk = np.argmin(np.diff(avgs))
        if exp_id == 3:
            avgs = gaussian_filter(avgs, sigma=.5)
            # avgs[0: brk] = gaussian_filter(avgs[0: brk], sigma=3)
            # avgs[brk+1: len(avgs)] = gaussian_filter(avgs[brk+1: len(avgs)], sigma=3)
            # avgs = gaussian_filter(avgs, sigma=1)
        if exp_id == 4:
            avgs = gaussian_filter(avgs, sigma=.5)
        plt.plot(X_avg, avgs, linewidth=3, color='purple')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    np.save(os.getcwd() + '/Data/' + mode + '_p_morph_joint_selected_exp_' + str(exp_id) + '.npy', p_morph_joint_selected)

def compute_cells_dec_perf(exp_id, cells_under_study, mode):
    #  Computes the decoding performance of each cell for each trial that belongs to morph0 or morph1
    #  It contains the average likelihood computed for trials within each morph level, as well as hit rate in decoding
    #  exp_id: experiment id
    #  cells_under_study: cells for wich we are computing the decoding performance (related to mode)
    #  mode: indicates if we are working with a small subset of cells (mode = short) or all cells (mode = all)

    cells_dec_perf = np.zeros(shape=[ncells, 5])  # Each row: [cell_id, avg_ll0, hit_rate0, avg_ll1, hit_rate1]
    for cell_id in cells_under_study:
        # print(cell_id)
        cells_dec_perf[cell_id, 0] = cell_id
        # Computing performance for morph0
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
        # Computing performance for morph1
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

def show_decode_results(cell_id, morphs, history):
    # For a fixed cell and any trial within a morph group the instantaneous decoding results are shown
    # cell_id: the fixed cell
    # morphs: trials within the fixed morph
    # history: only for visualization determines if we want to look at history-dependent models or not

    print('cell id = {}'.format(cell_id))

    # For nohist models 3nd and 4th are repetitions of columns 1 and 3 and are meaningless, for the sake of indexing.
    [X0_act, Y0_act, Y0hist_act, spline_mat0_act, cov0_act, mu_0_act, v_0_act, params_0_act] = \
        gamma_fit_0_act[cell_id, :]
    [X0_act_nohist, Y0_act_nohist, Y0hist_act_nohist, spline_mat0_act_nohist, cov0_act_nohist, mu_0_act_nohist,
     v_0_act_nohist, params_0_act_nohist] = gamma_fit_0_act_nohist[cell_id, :]
    [X0_inact, Y0_inact, Y0hist_inact, spline_mat0_inact, cov0_inact, mu_0_inact, v_0_inact, params_0_inact] = \
        gamma_fit_0_inact[cell_id, :]
    [X0_inact_nohist, Y0_inact_nohist, Y0hist_inact_nohist, spline_mat0_inact_nohist, cov0_inact_nohist,
     mu_0_inact_nohist, v_0_inact_nohist, params_0_inact_nohist] = gamma_fit_0_inact_nohist[cell_id, :]

    [X1_act, Y1_act, Y1hist_act, spline_mat1_act, cov1_act, mu_1_act, v_1_act, params_1_act] = \
        gamma_fit_1_act[cell_id, :]
    [X1_act_nohist, Y1_act_nohist, Y1hist_act_nohist, spline_mat1_act_nohist, cov1_act_nohist, mu_1_act_nohist,
     v_1_act_nohist, params_1_act_nohist] = gamma_fit_1_act_nohist[cell_id, :]
    [X1_inact, Y1_inact, Y1hist_inact, spline_mat1_inact, cov1_inact, mu_1_inact, v_1_inact, params_1_inact] = \
        gamma_fit_1_inact[cell_id, :]
    [X1_inact_nohist, Y1_inact_nohist, Y1hist_inact_nohist, spline_mat1_inact_nohist, cov1_inact_nohist,
     mu_1_inact_nohist, v_1_inact_nohist, params_1_inact_nohist] = gamma_fit_1_inact_nohist[cell_id, :]

    ind = np.where(active_trials_0[cell_id, :] == 1)[0]
    morph0_act = morph_0_trials[ind]
    ind = np.where(active_trials_0[cell_id, :] == 0)[0]
    morph0_inact = morph_0_trials[ind]
    ind = np.where(active_trials_1[cell_id, :] == 1)[0]
    morph1_act = morph_1_trials[ind]
    ind = np.where(active_trials_1[cell_id, :] == 0)[0]
    morph1_inact = morph_1_trials[ind]
    for tr_id in morphs:
        X_tr = p_morph[cell_id, tr_id, 0]
        Y_tr = p_morph[cell_id, tr_id, 1]
        p_morph_filt = p_morph[cell_id, tr_id, 2]
        p_morph_smooth = p_morph[cell_id, tr_id, 3]
        L_morph = p_morph[cell_id, tr_id, 4]
        ll_morph = p_morph[cell_id, tr_id, 5]

        if np.sum(ll_morph[0, :]) > np.sum(ll_morph[1, :]):
            print('tr_id = {}, Decoded result: morph 0'.format(tr_id))
        else:
            print('tr_id = {}, Decoded result: morph 1'.format(tr_id))

        plt.subplot(6, 2, 1)
        plt.plot(X_tr, L_morph[0, :], color=orange1, label='morph = 0')
        plt.plot(X_tr, L_morph[1, :], color=blue1, label='morph = 1')
        plt.plot(X_tr, np.zeros(X_tr.shape), color='gray')

        plt.title('morph 0')
        plt.ylabel('log-like')
        plt.legend()
        plt.subplot(6, 2, 2)
        plt.plot(X_tr, L_morph[0, :], color=orange1, label='morph = 0')
        plt.plot(X_tr, L_morph[1, :], color=blue1, label='morph = 1')
        plt.title('morph 1')

        plt.subplot(6, 2, 3)
        plt.plot(X_tr, p_morph_filt[0, :], color=orange1, label='morph = 0')
        plt.plot(X_tr, p_morph_filt[1, :], color=blue1, label='morph = 1')

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
            plt.plot(X0_act, mu_0_act_nohist + 1.96 * mu_0_act_nohist / np.sqrt(v_0_act_nohist), '.', markersize=1)
            plt.plot(X0_act, mu_0_act_nohist - 1.96 * mu_0_act_nohist / np.sqrt(v_0_act_nohist), '.', markersize=1)
            plt.plot(X_tr, Y_tr, ',')
            plt.ylabel('active')

            plt.subplot(6, 2, 9)
            plt.plot(X0_inact, mu_0_inact_nohist, '.', markersize=1)
            plt.plot(X0_inact, mu_0_inact_nohist + 1.96 * mu_0_inact_nohist / np.sqrt(v_0_inact_nohist), '.',
                     markersize=1)
            plt.plot(X0_inact, mu_0_inact_nohist - 1.96 * mu_0_inact_nohist / np.sqrt(v_0_inact_nohist), '.',
                     markersize=1)
            plt.plot(X_tr, Y_tr, ',')
            plt.ylabel('inactive')
            plt.xlabel('morph = 0')

            plt.subplot(6, 2, 8)
            plt.plot(X1_act, mu_1_act_nohist, '.', markersize=1)
            plt.plot(X1_act, mu_1_act_nohist + 1.96 * mu_1_act_nohist / np.sqrt(v_1_act_nohist), '.', markersize=1)
            plt.plot(X1_act, mu_1_act_nohist - 1.96 * mu_1_act_nohist / np.sqrt(v_1_act_nohist), '.', markersize=1)
            plt.plot(X_tr, Y_tr, ',')

            plt.subplot(6, 2, 10)
            plt.plot(X1_inact, mu_1_inact_nohist, '.', markersize=1)
            plt.plot(X1_inact, mu_1_inact_nohist + 1.96 * mu_1_inact_nohist / np.sqrt(v_1_inact_nohist), '.',
                     markersize=1)
            plt.plot(X1_inact, mu_1_inact_nohist - 1.96 * mu_1_inact_nohist / np.sqrt(v_1_inact_nohist), '.',
                     markersize=1)
            plt.plot(X_tr, Y_tr, ',')
            plt.xlabel('morph = 1')
        else:
            plt.subplot(6, 2, 7)
            plt.plot(X0_act, mu_0_act, '.', markersize=1)
            plt.plot(X0_act, mu_0_act + 1.96 * mu_0_act / np.sqrt(v_0_act), '.', markersize=1)
            plt.plot(X0_act, mu_0_act - 1.96 * mu_0_act / np.sqrt(v_0_act), '.', markersize=1)
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
        plt.ylabel('loglike \n diff')
        plt.xlabel('position')

        plt.subplot(6, 2, 12)
        plt.plot(X_tr, ll_morph[0, :] - ll_morph[1, :], color='m')
        plt.plot(X_tr, np.zeros(X_tr.shape), color='gray')
        plt.xlabel('position')
        plt.show()
        # return

def show_encode_results(cell_id, history):
    # For a fixed cell and any trial within a morph group the instantaneous decoding results are shown
    # cell_id: the fixed cell
    # morphs: trials within the fixed morph
    # history: only for visualization determines if we want to look at history-dependent models or not

    print('cell id = {}'.format(cell_id))

    # For nohist models 3nd and 4th are repetitions of columns 1 and 3 and are meaningless, for the sake of indexing.
    [X0_act, Y0_act, Y0hist_act, spline_mat0_act, cov0_act, mu_0_act, v_0_act, params_0_act] = \
        gamma_fit_0_act[cell_id, :]
    [X0_act_nohist, Y0_act_nohist, Y0hist_act_nohist, spline_mat0_act_nohist, cov0_act_nohist, mu_0_act_nohist,
     v_0_act_nohist, params_0_act_nohist] = gamma_fit_0_act_nohist[cell_id, :]
    [X0_inact, Y0_inact, Y0hist_inact, spline_mat0_inact, cov0_inact, mu_0_inact, v_0_inact, params_0_inact] = \
        gamma_fit_0_inact[cell_id, :]
    [X0_inact_nohist, Y0_inact_nohist, Y0hist_inact_nohist, spline_mat0_inact_nohist, cov0_inact_nohist,
     mu_0_inact_nohist, v_0_inact_nohist, params_0_inact_nohist] = gamma_fit_0_inact_nohist[cell_id, :]

    [X1_act, Y1_act, Y1hist_act, spline_mat1_act, cov1_act, mu_1_act, v_1_act, params_1_act] = \
        gamma_fit_1_act[cell_id, :]
    [X1_act_nohist, Y1_act_nohist, Y1hist_act_nohist, spline_mat1_act_nohist, cov1_act_nohist, mu_1_act_nohist,
     v_1_act_nohist, params_1_act_nohist] = gamma_fit_1_act_nohist[cell_id, :]
    [X1_inact, Y1_inact, Y1hist_inact, spline_mat1_inact, cov1_inact, mu_1_inact, v_1_inact, params_1_inact] = \
        gamma_fit_1_inact[cell_id, :]
    [X1_inact_nohist, Y1_inact_nohist, Y1hist_inact_nohist, spline_mat1_inact_nohist, cov1_inact_nohist,
     mu_1_inact_nohist, v_1_inact_nohist, params_1_inact_nohist] = gamma_fit_1_inact_nohist[cell_id, :]

    ind = np.where(active_trials_0[cell_id, :] == 1)[0]
    morph0_act = morph_0_trials[ind]
    ind = np.where(active_trials_0[cell_id, :] == 0)[0]
    morph0_inact = morph_0_trials[ind]
    ind = np.where(active_trials_1[cell_id, :] == 1)[0]
    morph1_act = morph_1_trials[ind]
    ind = np.where(active_trials_1[cell_id, :] == 0)[0]
    morph1_inact = morph_1_trials[ind]

    # just for normalizing
    norm_fact = 0
    if cell_id == 224:
        norm_fact = 4300
    llim = 5000
    ulim = 17000

    tr_ids = range(ntrials)
    for tr_id in tr_ids:
        X_tr = p_morph[cell_id, tr_id, 0]
        Y_tr = p_morph[cell_id, tr_id, 1]
        p_morph_filt = p_morph[cell_id, tr_id, 2]
        p_morph_smooth = p_morph[cell_id, tr_id, 3]
        L_morph = p_morph[cell_id, tr_id, 4]
        ll_morph = p_morph[cell_id, tr_id, 5]

        if tr_id in morph0_act:
        # if active_trials_all[cell_id, tr_id] == 1 and morph_lvl == 0:
            plt.subplot(2, 2, 1)
            plt.plot(X0_act, mu_0_act_nohist, '.', markersize=1, color=blue1, label='mean')
            plt.plot(X0_act, mu_0_act_nohist + 1.96 * mu_0_act_nohist / np.sqrt(v_0_act_nohist), '.', markersize=1, color=orange1, label='conf. band')
            plt.plot(X0_act, mu_0_act_nohist - 1.96 * mu_0_act_nohist / np.sqrt(v_0_act_nohist), '.', markersize=1, color=orange1, label='conf. band')
            plt.plot(X_tr, Y_tr, 'r,', label='observation')
            # plt.ylabel('active')
            plt.ylim([llim, ulim])
            plt.yticks([])
            plt.xticks([])


        if tr_id in morph1_act:
        # if active_trials_all[cell_id, tr_id] == 1 and morph_lvl == 1:
            plt.subplot(2, 2, 2)
            plt.plot(X1_act, norm_fact + mu_1_act_nohist, '.', markersize=1, color=blue1, label='mean')
            plt.plot(X1_act, norm_fact + mu_1_act_nohist + 1.96 * mu_1_act_nohist / np.sqrt(v_1_act_nohist), '.', markersize=1, color=orange1, label='conf. band')
            plt.plot(X1_act, norm_fact + mu_1_act_nohist - 1.96 * mu_1_act_nohist / np.sqrt(v_1_act_nohist), '.', markersize=1, color=orange1, label='conf. band')
            plt.plot(X_tr, norm_fact + Y_tr, 'r,', label='observation')
            plt.ylim([llim, ulim])
            plt.yticks([])
            plt.xticks([])

        if tr_id in morph0_inact:
        # if active_trials_all[cell_id, tr_id] == 0 and morph_lvl == 0:
            plt.subplot(2, 2, 3)
            plt.plot(X0_inact, mu_0_inact_nohist, '.', markersize=1, color=blue1, label='mean')
            plt.plot(X0_inact, mu_0_inact_nohist + 1.96 * mu_0_inact_nohist / np.sqrt(v_0_inact_nohist), '.', markersize=1, color=orange1, label='conf. band')
            plt.plot(X0_inact, mu_0_inact_nohist - 1.96 * mu_0_inact_nohist / np.sqrt(v_0_inact_nohist), '.', markersize=1, color=orange1, label='conf. band')
            plt.plot(X_tr, Y_tr, 'r,', label='observation')
            # plt.xlabel('morph = 0')
            # plt.ylabel('active')
            plt.ylim([llim, ulim])
            plt.yticks([])
            plt.xticks([])

        if tr_id in morph1_inact:
        # if active_trials_all[cell_id, tr_id] == 0 and morph_lvl == 1:
            plt.subplot(2, 2, 4)
            plt.plot(X1_inact, mu_1_inact_nohist, '.', markersize=1, color=blue1, label='mean')
            plt.plot(X1_inact, mu_1_inact_nohist + 1.96 * mu_1_inact_nohist / np.sqrt(v_1_inact_nohist), '.', markersize=1, color=orange1, label='conf. band')
            plt.plot(X1_inact, mu_1_inact_nohist - 1.96 * mu_1_inact_nohist / np.sqrt(v_1_inact_nohist), '.', markersize=1, color=orange1, label='conf. band')
            plt.plot(X_tr, Y_tr, 'r,', label='observation')
            # plt.xlabel('morph = 1')
            plt.ylim([llim, ulim])
            plt.yticks([])
            plt.xticks([])
    # plt.legend()
    plt.show()

def show_decode_results_all_trials(exp_id, cell_ids, p_morph):
    #  For each cell in cell_ids show the decoding morph for all trials along position splitted by morph levels and
    #  active/inactive. We also show the top-activity of moph0 and morph1 trials for each cell.
    #  exp_id: experiment id
    #  cell_ids: the set of cells that we want to show this for
    #  p_morph: stores the decoding results for all cells and trials
    X_range = np.arange(np.min(VRData[:, 3]), np.max(VRData[:, 3]), 0.5)
    all_morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
    active_trials = [active_trials_0, active_trials_d25, active_trials_d50, active_trials_d75, active_trials_1]
    for cell_id in cell_ids:
        print('cell id = {}'.format(cell_id))
        # '''
        plt.subplot(1, 2, 1)
        plt.plot(np.sort(top_activity_trials_0[cell_id, :]), '.')
        plt.title('morph = 0')
        plt.subplot(1, 2, 2)
        plt.plot(np.sort(top_activity_trials_1[cell_id, :]), '.')
        plt.title('morph = 1')
        plt.show()
        # '''
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

def show_decode_results_1morph_trials(exp_id, cell_ids, morph_lvl, morphs, act_tr):
    #  For each cell in cell_ids show the decoding morph for all trials within a certain morph level along the position
    #  active/inactive.
    #  exp_id: experiment id
    #  cell_ids: the set of cells that we want to show this for
    #  morph_lvl: morph level
    #  morphs: trials with given morph level
    #  act_tr: indicates the active trials of the given morph_lvl
    X_range = np.arange(np.min(VRData[:, 3]), np.max(VRData[:, 3]), 0.5)
    for cell_id in cell_ids:
        print('cell id = {}'.format(cell_id))
        # '''
        tr_ids = morphs
        tr_ids = np.concatenate((tr_ids[act_tr[cell_id, :] == 1], tr_ids[act_tr[cell_id, :] == 0]), axis=0)
        P0 = np.zeros(shape=[tr_ids.shape[0], X_range.shape[0]])
        for i in range(tr_ids.shape[0]):
            tr_id = tr_ids[i]
            ind = find_nearest(p_morph[cell_id, tr_id, 0], X_range)
            pp = p_morph[cell_id, tr_id, 3]
            P0[i, :] = pp[0, ind]
        sh = np.concatenate((act_tr[cell_id, act_tr[cell_id, :] == 1], act_tr[cell_id, act_tr[cell_id, :] == 0]), axis=0)
        sh = np.reshape(sh, newshape=[sh.shape[0], 1])
        plt.subplot2grid((1, 10), (0, 0), colspan=1)
        plt.imshow(sh, aspect='auto', extent=[0, 0.5, sh.shape[0], 0])
        plt.ylabel('Trials \n (m={})'.format(morph_lvl))
        plt.title('active/ \n inactive')

        plt.subplot2grid((1, 10), (0, 2), colspan=9)
        plt.imshow(P0, aspect='auto')
        plt.colorbar()
        plt.xlabel('position')
        plt.title('P(morph = 0)')
        plt.show()

def show_joint_decode_results_all_trials():
    #  Shows the instantenous decoded morph computed form the joint model for all trials in one single imshow plot
    all_morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
    X_range = np.arange(np.min(VRData[:, 3]), np.max(VRData[:, 3]), 0.5)
    for j in range(5):
        trial_ids = all_morphs[j]
        P0 = np.zeros(shape=[trial_ids.shape[0], X_range.shape[0]])
        for i in range(trial_ids.shape[0]):
            tr_id = trial_ids[i]
            ind = find_nearest(p_morph_joint[tr_id, 0], X_range)
            pp = p_morph_joint[tr_id, 3]
            P0[i, :] = pp[0, ind]
        plt.subplot(5, 1, j + 1)
        plt.imshow(P0, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.ylabel('m = ' + str(j * 0.25))
        if j == 4:
            plt.xlabel('position')
        if j == 0:
            plt.title('P(morph = 0)')
    plt.show()

def show_joint_decode_results_single_trial(morphs):
    #  Shows the instantenous decoded morph computed form the joint model for all trials in one single imshow plot
    X_range = np.arange(np.min(VRData[:, 3]), np.max(VRData[:, 3]), 0.5)
    for tr_id in morphs:
        print(tr_id)
        ind = find_nearest(p_morph_joint[tr_id, 0], X_range)
        pp = p_morph_joint[tr_id, 3]
        plt.plot(gaussian_filter(pp[0, ind], sigma=2))
        plt.yticks([])
        plt.xticks([])
        plt.show()

def show_activity_one_cell(exp_id, cell_id):
    activities = [activity_rates_morph_0, activity_rates_morph_1]
    morphs = [morph_0_trials, morph_1_trials]
    # activities = [activity_rates_morph_0, activity_rates_morph_d25, activity_rates_morph_d50, activity_rates_morph_d75, activity_rates_morph_1]
    # morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
    print(cell_id)
    for i in range(len(morphs)):
        curr_activity = activities[i]
        m = morphs[i]
        plt.subplot(len(morphs), 1, i+1)
        mat = curr_activity[cell_id, :, m].copy()
        mat1 = np.array(range(len(m)))
        mat1 = np.reshape(mat1, newshape=[mat1.shape[0], 1])
        mat2 = np.max(mat, axis=1)
        if cell_id == 56:
            mat2 = np.argmax(mat, axis=1)
        mat2 = np.reshape(mat2, newshape=[mat2.shape[0], 1])
        mat3 = np.concatenate((mat1, mat2), axis=1)
        ind = np.argsort(mat3[:, 1])
        mat3 = mat3[ind, :]
        plotted_mat = mat[mat3[:, 0].astype(int), :]
        plt.imshow(plotted_mat, aspect='auto', vmin=0, vmax=12000)
        plt.colorbar()
    plt.show()

def show_activity_one_trial(exp_id, tr_ids, cell_ids):
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

        print(tr_id)
        mat = activity[cell_ids, :, tr_id]
        # sort cells based on avg activity
        avg_act = np.mean(mat, axis=1)
        ind = np.argsort(avg_act)
        mat2 = activity[ind, :, tr_id]
        # plt.imshow(mat2, aspect='auto', vmin=0, vmax=12000)
        # plt.colorbar()
        # plt.show()

        # sort cells based on place fields
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
        plt.imshow(plotted_mat, aspect='auto', vmin=0, vmax=12000)
        plt.colorbar()
        plt.show()

def show_similarity (exp_id, tr_ids, cell_ids):
    # Compute the cosine similarity between individual trials and average activity of trials in known environment for a
    # set of cells.
    # tr_ids: set of trials that we want to compute this similarity for
    # cell_ids: set of cells that are used

    arr0 = activity_rates_morph_0[cell_ids, :, :]
    arr0 = arr0[:, :, morph_0_trials]
    avg0 = np.mean(arr0, axis=2)
    arr1 = activity_rates_morph_1[cell_ids, :, :]
    arr1 = arr1[:, :, morph_1_trials]
    avg1 = np.mean(arr1, axis=2)

    # computing SF
    '''
    for tr_id in tr_ids:
        print(tr_id)
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

        SF = np.zeros(shape=[curr_act.shape[1], 1])
        for i in range(SF.shape[0]):
            SF[i] = cos(curr_act[:, i], avg0[:, i])/(cos(curr_act[:, i], avg0[:, i]) + cos(curr_act[:, i], avg1[:, i]))
        plt.plot(SF)
        plt.show()
    '''

    # Computing similarity based on distance for each trial and cell (one trial and all cells can be obtained)
    '''
    for tr_id in tr_ids:
        print(tr_id)
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
        for i in range(arr0.shape[0]):
            print('cell_id = {}'.format(cell_ids[i]))
            diff0 = np.zeros(shape=[arr0.shape[1], ])
            for tr0 in range(arr0.shape[2]):
                diff = np.abs(arr0[i, :, tr0] - curr_act[i, :])
                diff0 += diff
            diff1 = np.zeros(shape=[arr1.shape[1], ])
            for tr1 in range(arr1.shape[2]):
                diff = np.abs(arr1[i, :, tr1] - curr_act[i, :])
                diff1 += diff
            avg_diff0 = gaussian_filter(diff0/arr0.shape[2], sigma=2)
            avg_diff1 = gaussian_filter(diff1/arr1.shape[2], sigma=2)
            plt.plot(avg_diff0 / (avg_diff0 + avg_diff1))
            plt.show()
    '''

    # Computing similarity based on distance for one cell and all trials
    # '''
    # morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
    morphs = [morph_d25_trials, morph_d50_trials, morph_d75_trials]
    l = 3
    for i in range(arr0.shape[0]):
        print('ind = {}, cell_id = {}'.format(i, cell_ids[i]))
        for j in range(l):
            curr_morph = morphs[j]
            print('morph = {}'.format((j+1)*.25))
            plt.subplot(l, 1, j+1)
            for tr_id in curr_morph:
                # print(tr_id)
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
        plt.show()
    # '''

def show_multiple_maps_neurips(cell_ids):
    for cell_id in cell_ids:
        activity_rates_morph_0[cell_id, :, :] = activity_rates_morph_0[cell_id, :, :] / np.mean(activity_rates_morph_0[cell_id, :, :])
        activity_rates_morph_1[cell_id, :, :] = activity_rates_morph_1[cell_id, :, :] / np.mean(activity_rates_morph_1[cell_id, :, :])
    for j in range(len(cell_ids)):
        cell_id = cell_ids[j]
        activities = [activity_rates_morph_0, activity_rates_morph_1]
        morphs = [morph_0_trials, morph_1_trials]
        # activities = [activity_rates_morph_0, activity_rates_morph_d25, activity_rates_morph_d50, activity_rates_morph_d75, activity_rates_morph_1]
        # morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
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
            # maxx0 = np.mean(activity_rates_morph_0[cell_id, :, morph_0_trials])
            # maxx1 = np.mean(activity_rates_morph_1[cell_id, :, morph_1_trials])
            # maxx = (maxx0 + maxx1)/2
            maxx = 1
            plotted_mat = plotted_mat/maxx
            print(plotted_mat.shape)
            plt.imshow(plotted_mat, aspect='auto', vmin=0, vmax=6)
            # plt.imshow(plotted_mat, aspect='auto')
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

        print(tr_id)
        mat = activity[cell_ids, :, tr_id]
        # sort cells based on avg activity
        avg_act = np.mean(mat, axis=1)
        ind = np.argsort(avg_act)
        mat2 = activity[ind, :, tr_id]
        # plt.imshow(mat2, aspect='auto', vmin=0, vmax=12000)
        # plt.colorbar()
        # plt.show()

        # sort cells based on place fields
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
        print(plotted_mat.shape)
        # x = input()
        plt.imshow(plotted_mat, aspect='auto', vmin=0, vmax=6)
        # plt.colorbar()
        # plt.yticks([])
        if j > 0:
            plt.yticks([])
        if j == 0:
            plt.yticks(np.arange(0, plotted_mat.shape[0], int(plotted_mat.shape[0] / 3)-1), [])
        plt.xticks(np.arange(0, plotted_mat.shape[1], int(plotted_mat.shape[1] / 3)), [])
        # plt.colorbar()
        plt.show()

def show_fluctuations_neurips(exp_id, cell_ids, plot_cells):
    # Compute the cosine similarity between individual trials and average activity of trials in known environment for a
    # set of cells.
    # tr_ids: set of trials that we want to compute this similarity for
    # cell_ids: set of cells that are used

    arr0 = activity_rates_morph_0[cell_ids, :, :]
    arr0 = arr0[:, :, morph_0_trials]
    avg0 = np.mean(arr0, axis=2)
    arr1 = activity_rates_morph_1[cell_ids, :, :]
    arr1 = arr1[:, :, morph_1_trials]
    avg1 = np.mean(arr1, axis=2)

    # Computing similarity based on distance for one cell and all trials
    # '''
    # morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
    morphs = [morph_d25_trials, morph_d50_trials, morph_d75_trials]
    l = len(morphs)
    for cell_id in plot_cells:
        i = np.where(cell_ids == cell_id)[0]
        i = i[0]
        print('i = {}'.format(i))
        print('cell_id = {}'.format(cell_id))
        for j in range(l):
            curr_morph = morphs[j]
            print('morph = {}'.format((j+1)*.25))
            plt.subplot(l, 1, j + 1)
            for tr_id in curr_morph:
                # print(tr_id)
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
                # plt.plot(np.arange(1, len(avg_diff0)+1), .5*np.ones(shape=[len(avg_diff0), 1]))
                plt.ylim([.1, .9])
                plt.yticks([])
                if i == 16:
                    plt.yticks([.25, .5, .75], [])
                if j == l-1:
                    plt.xticks(np.arange(0, len(avg_diff0), int(len(avg_diff0) / 3)), [])
                else:
                    plt.xticks([])
        plt.show()

    # '''

def show_decoding_results_neurips(exp_id, cell_ids):
    #  For each cell in cell_ids show the decoding morph for all trials along position splitted by morph levels and
    #  active/inactive. We also show the top-activity of moph0 and morph1 trials for each cell.
    #  exp_id: experiment id
    #  cell_ids: the set of cells that we want to show this for
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
            # plt.colorbar()
            plt.yticks([])
            if j == l-1:
                plt.xticks(np.arange(0, X_range.shape[0], int(X_range.shape[0] / 3)), [])
            else:
                plt.xticks([])
        plt.show()

def compute_trans_prob(exp_id, p_range, cell_ids, trial_ids):
    #  For each cell, first fit  2-Gamma models for each original environment. Then for each p, decode the represented
    #  environment during ambiguous environments and compute the likelihood of each. Visualize these as a function of p
    #  and compute its MLE for p and its sd, and test hypothesis that p = 0.
    #  exp_id: experiment id
    #  p_range: a numpy array that has the range of values of p, probability of jumping form one state to the other one
    #  cell_ids: cells that we want to do this for
    #  trial_ids: trials that we want to do this for

    cnt = 1
    # cells_under_study = [75, 559, 220, 72, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
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
                #  Computing ll_act0 and ll_inact0
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

                #  Computing ll_act1 and ll_inact1
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

            # Assigning active/inactive to current trial based on activity of this trial among all trials with the
            # same morph level
            '''
            identified_activity0 = 'inactive'
            identified_activity1 = 'inactive'
            ll0 = ll_inact0
            ll1 = ll_inact1
            if active_trials_all[cell_id, tr_id] == 1:  #  Then this is an active trial for this cell_id
                ll0 = ll_act0
                ll1 = ll_act1
                identified_activity0 = 'active'
                identified_activity1 = 'active'
            '''

            # Assigning active/inactive based on activity of trials for morph0 and morph1 models
            # '''
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
            # '''

            L0 = np.exp(ll0)
            L1 = np.exp(ll1)
            # Normalizing Likelihood
            denom = L0 + L1
            L0 = L0/denom
            L1 = L1/denom

            ll_morph = np.array([ll0, ll1])  #  No normalization
            L_morph = np.array([L0, L1])  # Already normalized
            for j in range(p_range.shape[0]):
                p = p_range[j]
                #  Running the filtering algorithm to compute probability of morph0 and morph1
                p_morph_filt = np.zeros(shape=[2, X_tr.shape[0]])
                for i in range(X_tr.shape[0]):
                    if i == 0:
                        p_morph_filt[0, i] = L0[i]
                        p_morph_filt[1, i] = L1[i]
                    if i > 0:
                        #  State transition model: to other state with prob. p
                        p_morph_filt[0, i] = L0[i] * ((1 - p) * p_morph_filt[0, i - 1] + p * p_morph_filt[1, i - 1])
                        p_morph_filt[1, i] = L1[i] * ((1 - p) * p_morph_filt[1, i - 1] + p * p_morph_filt[0, i - 1])
                    p_morph_filt[:, i] = p_morph_filt[:, i] / np.sum(p_morph_filt[:, i])

                #  Running the smoother algorithm to compute probability of morph0 and morph1
                p_morph_smooth = np.zeros(shape=[2, X_tr.shape[0]])
                for i in range(X_tr.shape[0] - 1, -1, -1):
                    if i == X_tr.shape[0] - 1:
                        p_morph_smooth[0, i] = p_morph_filt[0, i]
                        p_morph_smooth[1, i] = p_morph_filt[1, i]
                    if i < X_tr.shape[0] - 1:
                        #  State transition model: to other state with prob. p
                        p_2step_0 = (1 - p) * p_morph_filt[0, i] + p * p_morph_filt[1, i]
                        p_2step_1 = (1 - p) * p_morph_filt[1, i] + p * p_morph_filt[0, i]
                        p_morph_smooth[0, i] = p_morph_filt[0, i] * (
                                    (1 - p) * p_morph_smooth[0, i+1] / p_2step_0 + p * p_morph_smooth[1, i+1] / p_2step_1)
                        p_morph_smooth[1, i] = p_morph_filt[1, i]*(
                                    (1 - p) * p_morph_smooth[1, i+1] / p_2step_1 + p * p_morph_smooth[0, i+1] / p_2step_0)

                        p_morph_smooth[:, i] = p_morph_smooth[:, i] / np.sum(p_morph_smooth[:, i])
                new_p_ll = 0
                print(ll0.shape)
                print(ll1.shape)
                print(X_tr.shape)
                new_p_ll = np.sum(np.log(p_morph_smooth[0, :]*L0 + p_morph_smooth[1, :]*L1))
                p_ll[cell_id, tr_id, j] = new_p_ll
            # plt.plot(p_range, p_ll[tr_id, :])
            # plt.show()
        # p_ll_agg = np.sum(p_ll[cell_id, trial_ids, :], axis=0)
        # p_L_agg = p_ll_agg - np.mean(p_ll_agg)
        # p_L_agg = np.exp(p_L_agg)
        # p_L_agg = p_L_agg/np.sum(p_L_agg)
        # ind = np.argmax(p_ll_agg)
        # print('the best p = {}'.format(p_range[ind]))
        # plt.plot(p_range, p_L_agg)
        # plt.show()
    np.save(os.getcwd() + '/Data/' + mode + '_trans_prob_ll_exp_' + str(exp_id) + '.npy', p_ll)

def show_trans_prob(exp_id, cell_ids, trial_ids):
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
        print(np.sum(p_L_agg))
        plt.plot(p_range, p_L_agg)
        plt.ylim([-0.01, 0.08])
        plt.xticks([])
        plt.xticks([0, 0.33, 0.66, 1], [])
        plt.yticks([])
        if cell_id == cell_ids[0]:
            plt.yticks([0, .035, .07], [])
        plt.show()

def assess_goodness_of_fit(exp_id, cell_ids):
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
        if cell_id == 32:
            print(p_value)
    plt.subplot(2, 1, 1)
    plt.plot(np.sort(p_vals0))
    plt.plot([0, len(p_vals0)], [.05, .05], color='grey')
    plt.show()

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
        if cell_id == 32:
            print(p_value)
    plt.subplot(2, 1, 2)
    plt.plot(np.sort(p_vals1))
    plt.plot([0, len(p_vals1)], [.05, .05], color='grey')
    plt.show()

def assess_goodness_of_fit2(exp_id, cell_ids):
    perf = []
    for cell_id in cell_ids:
        cnt = 0
        for tr_id in morph_0_trials:
            # using smoother decoded probabilities
            ll = np.sum(np.log(p_morph[cell_id, tr_id, 3]), axis=1)

            # using log-likelihoods probabilities
            # ll = np.sum(p_morph[cell_id, tr_id, 5], axis=1)

            ll = ll - np.min(ll)
            score = np.exp(ll[0])/(np.exp(ll[0]) + np.exp(ll[1]))
            if score >= .5:
                cnt += 1
        if cnt/morph_0_trials.shape[0] < 0.5:
            print('bad cell id ={}'.format(cell_id))
        print('cell_id = {}, accuracy = {}'.format(cell_id, cnt/morph_0_trials.shape[0]))
        perf.append(cnt/morph_0_trials.shape[0])
    perf = np.sort(perf)
    plt.plot(perf)
    plt.show()

    perf = []
    for tr_id in morph_1_trials:
        cnt = 0
        for cell_id in cell_ids:
            ll = np.sum(np.log(p_morph[cell_id, tr_id, 3]), axis=1)
            ll = ll - np.min(ll)
            score = np.exp(ll[1]) / (np.exp(ll[0]) + np.exp(ll[1]))
            if score >= .5:
                cnt += 1
        print('tr_id = {}, accuracy = {}'.format(tr_id, cnt/cell_ids.shape[0]))
        perf.append(cnt/cell_ids.shape[0])
    perf = np.sort(perf)
    plt.plot(perf)
    plt.show()

def assess_goodness_of_fit_decoding(exp_id, cell_ids, tr_ids):
    p_morph_1map = np.empty(shape=[ncells, ntrials, 6], dtype=object)
    cnt = 1
    for cell_id in cell_ids:
        print('cell number {}'.format(cnt))
        cnt += 1
        for tr_id in tr_ids:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            plt.show()
            Y_hist_tr = compute_hist_cov(Y_tr, hist_wind)
            X_tr = X_tr[hist_wind:]
            Y_tr = Y_tr[hist_wind:]
            p_morph_filt_1map = np.zeros(shape=[2, X_tr.shape[0]])
            # using likelihoods computed previously for only one spatial map
            ll0 = goodness_of_fit[cell_id, tr_id, 0]
            ll1 = goodness_of_fit[cell_id, tr_id, 1]
            L0 = np.exp(ll0)
            L1 = np.exp(ll1)
            # Normalizing Likelihood
            denom = L0 + L1
            L0 = L0 / denom
            L1 = L1 / denom
            ll_morph = np.array([ll0, ll1])  # No normalization
            L_morph = np.array([L0, L1])  # Already normalized
            for i in range(X_tr.shape[0]):
                if i == 0:
                    p_morph_filt_1map[0, i] = L0[i]
                    p_morph_filt_1map[1, i] = L1[i]
                if i > 0:
                    #  State transition model: to other state with prob. p
                    p_morph_filt_1map[0, i] = L0[i] * ((1 - p) * p_morph_filt_1map[0, i - 1] + p * p_morph_filt_1map[1, i - 1])
                    p_morph_filt_1map[1, i] = L1[i] * ((1 - p) * p_morph_filt_1map[1, i - 1] + p * p_morph_filt_1map[0, i - 1])
                p_morph_filt_1map[:, i] = p_morph_filt_1map[:, i] / np.sum(p_morph_filt_1map[:, i])

            #  Running the smoother algorithm to compute probability of morph0 and morph1
            p_morph_smooth_1map = np.zeros(shape=[2, X_tr.shape[0]])
            for i in range(X_tr.shape[0] - 1, -1, -1):
                if i == X_tr.shape[0] - 1:
                    p_morph_smooth_1map[0, i] = p_morph_filt_1map[0, i]
                    p_morph_smooth_1map[1, i] = p_morph_filt_1map[1, i]
                if i < X_tr.shape[0] - 1:
                    #  State transition model: to other state with prob. p
                    p_2step_0 = (1 - p) * p_morph_filt_1map[0, i] + p * p_morph_filt_1map[1, i]
                    p_2step_1 = (1 - p) * p_morph_filt_1map[1, i] + p * p_morph_filt_1map[0, i]
                    p_morph_smooth_1map[0, i] = p_morph_filt_1map[0, i] * (
                            (1 - p) * p_morph_smooth_1map[0, i + 1] / p_2step_0 + p * p_morph_smooth_1map[1, i + 1] / p_2step_1)
                    p_morph_smooth_1map[1, i] = p_morph_filt_1map[1, i] * (
                            (1 - p) * p_morph_smooth_1map[1, i + 1] / p_2step_1 + p * p_morph_smooth_1map[0, i + 1] / p_2step_0)

                    p_morph_smooth_1map[:, i] = p_morph_smooth_1map[:, i] / np.sum(p_morph_smooth_1map[:, i])
            # print(p_morph_smooth_1map.shape)
            # pm = p_morph[cell_id, tr_id, 3]
            # print(pm.shape)
            # plt.plot(p_morph_smooth_1map[0, :], label='1map')
            # plt.plot(pm[0, :], label='2maps')
            # plt.legend()
            # plt.show()
            # print('cell_id = {}, tri_id = {}'.format(cell_id, tr_id))
            p_morph_1map[cell_id, tr_id, :] = [X_tr, Y_tr, p_morph_filt_1map, p_morph_smooth_1map, L_morph, ll_morph]
    np.save(os.getcwd() + '/Data/' + mode + '_p_morph_1map_exp_' + str(exp_id) + '.npy', p_morph_1map)

def show_map_impact_on_decoding(exp_id, cell_ids, tr_ids):
    score_1map = []
    score_2map = []
    for cell_id in cell_ids:
        print('cell_id = {}'.format(cell_id))
        prob_1map = []
        prob_2map = []
        agg_prob_1map = []
        agg_prob_2map = []
        for tr_id in tr_ids:
            pm = p_morph[cell_id, tr_id, 3]
            pm_1map = p_morph_1map[cell_id, tr_id, 3]
            prob_1map.append(np.mean(pm_1map[0, :]))
            prob_2map.append(np.mean(pm[0, :]))
            # prob_1map += list(pm_1map[0, :])
            # prob_2map += list(pm[0, :])
            # ll = np.sum(np.log(p_morph[cell_id, tr_id, 3]), axis=1)
            # ll = ll - np.min(ll)
            # agg_prob_2map.append(np.exp(ll[1]) / (np.exp(ll[0]) + np.exp(ll[1])))
            # ll = np.sum(np.log(p_morph_1map[cell_id, tr_id, 3]), axis=1)
            # ll = ll - np.min(ll)
            # agg_prob_1map.append(np.exp(ll[1]) / (np.exp(ll[0]) + np.exp(ll[1])))
        # plt.hist(prob_1map, label='K_j = 1', bins=10, alpha=.7)
        # plt.hist(prob_2map, label='K_j = 2', bins=10, alpha=.7)
        # plt.xticks([])
        # plt.yticks([])
        # plt.ylim([0, 14])
        # print('1map score = {}'.format(np.mean(prob_1map)))
        # print('2map score = {}'.format(np.mean(prob_2map)))
        score_1map.append(np.mean(prob_1map))
        score_2map.append(np.mean(prob_2map))
        # plt.plot(agg_prob_1map, label='1map')
        # plt.plot(agg_prob_2map, label='2maps')
        # plt.legend()
        # plt.show()
    # plt.hist(score_1map, label='1map', bins = 20, alpha=.7)
    # plt.hist(score_2map, label='2map', bins = 20, alpha=.7)
    # plt.legend()
    # plt.show()
    plt.plot(0.1 + np.sort(np.array(score_2map) - np.array(score_1map)), label='Approach (1)')
    # plt.show()

def naive_approach(exp_id, cell_ids, tr_ids, morph_lvl):
    naive_all = []
    smooth_all = []
    for cell_id in cell_ids:
        print('cell_id = {}'.format(cell_id))
        naive = []
        smooth = []
        for tr_id in tr_ids:
            X_tr = p_morph[cell_id, tr_id, 0]
            p_smooth = p_morph[cell_id, tr_id, 3]
            L = p_morph[cell_id, tr_id, 4]
            naive.append(np.mean(L[morph_lvl, :]/(L[0, :] + L[1, :])))
            # naive.append(np.mean(gaussian_filter(L[0, :]/(L[0, :] + L[1, :]), sigma=20)))
            smooth.append(np.mean(p_smooth[morph_lvl, :]))
            # plt.plot(X_tr, p_smooth[0, :], color=blue1, label='smooth')
            # plt.plot(X_tr, gaussian_filter(L[0, :]/(L[0, :] + L[1, :]), sigma=5), color=orange1, label='naive')
            # plt.legend()
            # plt.show()
        # plt.show()
        print('naive score = {}'.format(np.mean(naive)))
        print('smooth score = {}'.format(np.mean(smooth)))
        # plt.plot(naive, label='naive')
        # plt.plot(smooth, label='smooth')
        # plt.hist(naive, label='naive', bins=10, alpha=.7)
        # plt.hist(smooth, label='smooth', bins=10, alpha=.7)
        # plt.legend()
        # plt.show()
        naive_all.append(np.mean(naive))
        smooth_all.append(np.mean(smooth))
    # plt.hist(naive_all, label='naive', bins=10, alpha=.7)
    # plt.hist(smooth_all, label='smooth', bins=10, alpha=.7)
    # plt.show()
    if morph_lvl == 0:
        plt.plot(np.sort(np.array(smooth_all) - np.array(naive_all)), label='Approach (2)')
    if morph_lvl == 1:
        plt.plot(np.sort(np.array(smooth_all) - np.array(naive_all)), label='Approach (3)')
    # plt.show()





##################################   ENCODING  ##############################################

# Defining colors:
blue1 = (75/255, 139/255, 190/255)
orange1 = (255/255, 112/255, 59/255)
mr1 = 'b'
mr2 = blue1
mr3 = 'purple'
mr4 = 'pink'
mr5 = 'r'

# Which experiment to work with?
# Exp1:
# Exp2:
# Exp3: This seems to be Rare experiment
# Exp4: This seems to be Frequent experiment
exp_id = 3

# Reading the output of preprocessing code
F = np.load(os.getcwd() + '/Data/suite2p' + str(exp_id) + '/plane0/F.npy')  # ROIs by timepoints
Fneu = np.load(os.getcwd() + '/Data/suite2p' + str(exp_id) + '/plane0/Fneu.npy')  # ROIs by timepoints
iscell = np.load(os.getcwd() + '/Data/suite2p' + str(exp_id) + '/plane0/iscell.npy')  # ROIs by 2
stat = np.load(os.getcwd() + '/Data/suite2p' + str(exp_id) + '/plane0/stat.npy', allow_pickle=True)
# ROIs by 1, each one is a dictionary
spks = np.load(os.getcwd() + '/Data/suite2p' + str(exp_id) + '/plane0/spks.npy')  # ROIs by timepoints
ops = np.load(os.getcwd() + '/Data/suite2p' + str(exp_id) + '/plane0/ops.npy', allow_pickle=True)  # dictionary
VRData = np.load(os.getcwd() + '/Data/VRData' + str(exp_id) + '.npy')  # timepoints by 20 features

# Data for exp 3 and 4 have one more covariate (somewhere between 3 and 20), so we delete it to make index
# of covariates we work with consistent
if exp_id in [3, 4]:
    VRData = np.delete(VRData, 8, axis=1)

# Deleting first multiple seconds showing a very negative position
ind = np.where(VRData[:, 3] < -60)[0]
starting_time = max(ind)+1
F = F[:, starting_time:]
Fneu = Fneu[:, starting_time:]
spks = spks[:, starting_time:]
VRData = VRData[starting_time:, :]

# Making sure that all elements of F are positive -- delete data for cells with negative Fluorescence values at some pos
F_min = np.min(F, axis=1)
ind = np.where(F_min < 0)[0]
F = np.delete(F, ind, axis=0)
Fneu = np.delete(Fneu, ind, axis=0)
spks = np.delete(spks, ind, axis=0)

# Deleting data for the last trial which has only one data point
ind = np.where(VRData[:, 20] == np.max(VRData[:, 20]))[0]
VRData = np.delete(VRData, ind, 0)
F = np.delete(F, ind, 1)
Fneu = np.delete(Fneu, ind, 1)
spks = np.delete(spks, ind, 1)
S = spks.T  #  timepoints by ROIs

# Features of VRData along with column number (for exp1 and exp2 - Two Tower Timout data)
# 0: time, 1: morph, 2: trialnum, 3: pos', 4: dz', 5: lick, 6, reward, 7: tstart,
# 8: teleport, 9: rzone, 10: toutzone, 11: clickOn, 12: blockWalls, 13: towerJitter, 14: wallJitter,
# 15: bckgndJitter, 16: sanning, 17: manrewards, 18: speed, 19: lick rate, 20: trial number
# You can get the overal morph value by computing VRData[:, 1] + VRData[:, 12] + VRData[:, 13] + VRData[:, 14]


#  Computing basic statistics of the data
ncells = F.shape[0]
ntimepoints = F.shape[1]
ntrials = len(set(VRData[:, 20]))
min_pos = np.floor(min(VRData[:, 3]))
max_pos = np.ceil(max(VRData[:, 3]))

#  Extracting trials of each morph
morph_0_trials = morph_trials(0)  # 1 dim vector
morph_d25_trials = morph_trials(0.25)
morph_d50_trials = morph_trials(0.5)
morph_d75_trials = morph_trials(0.75)
morph_1_trials = morph_trials(1)

# Computing activity rate for all cells and all trials along with the position
# compute_activity_rate(exp_id, morphs=morph_0_trials, morph_lvl=0, breaks=200)
# compute_activity_rate(exp_id, morphs=morph_d25_trials, morph_lvl=0.25, breaks=200)
# compute_activity_rate(exp_id, morphs=morph_d50_trials, morph_lvl=0.5, breaks=200)
# compute_activity_rate(exp_id, morphs=morph_d75_trials, morph_lvl=0.75, breaks=200)
# compute_activity_rate(exp_id, morphs=morph_1_trials, morph_lvl=1, breaks=200)
# dim: cell * position * trial
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


# Computing deviance of Gamma fit for each cell and all trials with the given morph level
# compute_Gamma_dev_cell(exp_id, 0, morph_0_trials)
# compute_Gamma_dev_cell(exp_id, 0.25, morph_d25_trials)
# compute_Gamma_dev_cell(exp_id, 0.5, morph_d50_trials)
# compute_Gamma_dev_cell(exp_id, 0.75, morph_d75_trials)
# compute_Gamma_dev_cell(exp_id, 1, morph_1_trials)  # RERUN THIS
# compute_Gamma_dev_indic(exp_id)
# Gamma_dev_cell_morph_0 = np.load(os.getcwd() + '/Data/dev_cell_Gamma_exp_' + str(exp_id) + '_morph_0.npy')
# Gamma_dev_cell_morph_d25 = np.load(os.getcwd() + '/Data/dev_cell_Gamma_exp_' + str(exp_id) + '_morph_0.25.npy')
# Gamma_dev_cell_morph_d50 = np.load(os.getcwd() + '/Data/dev_cell_Gamma_exp_' + str(exp_id) + '_morph_0.5.npy')
# Gamma_dev_cell_morph_d75 = np.load(os.getcwd() + '/Data/dev_cell_Gamma_exp_' + str(exp_id) + '_morph_0.75.npy')
# Gamma_dev_cell_morph_1 = np.load(os.getcwd() + '/Data/dev_cell_Gamma_exp_' + str(exp_id) + '_morph_1.npy')

# Computing deviance of Gamma fit for each cell and all trials with the given morph level
# NOTE: Check compute_PP function if you want to use it
# compute_PP_dev_cell(exp_id, 0, morph_0_trials)
# compute_PP_dev_cell(exp_id, 0.25, morph_d25_trials)
# compute_PP_dev_cell(exp_id, 0.5, morph_d50_trials)
# compute_PP_dev_cell(exp_id, 0.75, morph_d75_trials)
# compute_PP_dev_cell(exp_id, 1, morph_1_trials)
# PP_dev_cell_morph_0 = np.load(os.getcwd() + '/Data/dev_cell_PP_exp_' + str(exp_id) + '_morph_0.npy')
# PP_dev_cell_morph_d25 = np.load(os.getcwd() + '/Data/dev_cell_PP_exp_' + str(exp_id) + '_morph_0.25.npy')
# pp_dev_cell_morph_d50 = np.load(os.getcwd() + '/Data/dev_cell_PP_exp_' + str(exp_id) + '_morph_0.5.npy')
# PP_dev_cell_morph_d75 = np.load(os.getcwd() + '/Data/dev_cell_PP_exp_' + str(exp_id) + '_morph_0.75.npy')
# PP_dev_cell_morph_1 = np.load(os.getcwd() + '/Data/dev_cell_PP_exp_' + str(exp_id) + '_morph_1.npy')

# Computing active trials of each morph level
# compute_active_trials(exp_id, morph_0_trials, 0)
# compute_active_trials(exp_id, morph_d25_trials, 0.25)
# compute_active_trials(exp_id, morph_d50_trials, 0.5)
# compute_active_trials(exp_id, morph_d75_trials, 0.75)
# compute_active_trials(exp_id, morph_1_trials, 1)
# dim: cell * trials of that morph level
active_trials_0 = np.load(os.getcwd() + '/Data/active_trials_morph_0_exp_' + str(exp_id) + '.npy')
active_trials_d25 = np.load(os.getcwd() + '/Data/active_trials_morph_0.25_exp_' + str(exp_id) + '.npy')
active_trials_d50 = np.load(os.getcwd() + '/Data/active_trials_morph_0.5_exp_' + str(exp_id) + '.npy')
active_trials_d75 = np.load(os.getcwd() + '/Data/active_trials_morph_0.75_exp_' + str(exp_id) + '.npy')
active_trials_1 = np.load(os.getcwd() + '/Data/active_trials_morph_1_exp_' + str(exp_id) + '.npy')
# dim: cell * trials of that morph level
top_activity_trials_0 = np.load(os.getcwd() + '/Data/top_activity_trials_morph_0_exp_' + str(exp_id) + '.npy')
top_activity_trials_d25 = np.load(os.getcwd() + '/Data/top_activity_trials_morph_0.25_exp_' + str(exp_id) + '.npy')
top_activity_trials_d50 = np.load(os.getcwd() + '/Data/top_activity_trials_morph_0.5_exp_' + str(exp_id) + '.npy')
top_activity_trials_d75 = np.load(os.getcwd() + '/Data/top_activity_trials_morph_0.75_exp_' + str(exp_id) + '.npy')
top_activity_trials_1 = np.load(os.getcwd() + '/Data/top_activity_trials_morph_1_exp_' + str(exp_id) + '.npy')
# Putting all active_trials together for each cell
# dim: cell * trials
active_trials_all = np.zeros(shape=[ncells, ntrials])
active_trials_all[:, morph_0_trials] = active_trials_0
active_trials_all[:, morph_d25_trials] = active_trials_d25
active_trials_all[:, morph_d50_trials] = active_trials_d50
active_trials_all[:, morph_d75_trials] = active_trials_d75
active_trials_all[:, morph_1_trials] = active_trials_1

# Computing log-likelihood of Normal fit with spline input for each cell and all trials in one morph
# compute_spline_Normal_loglike(exp_id, morph_lvl=0, morphs=morph_0_trials)
# compute_spline_Normal_loglike(exp_id, morph_lvl=0.25, morphs=morph_d25_trials)
# compute_spline_Normal_loglike(exp_id, morph_lvl=0.5, morphs=morph_d50_trials)
# compute_spline_Normal_loglike(exp_id, morph_lvl=0.75, morphs=morph_d75_trials)
# compute_spline_Normal_loglike(exp_id, morph_lvl=1, morphs=morph_1_trials)
# dim: cell * 1
spline_Normal_loglike_morph_0 = np.load(os.getcwd() + '/Data/spline_Normal_loglike_exp_' + str(exp_id) + '_morph_0.npy')
spline_Normal_loglike_morph_d25 = np.load(os.getcwd() + '/Data/spline_Normal_loglike_exp_' + str(exp_id) + '_morph_0.25.npy')
spline_Normal_loglike_morph_d50 = np.load(os.getcwd() + '/Data/spline_Normal_loglike_exp_' + str(exp_id) + '_morph_0.5.npy')
spline_Normal_loglike_morph_d75 = np.load(os.getcwd() + '/Data/spline_Normal_loglike_exp_' + str(exp_id) + '_morph_0.75.npy')
spline_Normal_loglike_morph_1 = np.load(os.getcwd() + '/Data/spline_Normal_loglike_exp_' + str(exp_id) + '_morph_1.npy')

# Computing distance between outputs of two spline-Normal based models for morph0 and morph1
# compute_spline_Normal_dist(exp_id)
# dim: cells * 1
spline_Normal_dist = np.load(os.getcwd() + '/Data/spline_Normal_dist_exp_' + str(exp_id) + '.npy')

# Computing cells that differentiate between morph0 and morph 1 the most based on the spline-Normal models
# dim: cells * 2
# each row: cell_id, distance of two models
l = np.reshape(spline_Normal_dist, newshape=[ncells, 1])
s = np.reshape(np.arange(0, ncells, 1).T, newshape=[ncells, 1])
imp_diff_cells = np.append(s, l, axis=1)
imp_diff_cells = imp_diff_cells[imp_diff_cells[:, 1].argsort()]
imp_diff_cells = np.flip(imp_diff_cells, axis=0)

# Computing cells that gives the largest log-likelihood for morph 0 based on the spline-Normal model
# dim: cells * 2
# each row: cell_id, log-liklihood
l = np.reshape(spline_Normal_loglike_morph_0, newshape=[ncells, 1])
s = np.reshape(np.arange(0, ncells, 1).T, newshape=[ncells, 1])
imp_env0_cells = np.append(s, l, axis=1)
imp_env0_cells = imp_env0_cells[imp_env0_cells[:, 1].argsort()]
imp_env0_cells = np.flip(imp_env0_cells, axis=0)

# Computing cells that gives the largest log-likelihood for morph 0 based on the spline-Normal model
# These cells aren't necessarily the cells that encode morph0 the best ... (examine more!)
# dim: cells * 2
# each row: cell_id, log-liklihood
l = np.reshape(spline_Normal_loglike_morph_1, newshape=[ncells, 1])
s = np.reshape(np.arange(0, ncells, 1).T, newshape=[ncells, 1])
imp_env1_cells = np.append(s, l, axis=1)
imp_env1_cells = imp_env1_cells[imp_env1_cells[:, 1].argsort()]
imp_env1_cells = np.flip(imp_env1_cells, axis=0)

# Computing difference between log-likelihood of two 2poly-Gamma models respectively based on morph0 and morph1 data
# Takes long to run - must be ran for all experiments
# compute_loglike_diff_Gamma(exp_id)
# loglike_diff_Gamma = np.load(os.getcwd() + '/Data/loglike_diff_Gamma' + str(exp_id) + '.npy')

# Computing difference between log-likelihood of two spline-Gamma models respectively based on morph0 and morph1 data
# Takes long to run - must be ran for all experiments
# loglike_diff_spline_Gamma = np.load(os.getcwd() + '/Data/loglike_diff_spline_Gamma_exp_' + str(exp_id) + '.npy', allow_pickle=True)

# Computing difference between log-likelihood of two spline-Normal models respectively based on morph0 and morph1 data
# Takes long to run - must be ran for all experiments
# compute_loglike_diff_spline_Normal(exp_id)
# loglike_diff_spline_Normal = np.load(os.getcwd() + '/Data/loglike_diff_spline_Normal_exp_' + str(exp_id) + '.npy')

##################################   VISUALIZATION & PLOTS - ENCODING  ##############################################
# Visualize the data to see that we have multiple spatial maps (trial to trial variability) and fluctuations in the
# represented environment
'''
for rank in range(ncells):
    cell_id = rank+1
    show_activity_one_cell(exp_id, cell_id)
'''

# Looking at all trials of one cell, one by one (useful for debugging)
'''
cell_id = 16
print('cell id = {}'.format(cell_id))
show_active_inactive_trials(cell_id, 0, morph_0_trials, active_trials_0)
show_active_inactive_trials(cell_id, 1, morph_1_trials, active_trials_1)
for tr_id in morph_1_trials:
    print('tr_id = {}'.format(tr_id))
    ind = np.where(VRData[:, 20] == tr_id)[0]
    plt.plot(VRData[ind, 3], F[cell_id, ind], '.')
    plt.show()
'''

# For each cell observe the activity of trial of morph0 and morph1 along with which trials are active/inactive
'''
for rank in range(100):
    cell_id = int(imp_diff_cells[rank, 0])
    # cell_id = 75
    print('cell id = {}'.format(cell_id))
    show_active_inactive_trials(cell_id, 0, morph_0_trials, active_trials_0)
    show_active_inactive_trials(cell_id, 1, morph_1_trials, active_trials_1)
'''

# Check if we really have active/inactive or not
'''
for rank in range(100):
    # cell_id = imp_diff_cells[rank, 0].astype(int)
    cell_id = 16
    print('total number of trails = {}'.format(active_trials_1[cell_id, :].shape))
    print('number of active trials = {}'.format(np.sum(active_trials_1[cell_id, :])))
    print('cell_id = {}'.format(cell_id))
    plt.plot(np.sort(top_activity_trials_0[cell_id, :]), '.')
    plt.title('morph = 0')
    plt.show()
    plt.plot(np.sort(top_activity_trials_1[cell_id, :]), '.')
    plt.title('morph = 1')
    plt.show()
'''

# Activity rate for different morph levels
'''
for rank in range(100, 0, -1):
    cell_id = int(imp_diff_cells[rank, 0])
    print('cell id: {}'. format(cell_id))
    show_activity_per_morphs(cell_id)
'''

# Compare 2-Poly GLM models: Gamma with fixed c, Gamma with alter c, and IRLS for a fixed cell and all trials within a morph
# Because all trials of one morph are put together and we don't have spline inputs, the fits are poor
'''
for rank in range(100):
    cell_id = int(imp_diff_cells[rank, 0])
    compare_glm_models(cell_id, morphs=morph_0_trials, acc_rate=activity_rates_morph_0, acc_count=activity_count_morph_0)
'''

# Comparing 2-poly alt Gamma fit for morph0 and morph1 for each cell
# I encounter singular matrices for some cells, and also since I don't use spline the fits are not great
# This doesn't work for exp3
'''
for rank in range(100):
    cell_id = int(imp_diff_cells[rank, 0])
    print('cell_id = {}'.format(cell_id))
    compare_gamma_for_diff_morphs(cell_id)
'''

# Look at the spline fit of activity vs. position for important cells and all trials in a specific morph level
# To see the pattern (e.g. we have active/inactive trials) and also to see if knots are chosen appropriately
'''
for rank in range(100):
    cell_id = int(imp_diff_cells[rank, 0])
    visualize_spline_fit(morph_lvl=0, morphs=morph_0_trials, cell_id=cell_id)
    visualize_spline_fit(morph_lvl=1, morphs=morph_1_trials, cell_id=cell_id)
'''

# Looking at spline fit of activity vs. position for important cells of morph0 using all trials in that morph
'''
for rank in range(100):
    cell_id = int(imp_env0_cells[rank, 0])
    visualize_spline_fit(morph_lvl=0, morphs=morph_0_trials, cell_id=cell_id)
'''

# Looking at spline fit of activity vs. position for important cells of morph1 using all trials in that morph
'''
for rank in range(100):
    cell_id = int(imp_env1_cells[rank, 0])
    visualize_spline_fit(morph_lvl=1, morphs=morph_1_trials, cell_id=cell_id)
'''

# For each cell, observe difference of 2-poly Gamma fits for all trials
# For exp3 need to be checked after computing loglike_diff_Gamma is ran
'''
for rank in range(100):
    cell_id = int(imp_diff_cells[rank, 0])
    print('cell_id = {}'.format(cell_id))
    mean_tr = []
    ind = np.where(loglike_diff_Gamma[cell_id, :, 4] == 0)[0]
    mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_Gamma[cell_id, ind, 3])]
    ind = np.where(loglike_diff_Gamma[cell_id, :, 4] == 0.25)[0]
    mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_Gamma[cell_id, ind, 3])]
    ind = np.where(loglike_diff_Gamma[cell_id, :, 4] == 0.50)[0]
    mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_Gamma[cell_id, ind, 3])]
    ind = np.where(loglike_diff_Gamma[cell_id, :, 4] == 0.75)[0]
    mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_Gamma[cell_id, ind, 3])]
    ind = np.where(loglike_diff_Gamma[cell_id, :, 4] == 1)[0]
    mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_Gamma[cell_id, ind, 3])]
    plt.plot(loglike_diff_Gamma[cell_id, :, 3])
    plt.plot(mean_tr)
    plt.show()
'''

# For each cell, observe difference of spline Gamma fits (w.r.t active/inactive) for all trials

# Change/check codes below after computing loglike-diff-spline-Gamma
########################################################################
'''
for rank in range(100):
    cell_id = int(imp_diff_cells[rank, 0])
    print('cell_id = {}'.format(cell_id))
    mean_tr = []
    ind = np.where(loglike_diff_Gamma[cell_id, :, 4] == 0)[0]
    mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_Gamma[cell_id, ind, 3])]
    ind = np.where(loglike_diff_Gamma[cell_id, :, 4] == 0.25)[0]
    mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_Gamma[cell_id, ind, 3])]
    ind = np.where(loglike_diff_Gamma[cell_id, :, 4] == 0.50)[0]
    mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_Gamma[cell_id, ind, 3])]
    ind = np.where(loglike_diff_Gamma[cell_id, :, 4] == 0.75)[0]
    mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_Gamma[cell_id, ind, 3])]
    ind = np.where(loglike_diff_Gamma[cell_id, :, 4] == 1)[0]
    mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_Gamma[cell_id, ind, 3])]
    plt.plot(loglike_diff_Gamma[cell_id, :, 3])
    plt.plot(mean_tr)
    plt.show()
'''
'''
mean_tr = []
ind = np.where(loglike_diff_spline_Gamma[0, :, 4] == 0)[0]
mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_spline_Gamma[0, ind, 3])]
ind = np.where(loglike_diff_spline_Gamma[0, :, 4] == 0.25)[0]
mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_spline_Gamma[0, ind, 3])]
ind = np.where(loglike_diff_spline_Gamma[0, :, 4] == 0.50)[0]
mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_spline_Gamma[0, ind, 3])]
ind = np.where(loglike_diff_spline_Gamma[0, :, 4] == 0.75)[0]
mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_spline_Gamma[0, ind, 3])]
ind = np.where(loglike_diff_spline_Gamma[0, :, 4] == 1)[0]
mean_tr = mean_tr + len(ind)*[np.mean(loglike_diff_spline_Gamma[0, ind, 3])]

plt.plot(loglike_diff_spline_Gamma[0, :, 3])
plt.plot(mean_tr)
plt.show()

# Show the activity for one trial and compare it to morph0 and morph1 activities 
plt.subplot(2, 1, 1)
ind = np.where(np.isin(VRData[:, 20], morph_0_trials))[0]
plt.plot(VRData[ind, 3], F[cell_id, ind], '.')
trr = 88
tr_id = loglike_diff_spline_Gamma[0, trr, 0].astype(int)
print('tr_id = {}'.format(tr_id))
ind = np.where(VRData[:, 20] == tr_id)[0]
plt.plot(VRData[ind, 3], F[cell_id, ind], '.')

plt.subplot(2, 1, 2)
ind = np.where(np.isin(VRData[:, 20], morph_1_trials))[0]
plt.plot(VRData[ind, 3], F[cell_id, ind], '.')
ind = np.where(VRData[:, 20] == tr_id)[0]
plt.plot(VRData[ind, 3], F[cell_id, ind], '.')
plt.show()
'''
# Observing the 95% CI for morph0 and morph1 active/inactive spline normal mixt fits
'''
ind = np.where(active_trials_0[cell_id, :] == 1)[0]
morph0_act = morph_0_trials[ind]
ind = np.where(active_trials_0[cell_id, :] == 0)[0]
morph0_inact = morph_0_trials[ind]

ind = np.where(np.isin(VRData[:, 20], morph0_act))[0]
X = VRData[ind, 3]
Y = F[cell_id, ind]
arg_sort = np.argsort(X)
X_disc_act = X[arg_sort]
Y0_disc_act = Y[arg_sort]
spline_mat0_act = compute_spline_mat(X_disc_act)
gauss_0_act = sm.GLM(Y0_disc_act, spline_mat0_act, family=sm.families.Gaussian(sm.families.links.identity()))
# gauss_0_act = sm.GLM(Y0_disc_act, spline_mat0_act, family=sm.families.Gamma(sm.families.links.log))
gauss_res_0_act = gauss_0_act.fit()
est_0_act = gauss_res_0_act.mu
scale_0_act = np.sqrt(gauss_res_0_act.scale*sm.families.Gaussian.variance(est_0_act))
# scale_0_act = np.sqrt(gauss_res_0_act.scale * sm.families.Gamma.variance(est_0_act))

plt.subplot(2, 2, 1)
plt.plot(X_disc_act, est_0_act)
plt.plot(X_disc_act, est_0_act + 1.96*scale_0_act)
plt.plot(X_disc_act, est_0_act - 1.96*scale_0_act)
plt.plot(X, Y, ',')
plt.title('morph = 0, active trials')

ind = np.where(np.isin(VRData[:, 20], morph0_inact))[0]
X = VRData[ind, 3]
Y = F[cell_id, ind]
arg_sort = np.argsort(X)
X_disc_inact = X[arg_sort]
Y0_disc_inact = Y[arg_sort]
spline_mat0_inact = compute_spline_mat(X_disc_inact)
gauss_0_inact = sm.GLM(Y0_disc_inact, spline_mat0_inact, family=sm.families.Gaussian(sm.families.links.identity()))
# gauss_0_inact = sm.GLM(Y0_disc_inact, spline_mat0_inact, family=sm.families.Gamma(sm.families.links.log))
gauss_res_0_inact = gauss_0_inact.fit()
est_0_inact = gauss_res_0_inact.mu
scale_0_inact = np.sqrt(gauss_res_0_inact.scale*sm.families.Gaussian.variance(est_0_inact))
# scale_0_inact = np.sqrt(gauss_res_0_inact.scale * sm.families.Gamma.variance(est_0_inact))

plt.subplot(2, 2, 3)
plt.plot(X_disc_inact, est_0_inact)
plt.plot(X_disc_inact, est_0_inact + 1.96*scale_0_inact)
plt.plot(X_disc_inact, est_0_inact - 1.96*scale_0_inact)
plt.plot(X, Y, ',')
plt.title('morph = 0, inactive trials')

ind = np.where(active_trials_1[cell_id, :] == 1)[0]
morph1_act = morph_1_trials[ind]
ind = np.where(active_trials_1[cell_id, :] == 0)[0]
morph1_inact = morph_1_trials[ind]

ind = np.where(np.isin(VRData[:, 20], morph1_act))[0]
X = VRData[ind, 3]
Y = F[cell_id, ind]
arg_sort = np.argsort(X)
X_disc_act = X[arg_sort]
Y1_disc_act = Y[arg_sort]
spline_mat1_act = compute_spline_mat(X_disc_act)
gauss_1_act = sm.GLM(Y1_disc_act, spline_mat1_act, family=sm.families.Gaussian(sm.families.links.identity()))
# gauss_1_act = sm.GLM(Y1_disc_act, spline_mat1_act, family=sm.families.Gamma(sm.families.links.log))
gauss_res_1_act = gauss_1_act.fit()
est_1_act = gauss_res_1_act.mu
scale_1_act = np.sqrt(gauss_res_1_act.scale*sm.families.Gaussian.variance(est_1_act))
# scale_1_act = np.sqrt(gauss_res_1_act.scale * sm.families.Gamma.variance(est_1_act))

plt.subplot(2, 2, 2)
plt.plot(X_disc_act, est_1_act)
plt.plot(X_disc_act, est_1_act + 1.96*scale_1_act)
plt.plot(X_disc_act, est_1_act - 1.96*scale_1_act)
plt.plot(X, Y, ',')
plt.title('morph = 1, active trials')

ind = np.where(np.isin(VRData[:, 20], morph1_inact))[0]
X = VRData[ind, 3]
Y = F[cell_id, ind]
arg_sort = np.argsort(X)
X_disc_inact = X[arg_sort]
Y1_disc_inact = Y[arg_sort]
spline_mat1_inact = compute_spline_mat(X_disc_inact)
gauss_1_inact = sm.GLM(Y1_disc_inact, spline_mat1_inact, family=sm.families.Gaussian(sm.families.links.identity()))
# gauss_1_inact = sm.GLM(Y1_disc_inact, spline_mat1_inact, family=sm.families.Gamma(sm.families.links.log))
gauss_res_1_inact = gauss_1_inact.fit()
est_1_inact = gauss_res_1_inact.mu
scale_1_inact = np.sqrt(gauss_res_1_inact.scale*sm.families.Gaussian.variance(est_1_inact))
# scale_1_inact = np.sqrt(gauss_res_1_inact.scale * sm.families.Gamma.variance(est_1_inact))

plt.subplot(2, 2, 4)
plt.plot(X_disc_inact, est_1_inact)
plt.plot(X_disc_inact, est_1_inact + 1.96*scale_1_inact)
plt.plot(X_disc_inact, est_1_inact - 1.96*scale_1_inact)
plt.plot(X, Y, ',')
plt.title('morph = 1, inactive trials')
plt.show()
'''
# Observing the 95% CI for morph0 and morph1 active/inactive in spline Gamma fits
'''
ind = np.where(active_trials_0[cell_id, :] == 1)[0]
morph0_act = morph_0_trials[ind]
ind = np.where(active_trials_0[cell_id, :] == 0)[0]
morph0_inact = morph_0_trials[ind]

ind = np.where(np.isin(VRData[:, 20], morph0_act))[0]
X = VRData[ind, 3]
Y = F[cell_id, ind]
arg_sort = np.argsort(X)
X_disc_act = X[arg_sort]
Y0_disc_act = Y[arg_sort]
spline_mat0_act = compute_spline_mat(X_disc_act)
gamma_0_act = sm.GLM(Y0_disc_act, spline_mat0_act, family=sm.families.Gamma(sm.families.links.identity))
gamma_res_0_act = gamma_0_act.fit()
mu_0_act = gamma_res_0_act.mu
v_0_act = 1/gamma_res_0_act.scale
sd_0_act = mu_0_act / np.sqrt(v_0_act)

plt.subplot(2, 2, 1)
plt.plot(X_disc_act, mu_0_act)
plt.plot(X_disc_act, mu_0_act + 1.96*sd_0_act)
plt.plot(X_disc_act, mu_0_act - 1.96*sd_0_act)
plt.plot(X, Y, ',')
plt.title('morph = 0, active trials')

ind = np.where(np.isin(VRData[:, 20], morph0_inact))[0]
X = VRData[ind, 3]
Y = F[cell_id, ind]
arg_sort = np.argsort(X)
X_disc_inact = X[arg_sort]
Y0_disc_inact = Y[arg_sort]
spline_mat0_inact = compute_spline_mat(X_disc_inact)
gamma_0_inact = sm.GLM(Y0_disc_inact, spline_mat0_inact, family=sm.families.Gamma(sm.families.links.identity))
gamma_res_0_inact = gamma_0_inact.fit()
mu_0_inact = gamma_res_0_inact.mu
v_0_inact = 1/gamma_res_0_inact.scale
sd_0_inact = mu_0_inact / np.sqrt(v_0_inact)

plt.subplot(2, 2, 3)
plt.plot(X_disc_inact, mu_0_inact)
plt.plot(X_disc_inact, mu_0_inact + 1.96*sd_0_inact)
plt.plot(X_disc_inact, mu_0_inact - 1.96*sd_0_inact)
plt.plot(X, Y, ',')
plt.title('morph = 0, inactive trials')

ind = np.where(active_trials_1[cell_id, :] == 1)[0]
morph1_act = morph_1_trials[ind]
ind = np.where(active_trials_1[cell_id, :] == 0)[0]
morph1_inact = morph_1_trials[ind]

ind = np.where(np.isin(VRData[:, 20], morph1_act))[0]
X = VRData[ind, 3]
Y = F[cell_id, ind]
arg_sort = np.argsort(X)
X_disc_act = X[arg_sort]
Y1_disc_act = Y[arg_sort]
spline_mat1_act = compute_spline_mat(X_disc_act)
gamma_1_act = sm.GLM(Y1_disc_act, spline_mat1_act, family=sm.families.Gamma(sm.families.links.identity))
gamma_res_1_act = gamma_1_act.fit()
mu_1_act = gamma_res_1_act.mu
v_1_act = 1/gamma_res_1_act.scale
sd_1_act = mu_1_act / np.sqrt(v_1_act)

plt.subplot(2, 2, 2)
plt.plot(X_disc_act, mu_1_act)
plt.plot(X_disc_act, mu_1_act + 1.96*sd_1_act)
plt.plot(X_disc_act, mu_1_act - 1.96*sd_1_act)
plt.plot(X, Y, ',')
plt.title('morph = 1, active trials')

ind = np.where(np.isin(VRData[:, 20], morph1_inact))[0]
X = VRData[ind, 3]
Y = F[cell_id, ind]
arg_sort = np.argsort(X)
X_disc_inact = X[arg_sort]
Y1_disc_inact = Y[arg_sort]
spline_mat1_inact = compute_spline_mat(X_disc_inact)
gamma_1_inact = sm.GLM(Y1_disc_inact, spline_mat1_inact, family=sm.families.Gamma(sm.families.links.identity))
gamma_res_1_inact = gamma_1_inact.fit()
mu_1_inact = gamma_res_1_inact.mu
v_1_inact = 1/gamma_res_1_inact.scale
sd_1_inact = mu_1_inact / np.sqrt(v_1_inact)

plt.subplot(2, 2, 4)
plt.plot(X_disc_inact, mu_1_inact)
plt.plot(X_disc_inact, mu_1_inact + 1.96*sd_1_inact)
plt.plot(X_disc_inact, mu_1_inact - 1.96*sd_1_inact)
plt.plot(X, Y, ',')
plt.title('morph = 1, inactive trials')
plt.show()
'''
########################################################################

##################################   DECODING   ##############################################

p = .02  # the probability of going from one state to another one randomely
hist_wind = 10  # The history windon for history dependent part of spline-Gamma models
mode = 'short'  # in short mode we only work with small number of cells (to facilitate coding and debugging)
# mode = 'all'  # in all mode the data from all cells is used

if mode == 'short':
    num = 100  # number of cells that we work with
    cells_under_study = imp_diff_cells[:num, 0].astype(int)
if mode == 'all':
    num = ncells  # number of cells that we work with
    cells_under_study = range(ncells)

# For any fixed cell, we decode morph instantaneously for all trials w.r.t models fitted in function decode_morphs
# decode_morphs(exp_id, p, mode=mode, visualize=False, visualize2=False, history=False)
p_morph = np.load(os.getcwd() + '/Data/' + mode + '_p_morph_exp_' + str(exp_id) + '.npy', allow_pickle=True)
goodness_of_fit = np.load(os.getcwd() + '/Data/' + mode + '_goodness_of_fit_exp_' + str(exp_id) + '.npy', allow_pickle=True)
gamma_fit_0_act = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_act_exp_' + str(exp_id) + '.npy', allow_pickle=True)
gamma_fit_0_act_nohist = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_act_nohist_exp_' + str(exp_id) + '.npy', allow_pickle=True)
gamma_fit_0_inact = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_inact_exp_' + str(exp_id) + '.npy', allow_pickle=True)
gamma_fit_0_inact_nohist = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_inact_nohist_exp_' + str(exp_id) + '.npy', allow_pickle=True)
# gamma_fit_0_all = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_0_all_exp_' + str(exp_id) + '.npy', allow_pickle=True)
gamma_fit_1_act = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_act_exp_' + str(exp_id) + '.npy', allow_pickle=True)
gamma_fit_1_act_nohist = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_act_nohist_exp_' + str(exp_id) + '.npy', allow_pickle=True)
gamma_fit_1_inact = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_inact_exp_' + str(exp_id) + '.npy', allow_pickle=True)
gamma_fit_1_inact_nohist = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_inact_nohist_exp_' + str(exp_id) + '.npy', allow_pickle=True)
# gamma_fit_1_all = np.load(os.getcwd() + '/Data/' + mode + '_gamma_fit_1_all_exp_' + str(exp_id) + '.npy', allow_pickle=True)

# decode_morphs_joint(exp_id, p, mode=mode, visualize=False)
p_morph_joint = np.load(os.getcwd() + '/Data/' + mode + '_p_morph_joint_exp_' + str(exp_id) + '.npy', allow_pickle=True)

# Computing performance of different cells in decoding
# compute_cells_dec_perf(exp_id, cells_under_study, mode)
cells_dec_perf = np.load(os.getcwd() + '/Data/' + mode + '_cells_dec_perf_exp_' + str(exp_id) + '.npy', allow_pickle=True)

# Sorting cells based on their decoding performance
# Sort1. Based on good hit rate performance for morph0 and morph1 trials
temp = cells_dec_perf[cells_under_study, :]
temp = np.flip(temp[np.argsort(temp[:, 2] + temp[:, 4]), :], axis=0)
sorted_hit_cells = temp[:, 0].astype(int)
# Sort2. Based on average log-liklihood achieved for morph0 and morph1 trials
temp = cells_dec_perf[cells_under_study, :]
temp = np.flip(temp[np.argsort(temp[:, 1] + temp[:, 3]), :], axis=0)
sorted_avg_ll_cells = temp[:, 0].astype(int)


##################################   VISUALIZATION & PLOTS - DECODING  ##############################################

# For each cell look at the activity of that cell for all trials (observe trial to trail variability)
'''
cell_ids = sorted_hit_cells
cell_ids = [220, 56, 27, 32]
for cell_id in cell_ids:
    show_activity_one_cell(exp_id, cell_id)
'''

# For each trial look at the activity of all cells for that trials, sorted by place fields
'''
tr_ids = morph_0_trials
tr_ids = [5, 39, 96, 1]
show_activity_one_trial(exp_id, tr_ids, imp_diff_cells[:400, 0].astype(int))
'''

# For each in intermediate environements show similarity of that trial to average of trials of known environments
'''
# cell_ids = sorted_hit_cells
cell_ids = imp_diff_cells[:100, 0].astype(int)
cell_ids = np.array([220, 75, 7, 151] + list(cell_ids))
show_similarity(exp_id, morph_0_trials, cell_ids)
'''

# For a fixed cell and any trial within a morph group looking at decoded morph results  $$$$$$
'''
# cell_ids = sorted_hit_cells
cell_ids = [72]
for rank in range(len(cell_ids)):
    cell_id = cell_ids[rank]
    morphs = morph_d50_trials
    show_decode_results(cell_id, morphs=morphs, history=False)
'''

# For a fixed cell and any trial within a morph group looking at encoding results$$$$$$
'''
# cell_ids = sorted_hit_cells
cell_ids = [75, 224]
for rank in range(len(cell_ids)):
    cell_id = cell_ids[rank]
    morphs = morph_0_trials
    show_encode_results(cell_id, history=False)
'''


# For each cell show the decoding morph for all trials within a fixed morph level splitted by active/inactive
'''
show_decode_results_1morph_trials(exp_id, sorted_hit_cells, morph_lvl=.25, morphs=morph_d25_trials, act_tr = active_trials_d25)
'''

# For each cell show the decoding morph for all trials split by morph levels and active/inactive plus the
# top-activity of morph0 and morph1 trials for that cell  $$$$$$
'''
cell_ids = sorted_hit_cells
cell_ids = [13, 220, 25]  # some good cells for Exp. 3
show_decode_results_all_trials(exp_id, cell_ids, p_morph)
'''

# For each trial within a certain morph, looking at decoded values for all cells -- not informative
'''
for tr_id in morph_d25_trials:
    P0 = np.zeros(shape=[num, p_morph[sorted_hit_cells[0], tr_id, 0].shape[0]])
    for i in range(num):
        cell_id = sorted_hit_cells[i]
        pp = p_morph[cell_id, tr_id, 3]
        P0[i, :] = pp[0, :]

    P0_exag = 0.5*np.ones(shape=P0.shape)
    P0_exag[P0 > 0.7] = 1
    P0_exag[P0 < 0.3] = 0

    print('trial id = {}'.format(tr_id))
    plt.subplot(3, 1, 1)
    plt.imshow(P0, aspect='auto')
    plt.colorbar()
    plt.xlabel('position')
    plt.ylabel('cell_id')
    plt.title('p morph')

    plt.subplot(3, 1, 2)
    plt.imshow(P0_exag, aspect='auto')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    A = np.mean(P0, axis=0)
    plt.plot(np.mean(P0, axis=0))
    # plt.plot(np.quantile(P0, q=0.25, axis=0))
    # plt.plot(np.quantile(P0, q=0.75, axis=0))
    plt.show()
'''

# The mean morph decoded value of all cells for all trials (trials of each morph in one color)  $$$$$$
'''
all_morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
# cols = ['b', 'g', 'c', 'm', 'r']
cols = [mr1, mr2, mr3, mr4, mr5]
for i in range(5):
    morphs = all_morphs[i]
    col = cols[i]
    b = True
    for tr_id in morphs:
        position = p_morph[sorted_hit_cells[0], tr_id, 0]
        P0 = np.zeros(shape=[num, position.shape[0]])
        for j in range(num):
            cell_id = sorted_hit_cells[j]
            pp = p_morph[cell_id, tr_id, 3]
            P0[j, :] = pp[0, :]
        if b:
            plt.plot(position, np.mean(P0, axis=0), color=col, label='morph='+str(i*0.25))
            b = False
        else:
            plt.plot(position, np.mean(P0, axis=0), color=col)
    # plt.show()

plt.title('Smoothing')
plt.ylabel('P(morph = 0)')
plt.xlabel('position')
plt.legend()
plt.show()
'''

# morph by morph, looking at joint decoding result for trials from that morph (each morph one color)  -- not informative
'''
all_morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
# cols = ['b', 'g', 'c', 'm', 'r']
cols = [mr1, mr2, mr3, mr4, mr5]
for i in range(5):
    morphs = all_morphs[i]
    col = cols[i]
    b = True
    for tr_id in morphs:
        pp = p_morph_joint[tr_id, :]
        pos = pp[0]
        sm = pp[3]
        sm = sm[0, :]
        plt.plot(pos, sm, color=col)
    plt.title('morphs='+str(i*0.25))
    plt.ylabel('P(morph = 0)')
    plt.xlabel('position')
    plt.show()
'''

# For each cell, looking at its mean activity during all trials within each morph level  $$$$$$
'''
all_morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
cols = [mr1, mr2, mr3, mr4, mr5]
# cols = ['b', 'g', 'c', 'm', 'r']
X_range = np.arange(np.min(VRData[:, 3]), np.max(VRData[:, 3]), 0.5)
for cell_id in sorted_hit_cells:
    print('cell id = {}'.format(cell_id))
    pp = np.zeros(shape=[ntrials, X_range.shape[0]])
    for tr_id in range(ntrials):
        pos = p_morph[cell_id, tr_id, 0]
        p_m = p_morph[cell_id, tr_id, 3]
        ind = find_nearest(pos, X_range)
        pp[tr_id, :] = p_m[0, ind]
    for i in range(5):
        morphs = all_morphs[i]
        col = cols[i]
        plt.plot(X_range, np.mean(pp[morphs, :], axis=0), color=col, label='morph='+str(i*0.25))
    plt.legend()
    plt.title('Smoothing')
    plt.xlabel('position')
    plt.ylabel('P(morph = 0)')
    plt.show()
'''

# For each trial, looking at decoded morph from the joint model ********
'''
trial_ids = morph_d75_trials
for tr_id in trial_ids:
    print('traisl id = {}'.format(tr_id))
    [X_tr, Y_tr, p_morph_filt_joint, p_morph_smooth_joint, L_morph_joint, ll_morph_joint] = p_morph_joint[tr_id, :]
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
'''

# Looking at decoded morph from the joint model for all trials in one imshow plot $$$$$$
'''
show_joint_decode_results_all_trials()
'''

# For each trial look at loglike-diff from the joint model and effect of each cell on it (Smooth) $$$$$$
'''
tr_ids = morph_d25_trials
for tr_id in tr_ids:
    print('trial id = {}'.format(tr_id))
    [X_tr_joint, Y_tr_joint, p_filt_joint, p_smooth_joint, L_joint, ll_joint] = p_morph_joint[tr_id, :]
    P = np.zeros(shape=[sorted_hit_cells.shape[0], X_tr_joint.shape[0]])
    for i in range(sorted_hit_cells.shape[0]):
        cell_id = sorted_hit_cells[i]
        [X_tr, Y_tr, P_filt, p_smooth, L, ll] = p_morph[cell_id, tr_id, :]
        P[i, :] = ll[0, :] - ll[1, :]
    P_norm = P  # Normalizing P for better visualization
    plt.subplot(4, 1, 1)
    plt.imshow(P_norm, aspect='auto', extent=[np.min(X_tr_joint), np.max(X_tr_joint), 0, num], vmin=-10, vmax=10)
    plt.colorbar()
    plt.ylabel('cell_id')
    plt.subplot(4, 1, 2)
    plt.imshow(P_norm, aspect='auto', extent=[np.min(X_tr_joint), np.max(X_tr_joint), 0, num], vmin=-10, vmax=10)
    plt.ylabel('cell_id')
    plt.subplot(4, 1, 3)
    plt.plot(ll_joint[0, :] - ll_joint[1, :], color='m')
    plt.xlim(0, X_tr.shape[0])
    plt.ylabel('loglikelihoods')
    plt.subplot(4, 1, 4)
    plt.plot(p_smooth_joint[0, :], color=orange1)
    plt.xlim(0, X_tr.shape[0])
    plt.ylim(-.5, 1.5)
    plt.xlabel('position')
    plt.ylabel('joint p_smooth')
    plt.show()
'''

# Conduct goodness-of-fit for existence multiple spatial maps
'''
cell_ids = imp_diff_cells[:100, 0].astype(int)
assess_goodness_of_fit(exp_id, cell_ids)
'''

# Fig. 1 for Cosyne
'''
X_range = np.arange(np.min(VRData[:, 3]), np.max(VRData[:, 3]), 0.5)
all_morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
active_trials = [active_trials_0, active_trials_d25, active_trials_d50, active_trials_d75, active_trials_1]
cells = [2003]
for x in range(len(cells)):
    cell_id = cells[x]
    print('cell id = {}'.format(cell_id))
    for j in range(5):
        tr_ids = all_morphs[j]
        act_tr = active_trials[j]
        P0 = np.zeros(shape=[tr_ids.shape[0], X_range.shape[0]])
        for i in range(tr_ids.shape[0]):
            tr_id = tr_ids[i]
            ind = find_nearest(p_morph[cell_id, tr_id, 0], X_range)
            pp = p_morph[cell_id, tr_id, 3]
            P0[i, :] = pp[0, ind]
        plt.subplot2grid((5, 22), (j, 10*x+x), colspan=10+x)
        plt.imshow(P0, aspect='auto')
        plt.xticks([])
        if x == 0:
            plt.yticks([])
            #plt.colorbar()
        #if x == 0:
            #plt.ylabel('m = ' + str(j*0.25))
        if j == 4:
            plt.xlabel('position')
            plt.xticks(np.arange(0, 1001, 200), [-50, 50, 150, 250, 350, 450])
        if j == 0:
            plt.title('Cell id = ' + str(cell_id))
plt.show()
'''

# Fig. 2 for Cosyne
'''
all_morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
X_range = np.arange(np.min(VRData[:, 3]), np.max(VRData[:, 3]), 0.5)
for j in range(5):
    trial_ids = all_morphs[j]
    P0 = np.zeros(shape=[trial_ids.shape[0], X_range.shape[0]])
    for i in range(trial_ids.shape[0]):
        tr_id = trial_ids[i]
        ind = find_nearest(p_morph_joint[tr_id, 0], X_range)
        pp = p_morph_joint[tr_id, 3]
        P0[i, :] = pp[0, ind]
    plt.subplot(5, 1, j+1)
    plt.imshow(P0, aspect='auto')
    #plt.colorbar()
    #plt.ylabel('m = ' + str(j*0.25))
    plt.yticks([])
    plt.xticks([])
    if j == 4:
        plt.xlabel('position')
        plt.xticks(np.arange(0, 1001, 200), [-50, 50, 150, 250, 350, 450])
plt.show()
'''

# Fig. 3 for Cosyne
'''
tr_ids = [83]
for tr_id in tr_ids:
    print('trial id = {}'.format(tr_id))
    [X_tr_joint, Y_tr_joint, p_filt_joint, p_smooth_joint, L_joint, ll_joint] = p_morph_joint[tr_id, :]
    P = np.zeros(shape=[sorted_hit_cells.shape[0], X_tr_joint.shape[0]])
    for i in range(sorted_hit_cells.shape[0]):
        cell_id = sorted_hit_cells[i]
        [X_tr, Y_tr, P_filt, p_smooth, L, ll] = p_morph[cell_id, tr_id, :]
        P[i, :] = ll[1, :] - ll[0, :]
    P_norm = P  # Normalizing P for better visualization

    plt.ylabel('cell_id')
    plt.imshow(P_norm, aspect='auto', extent=[np.min(X_tr_joint), np.max(X_tr_joint), 0, num], vmin=-10, vmax=10)
    plt.xticks([])
    plt.ylabel('individual \n log-likelihood \n difference')
    plt.colorbar()
    plt.show()

    plt.plot(ll_joint[1, :] - ll_joint[0, :], color='m')
    plt.xlim(0, X_tr.shape[0])
    plt.xticks([])
    plt.yticks([-99, 0, 99])
    plt.ylabel('joint \n log-likelihoods \n difference')
    plt.show()

    plt.plot(p_smooth_joint[1, :], color=orange1)
    plt.xlim(0, X_tr.shape[0])
    plt.xticks(np.arange(0, X_tr.shape[0], X_tr.shape[0] / 5), np.arange(-50, 451, 100))
    plt.ylim(-0.2, 1.2)
    plt.xlabel('position')
    plt.ylabel('posterior \n probability \n of m = 0')
    plt.show()
'''

# Figures for NuerIPS:
# cell_ids = [220, 75, 56, 13]
# show_multiple_maps_neurips(cell_ids)

# tr_ids = [5, 39, 96, 1]
# show_population_maps_neurips(exp_id, tr_ids, imp_diff_cells[:300, 0].astype(int))

# cell_ids = imp_diff_cells[:100, 0].astype(int)
# plot_cells = [220, 75, 56, 13]
# show_fluctuations_neurips(exp_id, cell_ids, plot_cells)

# cell_ids = imp_diff_cells[:100, 0].astype(int)
# cell_ids = [220, 75, 56, 13]
# cell_ids = [143, 182, 264, 19]
# show_decoding_results_neurips(exp_id, cell_ids)

# Everything for joint decoding
# cell_ids = imp_diff_cells[:100, 0].astype(int)
# cell_ids = [5, 559, 805]
# show_decoding_results_neurips(exp_id, cell_ids)

# print(morph_d50_trials)
# x = input()

# For exp 3
# '''
# population_cells = [75, 220, 202]
# population_cells = [75, 56, 13]
# decode_morphs_joint_selected(exp_id, p, mode=mode, visualize=False, visualize2=True, visualize3=False, visualize4=False, selected_trial=30, diff_trials=[30], selected_cells=population_cells)
# population_cells = [75, 220, 61]
# population_cells = [75, 56, 220]
# decode_morphs_joint_selected(exp_id, p, mode=mode,   visualize=False, visualize2=True, visualize3=False, visualize4=False, selected_trial=30, diff_trials=[30], selected_cells=population_cells)


# tr_id = 30
# group1 = [153, 202, 175, 559, 805, 376, 240]
# decode_morphs_joint_selected(exp_id, p, mode=mode, visualize=False, visualize2=True, visualize3=True, visualize4=False, selected_trial=tr_id, selected_cells=group1)
# group2 = [75, 12, 16, 3, 220, 35, 4]
# decode_morphs_joint_selected(exp_id, p, mode=mode, visualize=False, visualize2=True, visualize3=True, visualize4=False, selected_trial=tr_id, selected_cells=group2)
# group3 = group1 + group2
# decode_morphs_joint_selected(exp_id, p, mode=mode, visualize=False, visualize2=True, visualize3=True, visualize4=False, selected_trial=tr_id, selected_cells=group3)
# '''

'''
print(morph_0_trials)
print(morph_d25_trials)
print(morph_d50_trials)
print(morph_d75_trials)
print(morph_1_trials)
population_cells = imp_diff_cells[:100, 0].astype(int)
diff_trs = [30, 11, 74, 0, 61, 100]
decode_morphs_joint_selected(exp_id, p, mode=mode, visualize=False, visualize2=True, visualize3=False, visualize4=True, selected_trial=30, diff_trials=diff_trs, selected_cells=population_cells)
'''

# For exp 4
# cell_ids = imp_diff_cells[:10, 0].astype(int)
# show_decoding_results_neurips(exp_id, cell_ids)
# population_cells = imp_diff_cells[:100, 0].astype(int)
# decode_morphs_joint_selected(exp_id, p, mode=mode, visualize=False, visualize2=True, visualize3=False, visualize4=False, selected_trial=30, diff_trials=[30], selected_cells=population_cells)

p_range = np.arange(0, 1, .005)
# cell_ids = imp_diff_cells[:20, 0].astype(int)
# trial_ids = np.array(list(morph_d25_trials) + list(morph_d50_trials) + list(morph_d75_trials))
# compute_trans_prob(exp_id, p_range, cell_ids, trial_ids)
trans_prob_ll = np.load(os.getcwd() + '/Data/' + mode + '_trans_prob_ll_exp_' + str(exp_id) + '.npy')
# cell_ids = [75, 13, 12, 3]
# show_trans_prob(exp_id, cell_ids, trial_ids)

# cell_ids = imp_diff_cells[:100, 0].astype(int)
# assess_goodness_of_fit2(exp_id, cell_ids)
# trial_ids = np.array(list(morph_0_trials) + list(morph_d25_trials) + list(morph_d50_trials) + list(morph_d75_trials) + list(morph_1_trials))
# trial_ids = np.array(list(morph_0_trials))
# assess_goodness_of_fit_decoding(exp_id, cell_ids, trial_ids)
p_morph_1map = np.load(os.getcwd() + '/Data/' + mode + '_p_morph_1map_exp_' + str(exp_id) + '.npy', allow_pickle=True)

cell_ids = imp_diff_cells[:100, 0].astype(int)
# cell_ids = [75, 13, 56, 220]
trial_ids = np.array(list(morph_0_trials))
show_map_impact_on_decoding(exp_id, cell_ids, trial_ids)

# morphs = [2, 25, 65]
# show_joint_decode_results_single_trial(morphs)

# Comparing a naive approach that doesn't state-space component with our approach
# cell_ids = imp_diff_cells[:100, 0].astype(int)
naive_approach(exp_id, cell_ids, morph_0_trials, morph_lvl=0)
# naive_approach(exp_id, cell_ids, morph_1_trials, morph_lvl=1)
plt.plot([0, 100], [0, 0], color='grey', linestyle='dashed')
plt.xticks([])
plt.yticks([])
# plt.legend()
plt.show()

