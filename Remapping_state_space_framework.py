import os
import sys
import numpy as np
from numpy import genfromtxt
import matplotlib
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
import networkx as nx
import random
import community as community_louvain
import matplotlib.cm as cm
import colorsys

class state_model:
    # E: the number of different states
    # vals: the possible values for state
    # name: name of the state-transition model
    # q: probability of going from one state to another one (only for 1MC model)

    def __init__(self, input_E, input_q):
        self.name = "1MC"
        self.E = input_E
        self.vals = np.arange(0, self.E)
        self.q = input_q

    def set_q (self, new_q):
        self.q = new_q

    def set_E(self, new_E):
        self.E = new_E
        self.vals = np.arange(0, self.E)

    def get_model(self):
        return self.name


class map_model:
    # M: the number of different maps
    # vals: the possible values for maps
    # name: name of the state-transition model
    # state_model: the state model that the map model is build on top of that
    # map_prob: matrix R that shows the probability of using different maps for different environments (only for
    # "fixed_map" model)

    def __init__(self, input_M, input_state_model):
        self.name = "fixed_map"
        self.M = input_M
        self.vals = np.arange(0, self.M)
        self.state_model = input_state_model
        self.map_prob = 1 / self.state_model.E * np.ones(shape=[self.state_model.E, self.M])

    def set_E(self, new_M):
        self.M = new_M
        self.vals = np.arange(0, self.M)

    def set_state_model(self, new_state_model):
        self.state_model = new_state_model
        self.map_prob = 1/self.M * np.ones(shape=[self.state_model.E, self.M])
    def set_map_prob(self, new_map_prob):
        self.map_prob = new_map_prob

    def get_model(self):
        return self.name


class response_model:
    # name: name of the state-transition model
    # h: length of history window (only for "Gamma_history_dependent model)
    # map_model: the map model that this response model is build on top of that
    def __init__(self, input_cell_id, input_map_model, input_h):
        self.name = "Gamma_history_dependent"
        self.cell_id = input_cell_id
        self.h = input_h
        self.map_model = input_map_model
        self.observation_models = np.empty(shape=[self.map_model.M, ], dtype=object)  # params  ## Will be extended later
        self.vis_observation_models = np.empty(shape=[self.map_model.M, ], dtype=object)  # params  ## Will be extended later

    def fit_models(self, map_id, tr_ids):
        X = np.array(1)
        Y = np.array(1)
        Y_hist = np.array(1)
        first_tr = True
        for tr_id in tr_ids:
            # using disceretized data points:
            # step = (max_pos - min_pos) / nbreaks
            # X_tr = np.arange(min_pos, max_pos, step) + step / 2
            # Y_tr = activity_rates_all_morphs[cell_id, :, tr_id]

            # using actual data points:
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]

            Y_hist_tr = compute_hist_cov(Y_tr, self.h)
            X_tr = X_tr[self.h:]
            Y_tr = Y_tr[self.h:]
            if first_tr:
                X = X_tr.copy()
                Y = Y_tr.copy()
                Y_hist = Y_hist_tr
                first_tr = False
            else:
                X = np.concatenate((X, X_tr), axis=0)
                Y = np.concatenate((Y, Y_tr), axis=0)
                Y_hist = np.concatenate((Y_hist, Y_hist_tr), axis=0)

        # Fititng model with no history for visualization only
        spline_mat = compute_spline_mat(X)
        gamma_nohist_mod = sm.GLM(Y, spline_mat, family=sm.families.Gamma(sm.families.links.identity))
        gamma_nohist_res = gamma_nohist_mod.fit()
        gamma_nohist_mu = gamma_nohist_res.mu
        gamma_nohist_v = 1 / gamma_nohist_res.scale
        gamma_nohist_params = gamma_nohist_res.params

        # Fitting model with history
        des_mat = np.concatenate((spline_mat, Y_hist), axis=1)
        gamma_hist_mod = sm.GLM(Y, des_mat, family=sm.families.Gamma(sm.families.links.identity))
        gamma_hist_res = gamma_hist_mod.fit()
        gamma_hist_mu = gamma_hist_res.mu
        gamma_hist_v = 1 / gamma_hist_res.scale
        gamma_hist_params = gamma_hist_res.params

        # If you trully care about readability of your code you can define a class for fitted models, and let elements
        # such as X, Y, Y_hist, etc be attributes of it. Not gonna do that now.
        self.observation_models[map_id] = [X, Y, Y_hist, spline_mat, des_mat, gamma_hist_mu, gamma_hist_v,
                                           gamma_hist_params]
        self.vis_observation_models[map_id] = [X, Y, Y_hist, spline_mat, des_mat, gamma_nohist_mu, gamma_nohist_v,
                                               gamma_nohist_params]

    def decode_morph(self, debug1, debug2):
        trial_ids = range(ntrials)
        # trial_ids = morph_0_trials
        self.p_morph = np.empty(shape=[ntrials, 7], dtype=object)
        for tr_id in trial_ids:
            # print('computing likelihood for tr_id = {}'.format(tr_id))
            ind = np.where(VRData[:, 20] == tr_id)[0]
            X_tr = VRData[ind, :]
            X_tr = X_tr[:, 3]
            Y_tr = F[cell_id, ind]
            Y_hist_tr = compute_hist_cov(Y_tr, self.h)
            X_tr = X_tr[self.h:]
            Y_tr = Y_tr[self.h:]
            ll = np.zeros(shape=[self.map_model.M, X_tr.shape[0]])

            for map_id in range(self.map_model.M):
                # print('\t using map {}'.format(map_id))
                # Recalling parameters from models fitted before:
                obv_model = self.observation_models[map_id]
                X_model = obv_model[0]
                mu_model = obv_model[5]
                spline_model = obv_model[3]
                v_model = obv_model[6]
                params_model = obv_model[7]

                # compute likelihood
                for i in range(X_tr.shape[0]):
                    ind0 = find_nearest(X_model, [X_tr[i]])
                    Y_hist_now = np.reshape(Y_hist_tr[i, :], newshape=[1, Y_hist_tr.shape[1]])
                    inp = np.concatenate((spline_model[ind0, :], Y_hist_now), axis=1)
                    mu_est = inp.dot(params_model)[0]
                    mu_est = max(.1, mu_est)
                    sd_est = mu_est / np.sqrt(v_model)
                    ll[map_id, i] = -scipy.special.loggamma(v_model) + v_model * np.log(
                        v_model * Y_tr[i] / mu_est) - v_model * Y_tr[i] / mu_est - np.log(Y_tr[i])


            # Run filter algorithm
            curr_E = self.map_model.state_model.E
            filt = np.zeros(shape=[curr_E, self.map_model.M, self.map_model.M, X_tr.shape[0]])
            p_morph_filt = np.zeros(shape=[curr_E, X_tr.shape[0]])

            for i in range(X_tr.shape[0]):
                for map0 in range(self.map_model.M):
                    for map1 in range(self.map_model.M):
                        L0 = np.exp(ll[map0, i])
                        L1 = np.exp(ll[map1, i])
                        if i == 0:
                            filt[0, map0, map1, i] = self.map_model.map_prob[0, map0] * L0
                            filt[1, map0, map1, i] = self.map_model.map_prob[1, map1] * L1
                        else:
                            qq = self.map_model.state_model.q
                            filt[0, map0, map1, i] = L0 * ((1-qq)*filt[0, map0, map1, i-1] + qq*filt[1, map0, map1, i-1])
                            filt[1, map0, map1, i] = L1 * (qq*filt[0, map0, map1, i-1] + (1-qq)*filt[1, map0, map1, i-1])
                        # filt[:, map0, map1, i] = filt[:, map0, map1, i]/np.sum(filt[:, map0, map1, i])
                # Normalizing computed probabilities so they add to 1:
                filt[:, :, :, i] = filt[:, :, :, i] / np.sum(filt[:, :, :, i])
                p_morph_filt[0, i] = np.sum(filt[0, :, :, i])
                p_morph_filt[1, i] = np.sum(filt[1, :, :, i])

            # Run smoother algorithm
            p_morph_smooth = np.zeros(shape=[2, X_tr.shape[0]])
            for i in range(X_tr.shape[0] - 1, -1, -1):
                if i == X_tr.shape[0] - 1:
                    p_morph_smooth[0, i] = p_morph_filt[0, i]
                    p_morph_smooth[1, i] = p_morph_filt[1, i]
                if i < X_tr.shape[0] - 1:
                    p_2step_0 = (1 - qq) * p_morph_filt[0, i] + qq * p_morph_filt[1, i]
                    p_2step_1 = (1 - qq) * p_morph_filt[1, i] + qq * p_morph_filt[0, i]
                    p_morph_smooth[0, i] = p_morph_filt[0, i] * ((1 - qq) * p_morph_smooth[0, i+1] / p_2step_0 + qq * p_morph_smooth[1, i + 1] / p_2step_1)
                    p_morph_smooth[1, i] = p_morph_filt[1, i] * ((1 - qq) * p_morph_smooth[1, i+1] / p_2step_1 + qq * p_morph_smooth[0, i + 1] / p_2step_0)
                    p_morph_smooth[:, i] = p_morph_smooth[:, i] / np.sum(p_morph_smooth[:, i])
            L = np.exp(ll)
            L = L/np.sum(L, axis=0)  # The Likelihood is normalized, while the log-likelihood (ll) is not.
            self.p_morph[tr_id, :] = [X_tr, Y_tr, filt, p_morph_filt, p_morph_smooth, ll, L]

            if debug1:
                print('tr_id = {}'.format(tr_id))
                print('maps_prob: \n \t {}'.format(self.map_model.map_prob))
                for map0 in range(self.map_model.M):
                    for map1 in range(self.map_model.M):
                        print('map0={}, map1={}'.format(map0, map1))
                        plt.subplot(5, 2, 1)
                        map = self.observation_models[map0]
                        plt.plot(map[0], map[5], '.', color=blue1)
                        plt.plot(map[0], map[5] + 2*map[5]/np.sqrt(map[6]), '.', color=orange1)
                        plt.plot(map[0], map[5] - 2*map[5]/np.sqrt(map[6]), '.', color=orange1)
                        plt.plot(X_tr, Y_tr, 'r,')
                        plt.xlim([min_pos, max_pos])
                        plt.title('map = 0')
                        plt.ylabel('fitted models')

                        plt.subplot(5, 2, 2)
                        map = self.observation_models[map1]
                        plt.plot(map[0], map[5], '.', color=blue1)
                        plt.plot(map[0], map[5] + 2 * map[5] / np.sqrt(map[6]), '.', color=orange1)
                        plt.plot(map[0], map[5] - 2 * map[5] / np.sqrt(map[6]), '.', color=orange1)
                        plt.plot(X_tr, Y_tr, 'r,')
                        plt.xlim([min_pos, max_pos])
                        plt.title('map = 1')

                        plt.subplot(5, 2, 3)
                        plt.plot(X_tr, ll[map0, :])
                        plt.xlim([min_pos, max_pos])
                        plt.ylabel('likelihood')

                        plt.subplot(5, 2, 4)
                        plt.plot(X_tr, ll[map1, :])
                        plt.xlim([min_pos, max_pos])

                        plt.subplot(5, 2, 5)
                        plt.plot(X_tr, filt[0, map0, map1, :], color=blue1, label='env0')
                        plt.plot(X_tr, filt[1, map0, map1, :], color=orange1, label='env1')
                        plt.xlim([min_pos, max_pos])
                        plt.ylabel('spec. filter')
                        plt.legend()

                        plt.subplot(5, 2, 6)
                        plt.plot(X_tr, filt[0, map0, map1, :], color=blue1, label='env0')
                        plt.plot(X_tr, filt[1, map0, map1, :], color=orange1, label='env1')
                        plt.xlim([min_pos, max_pos])

                        plt.subplot(5, 2, 7)
                        plt.plot(X_tr, p_morph_filt[0, :], color=blue1, label='env0')
                        plt.plot(X_tr, p_morph_filt[1, :], color=orange1, label='env1')
                        plt.xlim([min_pos, max_pos])
                        plt.ylabel('filter')

                        plt.subplot(5, 2, 8)
                        plt.plot(X_tr, p_morph_filt[0, :], color=blue1, label='env0')
                        plt.plot(X_tr, p_morph_filt[1, :], color=orange1, label='env1')
                        plt.xlim([min_pos, max_pos])

                        plt.subplot(5, 2, 9)
                        plt.plot(X_tr, p_morph_smooth[0, :], color=blue1, label='env0')
                        plt.plot(X_tr, p_morph_smooth[1, :], color=orange1, label='env1')
                        plt.xlim([min_pos, max_pos])
                        plt.ylabel('smooth')

                        plt.subplot(5, 2, 10)
                        plt.plot(X_tr, p_morph_smooth[0, :], color=blue1, label='env0')
                        plt.plot(X_tr, p_morph_smooth[1, :], color=orange1, label='env1')
                        plt.xlim([min_pos, max_pos])
                        plt.show()

            if debug2:
                print('tr_id = {}'.format(tr_id))
                print('maps_prob: \n \t {}'.format(self.map_model.map_prob))
                for map_id in range(self.map_model.M):
                    plt.subplot(6+self.map_model.M, self.map_model.M, map_id + 1)
                    map = self.vis_observation_models[map_id]
                    plt.plot(map[0], map[5], '.', color=blue1)
                    plt.plot(map[0], map[5] + 2*map[5]/np.sqrt(map[6]), '.', color=orange1)
                    plt.plot(map[0], map[5] - 2*map[5]/np.sqrt(map[6]), '.', color=orange1)
                    plt.plot(X_tr, Y_tr, 'r,')
                    plt.xlim([min_pos, max_pos])
                    plt.title('map = {}'.format(map_id))
                    plt.ylabel('models')

                    plt.subplot(6+self.map_model.M, self.map_model.M, self.map_model.M + map_id + 1)
                    plt.plot(X_tr, ll[map_id, :])
                    plt.xlim([min_pos, max_pos])
                    if map_id == 0:
                        plt.ylabel('like')

                    for new_map in range(self.map_model.M):
                        plt.subplot(6 + self.map_model.M, self.map_model.M, (2+new_map)*self.map_model.M + map_id + 1)
                        plt.plot(X_tr, filt[0, map_id, new_map, :], color=blue1, label='env0')
                        plt.plot(X_tr, filt[1, map_id, new_map, :], color=orange1, label='env1')
                        plt.xlim([min_pos, max_pos])
                        if map_id == 0 and new_map == 0:
                            plt.legend()

                    plt.subplot(6+self.map_model.M, self.map_model.M, (2+self.map_model.M) * self.map_model.M + map_id + 1)
                    plt.plot(X_tr, p_morph_filt[0, :], color=blue1, label='env0')
                    plt.plot(X_tr, p_morph_filt[1, :], color=orange1, label='env1')
                    plt.xlim([min_pos, max_pos])
                    if map_id == 0:
                        plt.ylabel('filter')

                    plt.subplot(6 + self.map_model.M, self.map_model.M, (3 + self.map_model.M) * self.map_model.M + map_id + 1)
                    plt.plot(X_tr, p_morph_smooth[0, :], color=blue1, label='env0')
                    plt.plot(X_tr, p_morph_smooth[1, :], color=orange1, label='env1')
                    plt.xlim([min_pos, max_pos])
                    if map_id == 0:
                        plt.ylabel('smooth')

                    plt.subplot(6 + self.map_model.M, self.map_model.M, (4 + self.map_model.M) * self.map_model.M + map_id + 1)
                    for trr in morph_0_trials:
                        ind = np.where(VRData[:, 20] == trr)[0]
                        plt.plot(VRData[ind, 3], F[self.cell_id, ind], '.', markersize=2, alpha=.1, color=blue1, label='env0')
                    plt.xlim([min_pos, max_pos])
                    if map_id == 0:
                        plt.ylabel('env0')

                    plt.subplot(6 + self.map_model.M, self.map_model.M, (5 + self.map_model.M) * self.map_model.M + map_id + 1)
                    for trr in morph_1_trials:
                        ind = np.where(VRData[:, 20] == trr)[0]
                        plt.plot(VRData[ind, 3], F[self.cell_id, ind], '.', markersize=2, alpha=.1, color=orange1, label='env1')
                    plt.xlim([min_pos, max_pos])
                    if map_id == 0:
                        plt.ylabel('env1')

                plt.show()
            # '''

    # shows different maps
    def show_models(self):
        for map_id in range(self.map_model.M):
            map = self.vis_observation_models[map_id]
            plt.plot(map[0], map[5], '.')
            # plt.plot(map[0], map[5]+2*map[5]/np.sqrt(map[6]), '.')
            # plt.plot(map[0], map[5]-2*map[5]/np.sqrt(map[6]), '.')
        plt.show()

    # shows different maps and actual data points from env0 and env1 for each map
    def show_models2(self):
        for map_id in range(self.map_model.M):
            print('percentage of env0 trials in this map = {}'.format(self.map_model.map_prob[0, map_id]))
            map = self.vis_observation_models[map_id]
            plt.plot(map[0], map[5], '.')
            for tr_id in morph_0_trials:
                plt.plot(activity_rates_all_morphs[self.cell_id, :, tr_id], 'r,')
            for tr_id in morph_1_trials:
                plt.plot(activity_rates_all_morphs[self.cell_id, :, tr_id], 'b,')
            plt.show()

    def show_decoding_results(self, trial_by_trial, morph_by_morph):
        trial_ids = range(ntrials)
        if trial_by_trial:
            for tr_id in trial_ids:
                print('tr_id = {}'.format(tr_id))
                [X_tr, Y_tr, filt, p_morph_filt, p_morph_smooth, ll, L] = self.p_morph[tr_id, :]
                print('maps_prob: \n \t {}'.format(self.map_model.map_prob))
                for map_id in range(self.map_model.M):
                    plt.subplot(6 + self.map_model.M, self.map_model.M, map_id + 1)
                    map = self.vis_observation_models[map_id]
                    plt.plot(map[0], map[5], '.', color=blue1)
                    plt.plot(map[0], map[5] + 2 * map[5] / np.sqrt(map[6]), '.', color=orange1)
                    plt.plot(map[0], map[5] - 2 * map[5] / np.sqrt(map[6]), '.', color=orange1)
                    plt.plot(X_tr, Y_tr, 'r,')
                    plt.xlim([min_pos, max_pos])
                    plt.title('map = {}'.format(map_id))
                    plt.ylabel('models')

                    plt.subplot(6 + self.map_model.M, self.map_model.M, self.map_model.M + map_id + 1)
                    plt.plot(X_tr, ll[map_id, :])
                    plt.xlim([min_pos, max_pos])
                    if map_id == 0:
                        plt.ylabel('like')

                    for new_map in range(self.map_model.M):
                        plt.subplot(6 + self.map_model.M, self.map_model.M,
                                    (2 + new_map) * self.map_model.M + map_id + 1)
                        plt.plot(X_tr, filt[0, map_id, new_map, :], color=blue1, label='env0')
                        plt.plot(X_tr, filt[1, map_id, new_map, :], color=orange1, label='env1')
                        plt.xlim([min_pos, max_pos])
                        if map_id == 0 and new_map == 0:
                            plt.legend()

                    plt.subplot(6 + self.map_model.M, self.map_model.M,
                                (2 + self.map_model.M) * self.map_model.M + map_id + 1)
                    plt.plot(X_tr, p_morph_filt[0, :], color=blue1, label='env0')
                    plt.plot(X_tr, p_morph_filt[1, :], color=orange1, label='env1')
                    plt.xlim([min_pos, max_pos])
                    if map_id == 0:
                        plt.ylabel('filter')

                    plt.subplot(6 + self.map_model.M, self.map_model.M,
                                (3 + self.map_model.M) * self.map_model.M + map_id + 1)
                    plt.plot(X_tr, p_morph_smooth[0, :], color=blue1, label='env0')
                    plt.plot(X_tr, p_morph_smooth[1, :], color=orange1, label='env1')
                    plt.xlim([min_pos, max_pos])
                    if map_id == 0:
                        plt.ylabel('smooth')

                    plt.subplot(6 + self.map_model.M, self.map_model.M,
                                (4 + self.map_model.M) * self.map_model.M + map_id + 1)
                    for trr in morph_0_trials:
                        ind = np.where(VRData[:, 20] == trr)[0]
                        plt.plot(VRData[ind, 3], F[self.cell_id, ind], '.', markersize=2, alpha=.1, color=blue1,
                                 label='env0')
                    plt.xlim([min_pos, max_pos])
                    if map_id == 0:
                        plt.ylabel('env0')

                    plt.subplot(6 + self.map_model.M, self.map_model.M,
                                (5 + self.map_model.M) * self.map_model.M + map_id + 1)
                    for trr in morph_1_trials:
                        ind = np.where(VRData[:, 20] == trr)[0]
                        plt.plot(VRData[ind, 3], F[self.cell_id, ind], '.', markersize=2, alpha=.1, color=orange1,
                                 label='env1')
                    plt.xlim([min_pos, max_pos])
                    if map_id == 0:
                        plt.ylabel('env1')

                plt.show()

        if morph_by_morph:
            morphs = [morph_0_trials, morph_d25_trials, morph_d50_trials, morph_d75_trials, morph_1_trials]
            for j in range(len(morphs)):
                tr_ids = morphs[j]
                step = (max_pos - min_pos) / nbreaks
                X_disc = np.arange(min_pos, max_pos, step) + step / 2
                D = np.zeros(shape=[tr_ids.shape[0], X_disc.shape[0]])
                for k in range(len(tr_ids)):
                    tr_id = tr_ids[k]
                    [X_tr, Y_tr, filt, p_morph_filt, p_morph_smooth, ll, L] = self.p_morph[tr_id, :]
                    ind = find_nearest(X_tr, X_disc)
                    D[k, :] = p_morph_smooth[0, ind]
                plt.subplot(len(morphs), 1, j+1)
                plt.imshow(D, aspect='auto', vmin=0, vmax=1)
                plt.colorbar()
            plt.show()



def find_nearest(arr1, arr2):
    # For each element x of arr2, find the index of the closest element of arr1 to that.

    ans = []
    for x in arr2:
        y = np.argmin(np.abs(arr1 - x))
        ans.append(y)
    ans = np.array(ans)
    return ans


def generate_n_colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r/255, g/255, b/255))
    return ret


def compute_hist_cov(Y, hist_wind):
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
    X_disc = np.arange(min_pos, max_pos, step) + step/2
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
    if morph_lvl == 0:
        np.save(os.getcwd() + '/Data/disc_pos' + str(exp_id) + '.npy', X_disc)
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


def compute_M_single_cell(cell_id, E):
    morphs = np.concatenate((morph_0_trials, morph_1_trials), axis=0)
    # morphs = morph_1_trials
    G = nx.Graph()
    for i in range(len(morphs)):
        tr_id1 = morphs[i]
        for j in range(i+1, len(morphs)):
            tr_id2 = morphs[j]
            x = activity_rates_all_morphs[cell_id, :, tr_id1]
            y = activity_rates_all_morphs[cell_id, :, tr_id2]
            x = gaussian_filter(x, sigma=20)
            y = gaussian_filter(y, sigma=20)
            w = np.mean((x-np.mean(x))*(y-np.mean(y)))
            # w = 1/np.mean((x-y)**2)
            if w > 0:
                G.add_edge(tr_id1, tr_id2, weight=w)
    plt.show()

    # Using Louvain algorithm (throw away negative edges)
    partition = community_louvain.best_partition(G)
    clusters = dict()
    for item in partition.items():
        tr_id = item[0]
        part = item[1]
        if part not in clusters.keys():
            clusters[part] = [tr_id]
        else:
            clusters[part].append(tr_id)
    num_clusters = max(clusters.keys())+1  # +1 because cluster ids are 0-based

    # rates shows the P(E=e|A=a), i. e. for each map, what is the percent of each env
    '''
    rates = np.zeros(shape=[E, num_clusters])
    for item in partition.items():
        tr_id = item[0]
        part = item[1]
        if tr_id in morph_0_trials:
            rates[0, part] += 1 / len(clusters[part])
        if tr_id in morph_1_trials:
            rates[1, part] += 1 / len(clusters[part])
    print(rates)
    '''

    # rates shows the P(A=a|E=e), i. e. for each env, what is the percent of each map
    rates = np.zeros(shape=[E, num_clusters])
    for item in partition.items():
        tr_id = item[0]
        part = item[1]
        if tr_id in morph_0_trials:
            rates[0, part] += 1 / morph_0_trials.shape[0]
        if tr_id in morph_1_trials:
            rates[1, part] += 1 / morph_1_trials.shape[0]
    print(rates)

    # Drawing the graph
    '''
    print(partition)
    # pos = nx.spring_layout(G)
    pos = nx.circular_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
    '''
    # Showing neural activity of trials that belong to different clusters
    '''
    plt.subplot(2, 1, 1)
    print('number of detected different maps: {}'.format(num_clusters))
    cols = generate_n_colors(num_clusters)
    for j in range(num_clusters):
        first_tr = False
        for tr_id in clusters[j]:
            plt.plot(activity_rates_all_morphs[cell_id, :, tr_id], color=cols[j])

    # for j in range(num_clusters):
    #     plt.subplot(num_clusters, 1, j+1)
    #     for tr_id in clusters[j]:
    #         plt.plot(activity_rates_all_morphs[cell_id, :, tr_id])
    # plt.show()

    plt.subplot(2, 1, 2)
    cols = generate_n_colors(2)
    for tr_id in morph_0_trials:
        plt.plot(activity_rates_all_morphs[cell_id, :, tr_id], color=cols[0])
    for tr_id in morph_1_trials:
        plt.plot(activity_rates_all_morphs[cell_id, :, tr_id], color=cols[1])
    plt.show()

    '''
    # Using Girvan-Newman algorithm (throw away negative edges)
    '''
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    edges = np.array(edges)
    weights = np.array(weights)
    # weights = weights + np.min(weights)+1
    w_upp_quant = np.quantile(weights, .7)

    old_G = G.copy()
    for u, v, d in old_G.edges(data=True):
        if d['weight'] < w_upp_quant:
            G.remove_edge(u, v)
        # if d['weight'] < 0:
        #     G.remove_edge(u, v)
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    edges = np.array(edges)
    weights = np.array(weights)

    pos = nx.circular_layout(G)
    node_labels = dict()
    for node in pos.keys():
        node_labels[node] = node
    nx.draw(G, pos, node_color='b', edgelist=list(edges), edge_color=list(weights), width=2, edge_cmap=plt.cm.Blues)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    comp = nx.algorithms.community.centrality.girvan_newman(G)
    print(tuple(sorted(c) for c in next(comp)))
    plt.show()
    '''

    return [num_clusters, clusters, rates]


# def decode_morphs(exp_id, cell_id, tr_id):

''' ------------------------------- Reading and cleaning data ------------------------------- '''

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

# Cleaning the data
'''
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

# Detecting first jump at start of trials and removing it

for tr_id in range(len(set(VRData[:, 20]))):
    print(tr_id)
    ind = np.where(VRData[:, 20] == tr_id)[0]
    # bad_ind = [x for x in np.arange(1, len(pos)) if np.abs(pos[x] - pos[x-1])>10]
    bad_ind = np.max([x for x in ind if np.abs(VRData[x, 3] - VRData[x-1, 3]) > 10])
    rem = [x for x in ind if x < bad_ind]
    VRData = np.delete(VRData, rem, axis=0)
    F = np.delete(F, rem, axis=1)
    Fneu = np.delete(F, rem, axis=1)
np.save(os.getcwd() + '/Data/VRData_clean' + str(exp_id) + '.npy', VRData)
np.save(os.getcwd() + '/Data/F_clean' + str(exp_id) + '.npy', F)
np.save(os.getcwd() + '/Data/Fneu_clean' + str(exp_id) + '.npy', Fneu)
'''

VRData = np.load(os.getcwd() + '/Data/VRData_clean' + str(exp_id) + '.npy')  # timepoints by 20 features
F = np.load(os.getcwd() + '/Data/F_clean' + str(exp_id) + '.npy')  # timepoints by 20 features
Fneu = np.load(os.getcwd() + '/Data/Fneu_clean' + str(exp_id) + '.npy')  # timepoints by 20 features

# Features of VRData along with column number (for exp1 and exp2 - Two Tower Timout data)
# 0: time, 1: morph, 2: trialnum, 3: pos', 4: dz', 5: lick, 6, reward, 7: tstart,
# 8: teleport, 9: rzone, 10: toutzone, 11: clickOn, 12: blockWalls, 13: towerJitter, 14: wallJitter,
# 15: bckgndJitter, 16: sanning, 17: manrewards, 18: speed, 19: lick rate, 20: trial number
# You can get the overal morph value by computing VRData[:, 1] + VRData[:, 12] + VRData[:, 13] + VRData[:, 14]

# Detecting data points with huge jump
'''
for i in range(1, VRData.shape[0]):
    curr_time = VRData[i, 0]
    prev_time = VRData[i-1, 0]
    curr_pos = VRData[i, 3]
    prev_pos = VRData[i-1, 3]
    curr_tr = VRData[i, 20]
    prev_tr = VRData[i-1, 20]
    if curr_tr == prev_tr and np.abs(curr_pos - prev_pos) > 20:
        print('tr_id = {}'.format(curr_tr))
        print('prev_time = {}, curr_time = {}'.format(prev_time, curr_time))
        print('prev_pos = {}, curr_pos = {}'.format(prev_pos, curr_pos))
        print('trial of next data point is {}'.format(VRData[i+1, 20]))
        pos = VRData[:, 3]
        plt.plot(pos[i-20:i+20])
        plt.show()
print('end of inspection')
x = input()
'''

#  Computing basic statistics of the data
ncells = F.shape[0]  # number of cells
ntimepoints = F.shape[1]  # number of time points
ntrials = len(set(VRData[:, 20]))  # number of trials
min_pos = np.floor(min(VRData[:, 3]))  # smallest position
max_pos = np.ceil(max(VRData[:, 3]))  # largest position


''' ------------------------------- Encoding ------------------------------- '''

# Extracting trials of each morph level
morph_0_trials = morph_trials(0)  # 1-dim vector
morph_d25_trials = morph_trials(0.25)
morph_d50_trials = morph_trials(0.5)
morph_d75_trials = morph_trials(0.75)
morph_1_trials = morph_trials(1)

# For each morph level, constructing a matrix  that contains activity rates of all cells for all trials with given morph
# level as a function of position
nbreaks = 400  # number of bins in discretized location
# compute_activity_rate(exp_id, morphs=morph_0_trials, morph_lvl=0, breaks=nbreaks)
# compute_activity_rate(exp_id, morphs=morph_d25_trials, morph_lvl=0.25, breaks=nbreaks)
# compute_activity_rate(exp_id, morphs=morph_d50_trials, morph_lvl=0.5, breaks=nbreaks)
# compute_activity_rate(exp_id, morphs=morph_d75_trials, morph_lvl=0.75, breaks=nbreaks)
# compute_activity_rate(exp_id, morphs=morph_1_trials, morph_lvl=1, breaks=nbreaks)
disc_pos = np.load(os.getcwd() + '/Data/disc_pos' + str(exp_id) + '.npy')
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
activity_rates_all_morphs = activity_rates_morph_0 + activity_rates_morph_d25 + activity_rates_morph_d50 + \
                            activity_rates_morph_d75 + activity_rates_morph_1
activity_count_all_morphs = activity_count_morph_0 + activity_count_morph_d25 + activity_count_morph_d50 + \
                            activity_count_morph_d75 + activity_count_morph_1

# For each cell, for each morph level, computing the log-likelihood of fitted spline-Normal model
# compute_spline_Normal_loglike(exp_id, morph_lvl=0, morphs=morph_0_trials)
# compute_spline_Normal_loglike(exp_id, morph_lvl=0.25, morphs=morph_d25_trials)
# compute_spline_Normal_loglike(exp_id, morph_lvl=0.5, morphs=morph_d50_trials)
# compute_spline_Normal_loglike(exp_id, morph_lvl=0.75, morphs=morph_d75_trials)
# compute_spline_Normal_loglike(exp_id, morph_lvl=1, morphs=morph_1_trials)
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
# compute_spline_Normal_dist(exp_id)
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

mode = 'custom'  # Wroking with cells that are chosen by hand
# mode = 'short'  # Working with only a small subset of the cells (for debugging and basic visualizations)
# mode = 'all'  # Working with all of the cells

if mode == 'custom':
    cells_under_study = [220, 75, 56, 13]  # choosing cells to work with
    num = len(cells_under_study)
if mode == 'short':
    num = 2  # number of cells that we work with
    cells_under_study = imp_diff_cells[:num, 0].astype(int)  # choosing cells to work with
if mode == 'all':
    num = ncells  # number of cells that we work with
    cells_under_study = range(ncells)  # choosing cells to work with

# Making State_model, Map_model and Response_model for each cell
# '''
cell_models = dict()
for cell_id in cells_under_study:
    print('cell_id = {}'.format(cell_id))
    new_state_model = state_model(input_E=2, input_q=0.05)
    [M, clusters, rates] = compute_M_single_cell(cell_id, E=2)
    new_map_model = map_model(M, new_state_model)
    new_map_prob = np.zeros(shape=[new_state_model.E, M])
    new_map_model.set_map_prob(rates)
    new_response_model = response_model(cell_id, new_map_model, input_h=15)
    print('clusters = {}'.format(clusters))
    for cl in clusters.keys():
        # print('cell_id = {}, map_id = {}'.format(cell_id, cl))
        print(cl)
        new_response_model.fit_models(cl, clusters[cl])
    cell_models[cell_id] = new_response_model
np.save(os.getcwd() + '/Data/cell_models_' + mode +'_exp_' + str(exp_id) + '.npy', cell_models)
# '''
cell_models = np.load(os.getcwd() + '/Data/cell_models_' + mode +'_exp_' + str(exp_id) + '.npy', allow_pickle=True).item()

# Running filter/smooter for each cell to decode represented environment
'''
for cell_id in cell_models.keys():
    r_model = cell_models[cell_id]
    print('cell_id: {}'.format(cell_id))
    obv = r_model.observation_models[0]
    # r_model.show_models()
    # r_model.show_models2()
    r_model.decode_morph(debug1=False, debug2=False)
'''
np.save(os.getcwd() + '/Data/cell_models_' + mode + '_exp_' + str(exp_id) + '.npy', cell_models)
cell_models = np.load(os.getcwd() + '/Data/cell_models_' + mode +'_exp_' + str(exp_id) + '.npy', allow_pickle=True).item()

# Visualizing the fitted models and decoded environment for each cell
'''
for cell_id in cell_models.keys():
    r_model = cell_models[cell_id]
    print('cell_id: {}'.format(cell_id))
    obv = r_model.observation_models[0]
    # r_model.show_models()
    # r_model.show_models2()
    r_model.show_decoding_results(trial_by_trial=False, morph_by_morph=True)
'''










