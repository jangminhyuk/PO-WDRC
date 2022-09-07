#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

def summarize(out_lq_list, out_dr_list, dist, path, num, plot_results=True):
    x_list, J_list, y_list, u_list = [], [], [], []
    x_lqr_list, J_lqr_list, y_lqr_list, u_lqr_list = [], [], [], []
    time_list, time_lqr_list = [], []


    for out in out_dr_list:
         x_list.append(out['state_traj'])
         J_list.append(out['cost'])
         y_list.append(out['output_traj'])
         u_list.append(out['control_traj'])
         time_list.append(out['comp_time'])


    for out in out_lq_list:
         x_lqr_list.append(out['state_traj'])
         J_lqr_list.append(out['cost'])
         y_lqr_list.append(out['output_traj'])
         u_lqr_list.append(out['control_traj'])
         time_lqr_list.append(out['comp_time'])

    x_mean, J_mean, y_mean, u_mean = np.mean(x_list, axis=0), np.mean(J_list, axis=0), np.mean(y_list, axis=0), np.mean(u_list, axis=0)
    x_lqr_mean, J_lqr_mean, y_lqr_mean, u_lqr_mean = np.mean(x_lqr_list, axis=0), np.mean(J_lqr_list, axis=0), np.mean(y_lqr_list, axis=0), np.mean(u_lqr_list, axis=0)
    x_std, J_std, y_std, u_std = np.std(x_list, axis=0), np.std(J_list, axis=0), np.std(y_list, axis=0), np.std(u_list, axis=0)
    x_lqr_std, J_lqr_std, y_lqr_std, u_lqr_std = np.std(x_lqr_list, axis=0), np.std(J_lqr_list, axis=0), np.std(y_lqr_list, axis=0), np.std(u_lqr_list, axis=0)

    time_ar = np.array(time_list)
    time_lqr_ar = np.array(time_lqr_list)
    J_ar = np.array(J_list)
    J_lqr_ar = np.array(J_lqr_list)

    if plot_results:
        nx = x_mean.shape[1]
        T = u_mean.shape[0]
        nu = u_mean.shape[1]
        ny= y_mean.shape[1]

        fig = plt.figure(figsize=(6,4), dpi=300)

        t = np.arange(T+1)
        for i in range(nx):


            if x_lqr_list != []:
                plt.plot(t, x_lqr_mean[:,i,0], 'tab:red', label='LQG')
                plt.fill_between(t, x_lqr_mean[:,i, 0] + x_lqr_std[:,i,0],
                               x_lqr_mean[:,i,0] - x_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
            plt.plot(t, x_mean[:,i,0], 'tab:blue', label='WDRC')
            plt.fill_between(t, x_mean[:,i,0] + x_std[:,i,0],
                               x_mean[:,i,0] - x_std[:,i,0], facecolor='tab:blue', alpha=0.3)

            plt.xlabel(r'$t$', fontsize=22)
            plt.ylabel(r'$x_{}$'.format(i+1), fontsize=22)
            plt.legend(fontsize=18)
            plt.grid()
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlim([t[0], t[-1]])
            ax = fig.gca()
            ax.locator_params(axis='y', nbins=5)
            ax.locator_params(axis='x', nbins=5)
            fig.set_size_inches(6, 4)
            plt.savefig(path +'states_{}_{}.pdf'.format(i+1, num), dpi=300, bbox_inches="tight")
            plt.clf()

        t = np.arange(T)
        for i in range(nu):

            if u_lqr_list != []:
                plt.plot(t, u_lqr_mean[:,i,0], 'tab:red', label='LQG')
                plt.fill_between(t, u_lqr_mean[:,i,0] + u_lqr_std[:,i,0],
                             u_lqr_mean[:,i,0] - u_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)

            plt.plot(t, u_mean[:,i,0], 'tab:blue', label='WDRC')
            plt.fill_between(t, u_mean[:,i,0] + u_std[:,i,0],
                             u_mean[:,i,0] - u_std[:,i,0], facecolor='tab:blue', alpha=0.3)
            plt.xlabel(r'$t$', fontsize=20)
            plt.ylabel(r'$u_{}$'.format(i+1), fontsize=20)
            plt.legend(fontsize=18)
            plt.grid()
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlim([t[0], t[-1]])

            plt.savefig(path +'controls_{}_{}.pdf'.format(i+1, num), dpi=300, bbox_inches="tight")
            plt.clf()

        t = np.arange(T+1)
        for i in range(ny):
            if y_lqr_list != []:
                plt.plot(t, y_lqr_mean[:,i,0], 'tab:red', label='LQG')
                plt.fill_between(t, y_lqr_mean[:,i,0] + y_lqr_std[:,i,0],
                             y_lqr_mean[:,i, 0] - y_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
            plt.plot(t, y_mean[:,:,0], 'tab:blue', label='WDRC')
            plt.fill_between(t, y_mean[:,i,0] + y_std[:,i,0],
                             y_mean[:,i, 0] - y_std[:,i,0], facecolor='tab:blue', alpha=0.3)
            plt.xlabel(r'$t$', fontsize=20)
            plt.ylabel(r'$y_{}$'.format(i+1), fontsize=20)
            plt.legend(fontsize=18)
            plt.grid()
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlim([t[0], t[-1]])

            plt.savefig(path +'outputs_{}_{}.pdf'.format(i+1,num), dpi=300, bbox_inches="tight")
            plt.clf()


        plt.title('Optimal Value')
        t = np.arange(T+1)

        if J_lqr_list != []:
            plt.plot(t, J_lqr_mean, 'tab:red', label='LQG')
            plt.fill_between(t, J_lqr_mean + 0.25*J_lqr_std, J_lqr_mean - 0.25*J_lqr_std, facecolor='tab:red', alpha=0.3)

        plt.plot(t, J_mean, 'tab:blue', label='WDRC')
        plt.fill_between(t, J_mean + 0.25*J_std, J_mean - 0.25*J_std, facecolor='tab:blue', alpha=0.3)
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel(r'$V_t(x_t)$', fontsize=18)
        plt.legend(fontsize=18)
        plt.grid()
        plt.xlim([t[0], t[-1]])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig(path +'J_{}.pdf'.format(num), dpi=300, bbox_inches="tight")
        plt.clf()


        ax = fig.gca()
        t = np.arange(T+1)
        max_bin = np.max([J_ar[:,0], J_lqr_ar[:,0]])
        min_bin = np.min([J_ar[:,0], J_lqr_ar[:,0]])

        ax.hist(J_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:blue', label='WDRC', alpha=0.5, linewidth=0.5, edgecolor='tab:blue')
        ax.hist(J_lqr_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:red', label='LQG', alpha=0.5, linewidth=0.5, edgecolor='tab:red')

        ax.axvline(J_ar[:,0].mean(), color='navy', linestyle='dashed', linewidth=1.5)
        ax.axvline(J_lqr_ar[:,0].mean(), color='maroon', linestyle='dashed', linewidth=1.5)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1, 0]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)

        ax.grid()
        ax.set_axisbelow(True)
        plt.xlabel(r'Total Cost', fontsize=14)
        plt.ylabel(r'Frequency', fontsize=14)

        plt.savefig(path +'J_hist_{}.pdf'.format(num), dpi=300, bbox_inches="tight")
        plt.clf()


        plt.close('all')

    print('cost: {} ({})'.format(J_mean[0], J_std[0]) , 'cost_lqr:{} ({})'.format(J_lqr_mean[0],J_lqr_std[0]))
    print('time: {} ({})'.format(time_ar.mean(), time_ar.std()), 'time_lqr: {} ({})'.format(time_lqr_ar.mean(), time_lqr_ar.std()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform)
    parser.add_argument('--sim_type', required=False, default="multiple", type=str) #type of simulation runs (single or multiple)
    parser.add_argument('--num_sim', required=False, default=1000, type=int) #number of simulation runs to plot
    args = parser.parse_args()

    print('\n-------Summary-------')
    if args.sim_type == "multiple":
        path = "./results/{}/multiple/".format(args.dist)
        #Load data
        lqg_file = open(path + 'lqg.pkl', 'rb')
        wdrc_file = open(path + 'wdrc.pkl', 'rb')
        lqg_data = pickle.load(lqg_file)
        wdrc_data = pickle.load(wdrc_file)
        lqg_file.close()
        wdrc_file.close()

        #Plot and Summarize
        summarize(lqg_data, wdrc_data, args.dist, path, args.num_sim)
    else:
        path = "./results/{}/single/".format(args.dist)

        for i in range(args.num_sim):
            print('i: ', i)
            #Load data
            lqg_file = open(path + 'lqg.pkl', 'rb')
            wdrc_file = open(path + 'wdrc.pkl', 'rb')
            lqg_data = pickle.load(lqg_file)
            wdrc_data = pickle.load(wdrc_file)
            lqg_file.close()
            wdrc_file.close()

            #Plot and Summarize
            summarize(lqg_data, wdrc_data, args.dist, path, i)
            print('---------------------')

