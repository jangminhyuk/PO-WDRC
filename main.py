#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from numba import jit, cuda

import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from controllers.PO_WDRC import PO_WDRC
from plot import summarize
import os
import pickle

def uniform(a, b, N=1):
    n = a.shape[0]
    x = a + (b-a)*np.random.rand(N,n)# N by n matrix
    return x.T

def normal(mu, Sigma, N=1):
    n = mu.shape[0]
    w = np.random.normal(size=(N,n))
    if (Sigma == 0).all():
        x = mu
    else:
        x = mu + np.linalg.cholesky(Sigma) @ w.T # Sigma = LL^T (Cholesky decomposition)
    return x

def gen_sample_dist(dist, T, N_sample, mu_w=None, Sigma_w=None, w_max=None, w_min=None):
    if dist=="normal":
        w = normal(mu_w, Sigma_w, N=N_sample)
    elif dist=="uniform":
        w = uniform(w_max, w_min, N=N_sample)
        
    mean_ = np.average(w, axis = 1)
    diff = (w.T - mean_)[...,np.newaxis]
    var_ = np.average( (diff @ np.transpose(diff, (0,2,1))) , axis = 0)
    return np.tile(mean_[...,np.newaxis], (T, 1, 1)), np.tile(var_, (T, 1, 1))

def create_matrices(nx, ny, nu):
    # A_ = np.random.rand(nx,nx)
    # eigs = np.linalg.eigvals(A_)
    # max_eig = np.max(eigs)
    # min_eig = np.min(eigs)
    # A = 1.02*A_/(np.max([np.abs(max_eig),np.abs(min_eig)])) #The matrix A is scaled so that it is unstable ??
    # B = -3 + np.random.rand(nx,nu)*6
    # C = -3 + np.random.rand(ny,nx)*6
    
    
    # A = np.array([[0.518, 0.266, 0.331, 0.142], 
    #               [0.405, 0.806, 0.125, 0.416],
    #               [0.318, 0.466, 0.231, 0.242],
    #               [0.588, 0.815, 0.411, 0.414]])
    
    A = np.array([[0.518, 0.266],[0.405, 0.806]])
    B = np.array([[-2.972],[-2.271]])
    C = np.array([[1.023, 1.955]])
    
    #C = np.array([[1,0,0,0],[0,1,0,0]]) # only first two variables are observable
    return A, B, C

def save_data(path, data):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()

def main(dist, noise_dist, sim_type, num_sim, num_samples, num_noise_samples, T, plot_results):

    #Path for saving the results
    if sim_type == "multiple":
        path = "./results/{}/finite/multiple/".format(dist)
    else:
        path = "./results/{}/finite/single/".format(dist)
    if not os.path.exists(path):
        os.makedirs(path)

    #-------Initialization-------
    nx = 2 #state dimension
    nu = 1 #control input dimension
    ny = 1 #output dimension
    A, B, C = create_matrices(nx, nu, ny) #system matrices generation
    #cost weights
    Q = np.eye(nx)
    Qf = Q
    R = np.eye(nu)

    # ----------System disturbance parameters -----------
    if dist =="uniform":
        #M = 0.1*np.eye(ny) #observation noise covariance
        theta = 0.03 #Wasserstein ball radius
        #disturbance distribution parameters
        w_max = 0.2*np.ones(nx)
        w_min = -0.2*np.ones(nx)
        mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
        Sigma_w = 1/12*np.diag((w_max - w_min)**2)
        #initial state distribution parameters
        # x0_max = np.array([0.3, 0.3, 0.3, 0.3])
        # x0_min = np.array([0.1, 0.1, 0.1, 0.1])
        x0_max = np.array([0.3, 0.5])
        x0_min = np.array([0.1, 0.2])
        x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
        x0_cov = 1/12*np.diag((x0_max - x0_min)**2)

    elif dist == "normal":
        #M = 0.2*np.eye(ny) #observation noise covariance
        theta = 0.1 #Wasserstein ball radius
        #disturbance distribution parameters
        w_max = None
        w_min = None
        # mu_w = np.array([[0.01],[0.02],[0.01],[0.02]])
        # Sigma_w= np.array([[0.01, 0.005, 0.005, 0.005],
        #                    [0.005, 0.01, 0.005, 0.005],
        #                    [0.005, 0.005, 0.01, 0.005],
        #                    [0.005, 0.005, 0.005, 0.01]
        #                    ])
        mu_w = np.array([[0.01], [0.02]])
        Sigma_w = np.array([[0.01,0.005],[0.005, 0.01]])
        
        #initial state distribution parameters
        x0_max = None
        x0_min = None
        # x0_mean = np.array([[-1],[-1],[-1],[-1]])
        x0_mean = np.array([[-1],[-1]])
        x0_cov = 0.001*np.eye(nx)

    #--------Observation Noise Parameters-------------
    if noise_dist == "normal":
        v_max = None
        v_min = None
        # mu_v = np.array([[0.0],[0.0]])
        mu_v = np.array([[0.0]])
        M = 0.5*np.eye(ny)
        true_v_init = normal(np.zeros((ny,1)), M)
    elif noise_dist == "uniform":
        v_max = 0.5*np.ones(ny)
        v_min = -0.5*np.ones(ny)
        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
        M = 1/12*np.diag((v_max - v_min)**2)
        true_v_init = uniform(v_max, v_min)


    #-------Estimate the nominal distribution-------
    # system disturbances
    mu_hat, Sigma_hat = gen_sample_dist(dist, T, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
    
    # observation noises
    _, M0 = gen_sample_dist(noise_dist, 1, num_noise_samples, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min) # generate initial M0
    
    #wil not be used!
    v_hat, M_hat = gen_sample_dist(noise_dist, T, num_noise_samples, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min) # generate M hat!!!!!!!
    
    
    
    #-------Create a random system-------
    system_data = (A, B, C, Q, Qf, R, M)

    #-------Perform n  independent simulations and summarize the results-------
    output_lqg_list = []
    output_wdrc_list = []
    output_po_wdrc_list = []
    
    #Initialize WDRC and LQG controllers
    wdrc = WDRC(theta, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, M_hat, M0[0], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, true_v_init)
    po_wdrc = PO_WDRC(theta, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, M_hat, M0[0], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, true_v_init)
    lqg = LQG(T, dist, noise_dist, system_data, mu_hat, Sigma_hat, M_hat, M0[0], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, true_v_init)

    print('---------------------')
    for i in range(num_sim):
        print('i: ', i)
        
        #Recursively compute the value function and control matrices
        wdrc.backward()
        lqg.backward()
        po_wdrc.backward()
        
        #-------Generate True disturbance and noise vector that will be applied to the system------#
        # Apply same noises to each controller in order to compare performance in the same condition
        #System disturbance
        if dist=="normal":
            true_w = normal(mu_w, Sigma_w, T).T
        elif dist=="uniform":
            true_w = uniform(w_max, w_min, T).T    
        #observation noise
        if noise_dist == "normal":
            true_v = normal(np.zeros((ny,1)), M, T).T
        elif noise_dist == "uniform":
            true_v = uniform(v_max, v_min, T).T
        
        #-------------------------------------------------------
        #Perform state estimation and apply the controller
        output_wdrc = wdrc.forward(true_w, true_v)
        output_lqg = lqg.forward(true_w, true_v)
        output_po_wdrc = po_wdrc.forward(true_w, true_v)
        
        output_wdrc_list.append(output_wdrc)
        output_lqg_list.append(output_lqg)
        output_po_wdrc_list.append(output_po_wdrc)
        
        print('cost (LQG):', output_lqg['cost'][0], 'time (LQG):', output_lqg['comp_time'])
        print('cost (WDRC):', output_wdrc['cost'][0], 'time (WDRC):', output_wdrc['comp_time'])
        print('cost (DR_KF_WDRC):', output_po_wdrc['cost'][0], 'time (DR_KF_WDRC):', output_po_wdrc['comp_time'])

    #Save results
    save_data(path + 'lqg.pkl', output_lqg_list)
    save_data(path + 'wdrc.pkl', output_wdrc_list)
    save_data(path + 'po_wdrc.pkl', output_po_wdrc_list)

    #Summarize and plot the results
    print('\n-------Summary-------')
    print("System disturbance : ", dist, "/ Observation noise : ", noise_dist)
    if sim_type == "multiple":
        summarize(output_lqg_list, output_wdrc_list, output_po_wdrc_list, dist, path, num_sim, plot_results)
    else:
        for i in range(num_sim):
            print('i: ', i)
            summarize([output_lqg_list[i]], [output_wdrc_list[i]], [output_po_wdrc_list[i]], dist, path, i, plot_results)
            print('---------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #system disurbance distribution (normal or uniform)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #Observation noise distribution (normal or uniform)
    parser.add_argument('--sim_type', required=False, default="multiple", type=str) #simulation type (single or multiple)
    parser.add_argument('--num_sim', required=False, default=100, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=5, type=int) #number of disturbance samples
    parser.add_argument('--num_noise_samples', required=False, default=5, type=int) #number of noise samples
    parser.add_argument('--horizon', required=False, default=50, type=int) #horizon length
    parser.add_argument('--plot', required=False, action="store_true") #plot results+


    args = parser.parse_args()
    np.random.seed(100)
    main(args.dist, args.noise_dist, args.sim_type, args.num_sim, args.num_samples, args.num_noise_samples, args.horizon, args.plot) 