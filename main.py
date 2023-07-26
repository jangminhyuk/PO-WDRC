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
    A_ = np.random.rand(nx,nx)
    eigs = np.linalg.eigvals(A_)
    max_eig = np.max(eigs)
    min_eig = np.min(eigs)
    A = 1.02*A_/(np.max([np.abs(max_eig),np.abs(min_eig)])) #The matrix A is scaled so that it is unstable ??
    B = -3 + np.random.rand(nx,nu)*6
    C = -3 + np.random.rand(ny,nx)*6
    return A, B, C

def save_data(path, data):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()

def main(dist, sim_type, num_sim, num_samples, T, plot_results):

    #Path for saving the results
    if sim_type == "multiple":
        path = "./results/{}/finite/multiple/".format(dist)
    else:
        path = "./results/{}/finite/single/".format(dist)
    if not os.path.exists(path):
        os.makedirs(path)

    #-------Initialization-------
    nx = 4 #state dimension
    nu = 2 #control input dimension
    ny = 2 #output dimension
    A, B, C = create_matrices(nx, nu, ny) #system matrices generation
    #cost weights
    Q = np.eye(nx)
    Qf = Q
    R = np.eye(nu)

    if dist =="uniform":
        M = 0.1*np.eye(ny) #observation noise covariance
        theta = 0.03 #Wasserstein ball radius
        #disturbance distribution parameters
        w_max = 0.05*np.ones(nx)
        w_min = -0.05*np.ones(nx)
        mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
        Sigma_w = 1/12*np.diag((w_max - w_min)**2)
        #initial state distribution parameters
        x0_max = np.array([0.3, 0.5, 0.4, 0.4])
        x0_min = np.array([0.1, 0.2, 0.1, 0.2])
        x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
        x0_cov = 1/12*np.diag((x0_max - x0_min)**2)

    elif dist == "normal":
        M = 0.2*np.eye(ny) #observation noise covariance
        theta = 0.1 #Wasserstein ball radius
        #disturbance distribution parameters
        w_max = None
        w_min = None
        mu_w = np.array([[0.01],[0.02],[0.01],[0.02]])
        Sigma_w= np.array([[0.01, 0.005, 0.005, 0.005],
                           [0.005, 0.01, 0.005, 0.005],
                           [0.005, 0.005, 0.01, 0.005],
                           [0.005, 0.005, 0.005, 0.01]
                           ])
        #initial state distribution parameters
        x0_max = None
        x0_min = None
        x0_mean = np.array([[-1],[-1],[-1],[-1]])
        x0_cov = 0.001*np.eye(nx)

    #observation noise paramters (normal noise)
    v_max = None
    v_min = None
    mu_v = np.array([[0.0],[0.0]])
    _, M0 = gen_sample_dist("normal", 1, num_samples, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min) # generate initial M0
    
    #-------Estimate the nominal distribution-------
    mu_hat, Sigma_hat = gen_sample_dist(dist, T, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
    
    v_hat, M_hat = gen_sample_dist("normal", T, num_samples, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min) # generate M hat!!!!!!!
    
    #-------Create a random system-------
    system_data = (A, B, C, Q, Qf, R, M)

    #-------Perform n  independent simulations and summarize the results-------
    output_lqg_list = []
    output_wdrc_list = []
    output_po_wdrc_list = []
    
    #Initialize WDRC and LQG controllers
    #wdrc = WDRC(theta, T, dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min)
    po_wdrc = PO_WDRC(theta, T, dist, system_data, mu_hat, Sigma_hat, M_hat, M0, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min)
    lqg = LQG(T, dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min)

    print('---------------------')
    for i in range(num_sim):
        print('i: ', i)
        
        #Recursively compute the value function and control matrices
        #wdrc.backward()
        lqg.backward()
        po_wdrc.backward()
        
        #Perform state estimation and apply the controller
        #output_wdrc = wdrc.forward()
        output_lqg = lqg.forward()
        output_po_wdrc = po_wdrc.forward()
        
        #output_wdrc_list.append(output_wdrc)
        output_lqg_list.append(output_lqg)
        output_po_wdrc_list.append(output_po_wdrc)
        
        #print('cost (WDRC):', output_wdrc['cost'][0], 'time (WDRC):', output_wdrc['comp_time'])
        print('cost (LQG):', output_lqg['cost'][0], 'time (LQG):', output_lqg['comp_time'])
        print('cost (PO_WDRC):', output_po_wdrc['cost'][0], 'time (PO_WDRC):', output_po_wdrc['comp_time'])

    #Save results
    save_data(path + 'wdrc.pkl', output_wdrc_list)
    save_data(path + 'lqg.pkl', output_lqg_list)
    save_data(path + 'po_wdrc.pkl', output_po_wdrc_list)

    #Summarize and plot the results
    print('\n-------Summary-------')
    if sim_type == "multiple":
        summarize(output_lqg_list, output_wdrc_list, dist, path, num_sim, plot_results)
    else:
        for i in range(num_sim):
            print('i: ', i)
            summarize([output_lqg_list[i]], [output_wdrc_list[i]], dist, path, i, plot_results)
            print('---------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform)
    parser.add_argument('--sim_type', required=False, default="multiple", type=str) #simulation type (single or multiple)
    parser.add_argument('--num_sim', required=False, default=1000, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=5, type=int) #number of disturbance samples
    parser.add_argument('--horizon', required=False, default=50, type=int) #horizon length
    parser.add_argument('--plot', required=False, action="store_true") #plot results+


    args = parser.parse_args()
    np.random.seed(100)
    main(args.dist, args.sim_type, args.num_sim, args.num_samples, args.horizon, args.plot)