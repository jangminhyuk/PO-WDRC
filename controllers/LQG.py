#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from numba import cuda

import numpy as np
import time

class LQG:
    def __init__(self, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, M_hat, M0, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, true_v_init):
        self.dist = dist
        self.noise_dist = noise_dist
        self.T = T
        self.A, self.B, self.C, self.Q, self.Qf, self.R, self.M = system_data # The Controller can only use nominal M_hat
        self.nx = self.B.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]
        self.x0_mean = x0_mean
        self.x0_cov = x0_cov
        self.mu_hat = mu_hat
        self.Sigma_hat = Sigma_hat
        self.mu_hat0 = mu_hat[0]
        self.Sigma_hat0 = Sigma_hat[0]
        self.M_hat = M_hat
        self.M0 = M0
        self.mu_w = mu_w
        self.Sigma_w = Sigma_w
        if self.dist=="uniform":
            self.x0_max = x0_max
            self.x0_min = x0_min
            self.w_max = w_max
            self.w_min = w_min

        
        # if noise_dist == "normal":
        #     self.true_v_init = self.normal(np.zeros((self.ny,1)), self.M) # not used in code !!
        # elif noise_dist == "uniform":
        #     self.v_max = v_max
        #     self.v_min = v_min
        #     self.true_v_init = self.uniform(v_max, v_min) # not used in code !!
        
        self.true_v_init = true_v_init
        if noise_dist == "uniform":
            self.v_max = v_max
            self.v_min = v_min
            
        
        #Initial state
        if self.dist=="normal":
            self.x0_init = self.normal(self.x0_mean, self.x0_cov)
        elif self.dist=="uniform":
            self.x0_init = self.uniform(self.x0_max, self.x0_min)
            
        self.J = np.zeros(self.T+1)
        self.P = np.zeros((self.T+1, self.nx, self.nx))
        self.S = np.zeros((self.T+1, self.nx, self.nx))
        self.r = np.zeros((self.T+1, self.nx, 1))
        self.z = np.zeros(self.T+1)
        self.K = np.zeros(( self.T, self.nu, self.nx))
        self.L = np.zeros(( self.T, self.nu, 1))


    def uniform(self, a, b, N=1):
        n = a.shape[0]
        x = a + (b-a)*np.random.rand(N,n)
        return x.T

    def normal(self, mu, Sigma, N=1):
        n = mu.shape[0]
        w = np.random.normal(size=(N,n))
        if (Sigma == 0).all():
            x = mu
        else:
            x = mu + np.linalg.cholesky(Sigma) @ w.T
        return x

    def kalman_filter(self, x, P, y, M_hat, mu_w = None, P_w = None, u = None):
        #Performs state estimation based on the current state estimate, control input and new observation
        if u is None:
            #Initial state estimate
            x_ = x
            P_ = P
        else:
            #Prediction update
            x_ = self.A @ x + self.B @ u + mu_w
            P_ = self.A @ P @ self.A.T + P_w

        #Measurement update
        resid = y - self.C @ x_

        temp = np.linalg.solve(self.C @ P_ @ self.C.T + M_hat, self.C @ P_)
        P_new = P_ - P_ @ self.C.T @ temp
        x_new = x_ + P_new @ self.C.T @ np.linalg.inv(M_hat) @ resid
        return x_new, P_new

    def riccati(self, Phi, P, S, r, z, Sigma_hat, mu_hat):
        #Riccati equation for standard LQG

        temp = np.linalg.inv(np.eye(self.nx) + P @ Phi)
        P_ = self.Q + self.A.T @ temp @ P @ self.A
        S_ = self.Q + self.A.T @ (P + S) @ self.A - P_
        r_ = self.A.T @ temp @ (r + P @ mu_hat)
        # IN Z, np.trace(S+P ...) part means the LQG part
        z_ = z + np.trace((S + P) @ Sigma_hat) \
            + (2*mu_hat - Phi @ r).T @ temp @ r + mu_hat.T @ temp @ P @ mu_hat
        temp2 = np.linalg.solve(self.R, self.B.T)
        K = - temp2 @ temp @ P @ self.A
        L = - temp2 @ temp @ (r + P @ mu_hat)   
        return P_, S_, r_, z_, K, L

    def get_obs(self, x, v):
        #Get new noisy observation
        obs = self.C @ x + v
        return obs

    def backward(self):
        #Compute P, S, r, z, K and L backward in time

        self.P[self.T] = self.Qf
        Phi = self.B @ np.linalg.inv(self.R) @ self.B.T
        for t in range(self.T-1, -1, -1):
            #self.P[t], self.S[t], self.r[t], self.z[t], self.K[t], self.L[t]  = self.riccati(Phi, self.P[t+1], self.S[t+1], self.r[t+1], self.z[t+1], self.Sigma_hat[t], self.mu_hat[t])
            self.P[t], self.S[t], self.r[t], self.z[t], self.K[t], self.L[t]  = self.riccati(Phi, self.P[t+1], self.S[t+1], self.r[t+1], self.z[t+1], self.Sigma_hat0, self.mu_hat0) #
   
    def forward(self, true_w, true_v):
        #Apply the controller forward in time.
        start = time.time()
        x = np.zeros((self.T+1, self.nx, 1))
        y = np.zeros((self.T+1, self.ny, 1))
        u = np.zeros((self.T, self.nu, 1))

        x_mean = np.zeros((self.T+1, self.nx, 1))
        x_cov = np.zeros((self.T+1, self.nx, self.nx))
        J = np.zeros(self.T+1)

        x[0] = self.x0_init
        y[0] = self.get_obs(x[0], self.true_v_init) #initial observation
        x_mean[0], x_cov[0] = self.kalman_filter(self.x0_mean, self.x0_cov, y[0], self.M0) #initial state estimation
        
        for t in range(self.T):
            # #disturbance sampling
            # if self.dist=="normal":
            #     true_w = self.normal(self.mu_w, self.Sigma_w)
            # elif self.dist=="uniform":
            #     true_w = self.uniform(self.w_max, self.w_min)
            
            # #observation noise
            # if self.noise_dist == "normal":
            #     true_v = self.normal(np.zeros((self.ny,1)), self.M) #observation noise
            # elif self.noise_dist == "uniform":
            #     true_v = self.uniform(self.v_max, self.v_min)

            #Apply the control input to the system
            u[t] = self.K[t] @ x_mean[t] + self.L[t]
            x[t+1] = self.A @ x[t] + self.B @ u[t] + true_w[t].reshape((-1,1))
            y[t+1] = self.get_obs(x[t+1], true_v[t].reshape((-1,1)))

            #Update the state estimation (using the nominal mean and covariance)
            #x_mean[t+1], x_cov[t+1] = self.kalman_filter(x_mean[t], x_cov[t], y[t+1], self.M_hat[t], self.mu_hat[t], self.Sigma_hat[t], u=u[t])
            x_mean[t+1], x_cov[t+1] = self.kalman_filter(x_mean[t], x_cov[t], y[t+1], self.M0, self.mu_hat0, self.Sigma_hat0, u=u[t])

            #Compute the total cost
            J[self.T] = x[self.T].T @ self.Qf @ x[self.T]
            for t in range(self.T-1, -1, -1):
                J[t] = J[t+1] + x[t].T @ self.Q @ x[t] + u[t].T @ self.R @ u[t]

        end = time.time()
        time_ = end-start
        return {'comp_time': time_,
                'state_traj': x,
                'output_traj': y,
                'control_traj': u,
                'cost': J}