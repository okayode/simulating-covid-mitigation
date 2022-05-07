#!/usr/bin/env python3

import numpy as np
#import tensorflow.keras as keras
import pandas as pd
import sys
import tensorflow.compat.v1 as tf
import timeit
import time
import csv
import datetime
import scipy.io
import scipy.optimize
from scipy import optimize
from scipy.interpolate import CubicSpline
#from statsmodels.tsa.holtwinters import SimpleExpSmothing, Holt
from tqdm import tqdm

tf.disable_v2_behavior()

##########################################################################################################
# load data
df0 = pd.read_csv('world_recovered.csv')
df1 = pd.read_csv('world_confirmed.csv')

##########################################################################################################
# process data

today = '12/11/20' # Update this to include more data 
days = pd.date_range(start='1/31/20',end=today) 
dd = np.arange(len(days))

total_cases = [df1[day.strftime('%-m/%-d/%y')].sum() for day in days] 
total_recov = [df0[day.strftime('%-m/%-d/%y')].sum() for day in days] 

row_r=df0['Country_Region'].tolist().index('Italy')
total_recov = [df0[day.strftime('%-m/%-d/%y')][row_r] for day in days]

row_c=df1['Country_Region'].tolist().index('Italy')
total_cases = [df1[day.strftime('%-m/%-d/%y')][row_c] for day in days]

t = np.reshape(dd, [-1])
R = np.reshape(total_recov, [-1])
new_R = R*100/(60.36*10**6) # rescale y-axis

I = np.reshape(total_cases, [-1])
new_I = I*100/(60.36*10**6) # rescale y-axis


# generating more data points for training
nd = 3000
cs1 = CubicSpline(t,new_I)
cs2 = CubicSpline(t,new_R)

Td = np.linspace(0,315,nd)

cs_I = cs1(Td)
cs_R = cs2(Td)


class PINN_constantParam:
    # Initialize the class
    def __init__(self, t, I, R, layers, lb, ub):
        
        self.lb = lb
        self.ub = ub
        
        self.t = t
        
        self.I = I
        self.R = R
        
        self.layers = layers
        
        self.weights, self.biases = self.initialize_NN(layers)
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.beta = tf.Variable([1.0], dtype=tf.float32)
        self.gamma = tf.Variable([1.0], dtype=tf.float32)
        self.xi = tf.Variable([1.0], dtype=tf.float32)
        self.mu = tf.Variable([1.0], dtype=tf.float32)
        
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.I_tf = tf.placeholder(tf.float32, shape=[None, self.I.shape[1]])
        self.R_tf = tf.placeholder(tf.float32, shape=[None, self.R.shape[1]])
        
                
        
        self.S_pred, self.I_pred, self.J_pred, self.R_pred, self.U_pred = self.net_ASIR(self.t_tf)
        
        self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7 = self.net_l(self.t_tf)
        
        self.loss = tf.reduce_mean(tf.square(self.I_tf - self.I_pred)) + \
            tf.reduce_mean(tf.square(self.I_tf[0]*((1-self.xi)/self.xi) - self.J_pred[0])) + \
            tf.reduce_mean(tf.square(self.R_tf - self.R_pred)) + \
            tf.reduce_mean(tf.square(self.R_tf[0]*((1-self.xi)/self.xi) - self.U_pred[0])) + \
            tf.reduce_mean(tf.square(self.l1)) + \
            tf.reduce_mean(tf.square(self.l2)) + \
            tf.reduce_mean(tf.square(self.l3)) + \
            tf.reduce_mean(tf.square(self.l4)) + \
            tf.reduce_mean(tf.square(self.l5)) + \
            tf.reduce_mean(tf.square(self.l6)) + \
            tf.reduce_mean(tf.square(self.l7))
        
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = self.optimizer.minimize(self.loss)
        self.loss_log = []
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, t, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(t - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        #Y = tf.nn.softplus(tf.add(tf.matmul(H, W), b))
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    
    def net_ASIR(self, t):
        ASIR = self.neural_net(t, self.weights, self.biases)
        S = ASIR[:,0:1]
        I = ASIR[:,1:2]
        J = ASIR[:,2:3]
        R = ASIR[:,3:4]
        U = ASIR[:,4:5]

        return S, I, J, R, U
    
    def net_l(self, t):
        NN = 100
        gamma = self.gamma
        beta = self.beta
        xi = self.xi
        mu = self.mu
        
        S, I, J, R, U = self.net_ASIR(t)
        S_t = tf.gradients(S, t)[0]
        I_t = tf.gradients(I, t)[0]
        J_t = tf.gradients(J, t)[0]
        R_t = tf.gradients(R, t)[0]
        U_t = tf.gradients(U, t)[0]
        
        l1 = S_t + beta*(I+J)*S/NN 
        l2 = I_t - xi*beta*(I+J)*S/NN + gamma*I
        l3 = J_t - (1-xi)*beta*(I+J)*S/NN + mu*J
        l4 = R_t - gamma*I 
        l5 = U_t - mu*J 
        
        l6 = NN - (S + I + J + R + U)
        l7 = 0 - (S_t + I_t + J_t + R_t + U_t)
        
        return l1, l2, l3, l4, l5, l6, l7
        
        
    def train(self, nIter):
        tf_dict = {self.t_tf: self.t, self.I_tf: self.I, self.R_tf: self.R}
        start_time = timeit.default_timer()

        for it in tqdm(range(nIter)):
            self.sess.run(self.train_op, tf_dict)
            if it % 100 == 0:
               elapsed = timeit.default_timer() - start_time
               loss_value = self.sess.run(self.loss, tf_dict)
               self.loss_log.append(loss_value)
               beta_value = self.sess.run(self.beta)
               xi_value = self.sess.run(self.xi)
               gamma_value = self.sess.run(self.gamma)
               mu_value = self.sess.run(self.mu)
               start_time = timeit.default_timer()
        
        
        
    def predict(self, t_star):
        tf_dict = {self.t_tf: t_star}
        
        S_star = self.sess.run(self.S_pred, tf_dict)
        I_star = self.sess.run(self.I_pred, tf_dict)
        J_star = self.sess.run(self.J_pred, tf_dict)
        R_star = self.sess.run(self.R_pred, tf_dict)
        U_star = self.sess.run(self.U_pred, tf_dict)
        
        return S_star, I_star, J_star, R_star, U_star
        

##########################################################################################################
# training the network

niter = 40000  # number of Epochs
layers = [1, 64, 64, 64, 64, 5]
t_train = Td.flatten()[:,None]
I_train = cs_I.flatten()[:,None]
R_train = cs_R.flatten()[:,None]         

# Doman bounds
lb = t_train.min(0)
ub = t_train.max(0)

model = PINN_constantParam(t_train, I_train, R_train, layers, lb, ub)
model.train(niter)

# prediction
S_pred, I_pred, J_pred, R_pred, U_pred = model.predict(t_train)


mse_train_loss = model.loss_log
rmse_train_loss = np.sqrt(mse_train_loss)
print("rmse_train_loss:",*["%.8f"%(x) for x in rmse_train_loss[0:400]])

# flatten array
T0 = t.flatten()
T1 = t_train.flatten()
I0 = new_I.flatten()
R0 = new_R.flatten()
S1 = S_pred.flatten()
I1 = I_pred.flatten()
J1 = J_pred.flatten()
R1 = R_pred.flatten()
U1 = U_pred.flatten()

# convert float to list
T0 = T0.tolist()
T1 = T1.tolist()
I0 = I0.tolist()
R0 = R0.tolist()
S1 = S1.tolist()
I1 = I1.tolist()
J1 = J1.tolist()
R1 = R1.tolist()
U1 = U1.tolist()

print("days:",*["%.8f"%(x) for x in T0[0:nd]])
print("time:",*["%.8f"%(x) for x in T1[0:nd]])
print("cases:",*["%.8f"%(x) for x in I0[0:nd]])
print("wellness:",*["%.8f"%(x) for x in R0[0:nd]])
print("susceptible:",*["%.8f"%(x) for x in S1[0:nd]])
print("infectd:",*["%.8f"%(x) for x in I1[0:nd]])
print("INFasymp:",*["%.8f"%(x) for x in J1[0:nd]])
print("recoverd:",*["%.8f"%(x) for x in R1[0:nd]])
print("RECasymp:",*["%.8f"%(x) for x in U1[0:nd]])


beta_value = model.sess.run(model.beta)
xi_value = model.sess.run(model.xi)
gamma_value = model.sess.run(model.gamma)
mu_value = model.sess.run(model.mu)


# learned parameters
print("beta:",*["%.8f"%(x) for x in beta_value])
print("xi:",*["%.8f"%(x) for x in xi_value])
print("gamma:",*["%.8f"%(x) for x in gamma_value])
print("mu:",*["%.8f"%(x) for x in mu_value])

##########################################################################################################



