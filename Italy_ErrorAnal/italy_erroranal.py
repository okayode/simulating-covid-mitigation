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
        #self.xi = xi
        
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
            #if it % 100 == 0:
            elapsed = timeit.default_timer() - start_time
            loss_value = self.sess.run(self.loss, tf_dict)
            self.loss_log.append(loss_value)
            beta_value = self.sess.run(self.beta)
            #xi_value = self.sess.run(self.xi)
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

niter = 60000  # number of Epochs
layers1 = [1, 64, 64, 64, 5]
t_train = Td.flatten()[:,None]
I_train = cs_I.flatten()[:,None]
R_train = cs_R.flatten()[:,None]      
#D_train = cs_D.flatten()[:,None]      

# Doman bounds
lb = t_train.min(0)
ub = t_train.max(0)


from sklearn.model_selection import train_test_split

# random splits
T_train1,T_test1,I_train1,I_test1 = train_test_split(t_train,I_train,test_size=0.4,random_state=5)
T_train2,T_test2,R_train1,R_test1 = train_test_split(t_train,R_train,test_size=0.4,random_state=5)

T_train = np.sort(T_train1,axis=0)
T_test = np.sort(T_test1,axis=0)
I_train1 = np.sort(I_train1,axis=0)
I_test1 = np.sort(I_test1,axis=0)
R_train1 = np.sort(R_train1,axis=0)
R_test1 = np.sort(R_test1,axis=0)

# Epochs
model1 = PINN_constantParam(T_train, I_train1, R_train1, layers1, lb, ub)
model2 = PINN_constantParam(T_test, I_test1, R_test1, layers1, lb, ub)

model1.train(niter)
model2.train(niter)

mse_train_loss = model1.loss_log
mse_test_loss = model2.loss_log

print("mseTrainLoss:",*["%.8f"%(x) for x in mse_train_loss])
print("mseTestLoss:",*["%.8f"%(x) for x in mse_test_loss])

# Depth
niter2 = 30000
layersA = [1, 64, 5]
layersB = [1, 64, 64, 5]
layersC = [1, 64, 64, 64, 5]
layersD = [1, 64, 64, 64, 64, 5]
layersE = [1, 64, 64, 64, 64, 64, 5]
layersF = [1, 64, 64, 64, 64, 64, 64, 5]

model1A = PINN_constantParam(T_train, I_train1, R_train1, layersA, lb, ub)
model2A = PINN_constantParam(T_test, I_test1, R_test1, layersA, lb, ub)
model1B = PINN_constantParam(T_train, I_train1, R_train1, layersB, lb, ub)
model2B = PINN_constantParam(T_test, I_test1, R_test1, layersB, lb, ub)
model1C = PINN_constantParam(T_train, I_train1, R_train1, layersC, lb, ub)
model2C = PINN_constantParam(T_test, I_test1, R_test1, layersC, lb, ub)
model1D = PINN_constantParam(T_train, I_train1, R_train1, layersD, lb, ub)
model2D = PINN_constantParam(T_test, I_test1, R_test1, layersD, lb, ub)
model1E = PINN_constantParam(T_train, I_train1, R_train1, layersE, lb, ub)
model2E = PINN_constantParam(T_test, I_test1, R_test1, layersE, lb, ub)
model1F = PINN_constantParam(T_train, I_train1, R_train1, layersF, lb, ub)
model2F = PINN_constantParam(T_test, I_test1, R_test1, layersF, lb, ub)


model1A.train(niter2); model2A.train(niter2); model1B.train(niter2); model2B.train(niter2)
model1C.train(niter2); model2C.train(niter2); model1D.train(niter2); model2D.train(niter2)
model1E.train(niter2); model2E.train(niter2); model1F.train(niter2); model2F.train(niter2)

mse_train_depthA = model1A.loss_log; mse_test_depthA = model2A.loss_log
mse_train_depthB = model1B.loss_log; mse_test_depthB = model2B.loss_log
mse_train_depthC = model1C.loss_log; mse_test_depthC = model2C.loss_log
mse_train_depthD = model1D.loss_log; mse_test_depthD = model2D.loss_log
mse_train_depthE = model1E.loss_log; mse_test_depthE = model2E.loss_log
mse_train_depthF = model1F.loss_log; mse_test_depthF = model2F.loss_log

print("mseTrainDepthA:",*["%.8f"%(x) for x in mse_train_depthA])
print("mseTestDepthA:",*["%.8f"%(x) for x in mse_test_depthA])

print("mseTrainDepthB:",*["%.8f"%(x) for x in mse_train_depthB])
print("mseTestDepthB:",*["%.8f"%(x) for x in mse_test_depthB])

print("mseTrainDepthC:",*["%.8f"%(x) for x in mse_train_depthC])
print("mseTestDepthC:",*["%.8f"%(x) for x in mse_test_depthC])

print("mseTrainDepthD:",*["%.8f"%(x) for x in mse_train_depthD])
print("mseTestDepthD:",*["%.8f"%(x) for x in mse_test_depthD])

print("mseTrainDepthE:",*["%.8f"%(x) for x in mse_train_depthE])
print("mseTestDepthE:",*["%.8f"%(x) for x in mse_test_depthE])

print("mseTrainDepthF:",*["%.8f"%(x) for x in mse_train_depthF])
print("mseTestDepthF:",*["%.8f"%(x) for x in mse_test_depthF])


# Width
layersAA = [1, 16, 16, 16, 5]
layersBB = [1, 32, 32, 32, 5]
layersCC = [1, 64, 64, 64, 5]
layersDD = [1, 128, 128, 128, 5]

model1AA = PINN_constantParam(T_train, I_train1, R_train1, layersAA, lb, ub)
model2AA = PINN_constantParam(T_test, I_test1, R_test1, layersAA, lb, ub)
model1BB = PINN_constantParam(T_train, I_train1, R_train1, layersBB, lb, ub)
model2BB = PINN_constantParam(T_test, I_test1, R_test1, layersBB, lb, ub)
model1CC = PINN_constantParam(T_train, I_train1, R_train1, layersCC, lb, ub)
model2CC = PINN_constantParam(T_test, I_test1, R_test1, layersCC, lb, ub)
model1DD = PINN_constantParam(T_train, I_train1, R_train1, layersDD, lb, ub)
model2DD = PINN_constantParam(T_test, I_test1, R_test1, layersDD, lb, ub)

model1AA.train(niter2); model2AA.train(niter2); model1BB.train(niter2); model2BB.train(niter2)
model1CC.train(niter2); model2CC.train(niter2); model1DD.train(niter2); model2DD.train(niter2)

mse_train_widthA = model1AA.loss_log; mse_test_widthA = model2AA.loss_log
mse_train_widthB = model1BB.loss_log; mse_test_widthB = model2BB.loss_log
mse_train_widthC = model1CC.loss_log; mse_test_widthC = model2CC.loss_log
mse_train_widthD = model1DD.loss_log; mse_test_widthD = model2DD.loss_log

print("mseTrainWidthA:",*["%.8f"%(x) for x in mse_train_widthA])
print("mseTestWidthA:",*["%.8f"%(x) for x in mse_test_widthA])

print("mseTrainWidthB:",*["%.8f"%(x) for x in mse_train_widthB])
print("mseTestWidthB:",*["%.8f"%(x) for x in mse_test_widthB])

print("mseTrainWidthC:",*["%.8f"%(x) for x in mse_train_widthC])
print("mseTestWidthC:",*["%.8f"%(x) for x in mse_test_widthC])

print("mseTrainWidthD:",*["%.8f"%(x) for x in mse_train_widthD])
print("mseTestWidthD:",*["%.8f"%(x) for x in mse_test_widthD])



##########################################################################################################


