# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import scipy as scipy

from numpy import genfromtxt
from sklearn.metrics import confusion_matrix
#############################################################################
df_X_train = genfromtxt('X_train.csv', delimiter=',')
df_Y_train = genfromtxt('Y_train.csv', delimiter=',')
df_X_test = genfromtxt('X_test.csv', delimiter=',')
df_Y_test = genfromtxt('Y_test.csv', delimiter=',')


#############################################################################

#
#def class_prior ( Y_train):
#    
#    tot_records = len(Y_train)
#    tot_spam =  np.sum(Y_train == 1)
#    tot_not_spam =  np.sum(Y_train == 0)
#    p_spam = tot_spam/tot_records
#    p_not_spam = tot_not_spam/tot_records 
#    pi_hat = tot_spam/tot_records
#    return tot_records , tot_spam , tot_not_spam , p_spam , p_not_spam , pi_hat
#
##############################################################################
#
#def class_conditional_Bernoulli (X_train , Y_train  , tot_spam , tot_not_spam):
#    Tot_spam = [0 for col in range(54)]
#    Tot_not_spam = [0 for col in range(54)]
#    Theta_spam = [0 for col in range(54)]
#    Theta_not_spam = [0 for col in range(54)]
#    
#    for j in range(54):    
#        for i in range(len(X_train)):
#      
#
## Calculating Total spam per parameter per class       
#         if (Y_train[i] == 1.0):
#            Tot_spam[j] = Tot_spam[j] + X_train[i][j]
#         if (Y_train[i] == 0.0):
#            Tot_not_spam[j] = Tot_not_spam[j] + X_train[i][j]
#    
#        Theta_spam[j] = Tot_spam[j] /tot_spam
#
#        Theta_not_spam[j] = Tot_not_spam[j] /tot_not_spam
#    
#    return Theta_spam , Theta_not_spam 
#
##############################################################################
#
#def class_conditional_Pareto(X_train , Y_train  , tot_spam , tot_not_spam):
#    Tot_spam = [0 for col in range(0, 3)]
#    Tot_not_spam = [0 for col in range(0, 3)]
#    Theta_spam = [0 for col in range(0, 3)]
#    Theta_not_spam = [0 for col in range(0, 3)]
#    
#    for j in range(0, 3):    
#        for i in range(len(X_train)):
#         if (Y_train[i] == 1.0):
#            Tot_spam[j] = Tot_spam[j] + np.log(X_train[i][54+j])
#         if (Y_train[i] == 0.0):
#            Tot_not_spam[j] = Tot_not_spam[j] + np.log(X_train[i][54+j])   
#        Theta_spam[j] =  tot_spam/Tot_spam[j]
#        Theta_not_spam[j] =  tot_not_spam/Tot_not_spam[j]   
#    return Theta_spam , Theta_not_spam 
#
#
#
# #############################################################################   
#tot_records , tot_spam , tot_not_spam , p_spam , p_not_spam , Pi_Hat =  class_prior (df_Y_train)   
#
#Theta_spam_Ber, Theta_not_spam_Ber =  class_conditional_Bernoulli (df_X_train , df_Y_train , tot_spam , tot_not_spam)
#
#
#Theta_spam_preto, Theta_not_spam_preto =  class_conditional_Pareto (df_X_train , df_Y_train , tot_spam , tot_not_spam)
#
##############################################################################
#
#   
#def posterior (Theta_spam_Ber , Theta_not_spam_Ber , Theta_spam_preto , Theta_not_spam_preto ,X_test):
#  
#      Bern_spam_1st_term =   np.power( Theta_spam_Ber ,  X_test[: , range(54)])  
#      
#      a =  [1-x for x in  X_test[: , range(54)]]
#     
#      b_spam =  [1-x for x in  Theta_spam_Ber]
#       
#      Bern_spam_2nd_term = np.power(b_spam, a)
#      
#      Predict_spam_Bern = np.multiply(Bern_spam_1st_term , Bern_spam_2nd_term)
#      
#      
#      Bern_not_spam_1st_term =   np.power( Theta_not_spam_Ber ,  X_test[: , range(54)])  
#      a =  [1-x for x in  X_test[: , range(54)]]
#      b_not_spam =  [1-x for x in  Theta_not_spam_Ber]
#      Bern_not_spam_2nd_term = np.power(b_not_spam, a)
#      Predict_not_spam_Bern = np.multiply(Bern_not_spam_1st_term , Bern_not_spam_2nd_term)
#      
#      
#      Pareto_spam_1st_term = [-1*(1+x) for x in  Theta_spam_preto]
#      Pareto_spam_2nd_term = np.power(   X_test[: , range(54,57)] , Pareto_spam_1st_term)
#      Predict_spam_pareto = np.multiply(Theta_spam_preto , Pareto_spam_2nd_term)
#             
#      
#      Pareto_not_spam_1st_term = [-1*(1+x) for x in  Theta_not_spam_preto]
#      Pareto_not_spam_2nd_term = np.power(   X_test[: , range(54,57)] , Pareto_not_spam_1st_term)
#      Predict_not_spam_pareto = np.multiply(Theta_not_spam_preto , Pareto_not_spam_2nd_term)
#         
#     
#      
#      return  Predict_spam_Bern     , Predict_not_spam_Bern , Predict_spam_pareto , Predict_not_spam_pareto
#        
##############################################################################         
#
#
#
#def predict (Predict_spam_Bern , Predict_not_spam_Bern , Predict_spam_pareto , Predict_not_spam_pareto , p_spam , p_not_spam , Pi_Hat):
#    
#    
#    Y_predict = []
#    
#    for i in range(93):
#        # initialize to class prior probabilities
#        prob_spam =  Pi_Hat
#        prob_not_spam = 1 - Pi_Hat
#        for j1 in range (54):
#            prob_spam  = prob_spam * Predict_spam_Bern[i,j1]
#            
#        for k1 in range(3):
#            prob_spam = prob_spam * Predict_spam_pareto[i,k1]
#    
#        for j2 in range (54):
#            prob_not_spam  = prob_not_spam * Predict_not_spam_Bern[i,j2]
#        for k2 in range(3):
#            prob_not_spam = prob_not_spam * Predict_not_spam_pareto[i,k2]    
#        
#        if (prob_spam > prob_not_spam):
#            Y_predict.append(1)
#        else:
#            Y_predict.append(0)
#
#    return Y_predict
#
#
#Predict_spam_Bern , Predict_not_spam_Bern , Predict_spam_pareto , Predict_not_spam_pareto = posterior ( Theta_spam_Ber , Theta_not_spam_Ber , Theta_spam_preto , Theta_not_spam_preto ,df_X_test)
#
#Y_predict = predict (Predict_spam_Bern , Predict_not_spam_Bern , Predict_spam_pareto , Predict_not_spam_pareto , p_spam , p_not_spam, Pi_Hat)
#
### get the confusion Matrix .
#c_mat = confusion_matrix(df_Y_test , Y_predict)
#print "Confusion Matrix: \n" , c_mat
#print "Naive Bayes Accuracy :" , (c_mat.trace()/93.0 )*100 , "%"
#
#
####################################################################################
###  K-NN
#
### rows has  number of test data point 
### columns has number of training data points
#def Diff_Matrix ( X_Train , X_test ):
#    #Diff = np.zeros((len(X_test), len(X_Train))) 
#    Diff_list = [[] for j   in range(len(X_test))]    
#    
#    for i in range(len(X_test)):
#        for j in range(len(X_Train)):
#        ## calcuate L1 distance.
#           Diff = sum(abs(X_test[i] - X_Train[j])) 
#           Diff_list[i].append((Diff , j) )
#    return Diff_list
#
#
#def KNN ( Diff_Matrix , X_train,Y_train,Y_test, K):
### we need the distance of the test vector  from each of the training vectors.
### hence we are interested in the Row  of the  Diff Matrix for each Test vector
#    accuracy_list = []
#    for k in range(1,K+1):
#      
#      Y_predict = []
#      for i in range(len(Diff_Matrix)):
#          Nearest = []          
#          Nearest = sorted(Diff_Matrix[i],key=lambda x: x[0])[0:k]
#
#          NN_values = []                   
#          for n in Nearest: 
#
#              index = n[1] 
#
#              NN_values.append(Y_train[index])             
#          if (np.average(NN_values) > 0.5):
#              Y_predict.append(1)
#          if (np.average(NN_values) < 0.5):
#              Y_predict.append(0)  
#          if (np.average(NN_values) == 0.5):
#              Y_predict.append(randint(0,1))
# 
#      c_mat = confusion_matrix(list(Y_test) , list(Y_predict))
#      print "KNN Confusion Matrix: \n" , c_mat
#      accuracy = c_mat.trace()/len(Y_predict)
#      print k , "KNN Accuracy :" , accuracy*100 , "%"
#      accuracy_list.append(accuracy)
#    return accuracy_list        
#              
#Diff_matrix =  Diff_Matrix(df_X_train , df_X_test)
#accuracy_list = KNN (Diff_matrix , df_X_train,df_Y_train,df_Y_test, 20)
#plt.figure(1)
#plt.plot(range(1,21) ,accuracy_list)    
##        
#####################################################################################        
##
##plt.stem(sum(Predict_not_spam_Bern[range(0,93),:]), markerfmt=" ")
#plt.figure(2)
#plt.stem(Theta_spam_Ber, markerfmt=" ")
#plt.figure(3)
#plt.stem(Theta_not_spam_Ber, markerfmt=" ")
#plt.show()

################################################################################### 
def preprocess ( X_train , X_test , Y_train , Y_test):
    Y_train_new = []
    Y_test_new = []   
    X_train_new = np.insert(X_train, 0, 1, axis=1)
    X_test_new = np.insert(X_test, 0, 1, axis=1)
    for y in Y_train:
        if (y == 1):
            Y_train_new.append(1)
        else:
            Y_train_new.append(-1)
    for y in Y_test:
        if (y == 1):
            Y_test_new.append(1)
        else:
            Y_test_new.append(-1)
    return np.matrix(Y_train_new) , np.matrix(Y_test_new), np.matrix(X_train_new).T, np.matrix(X_test_new).T
            
            
#def Sigmoid (X_train, N):

Y_train_new , Y_test_new , X_train_new , X_test_new = preprocess( df_X_train , df_X_test , df_Y_train , df_Y_test)  
################################################################################### 
def sigmoid( X , W):
   sig = (X.T * W )
   sigmoid =  scipy.special.expit(sig) 
   if (sigmoid[1] == 0 ):
       sigmoid[1] = .0000000000000000001
   return sigmoid 

W = np.matrix([0 for x in range(0, 58)]).T
sigmoid = sigmoid ( X_train_new , W)


################################################################################### 
