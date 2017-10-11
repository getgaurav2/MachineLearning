# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


from numpy import genfromtxt
#############################################################################
df_X_train = genfromtxt('X_train.csv', delimiter=',')
df_Y_train = genfromtxt('Y_train.csv', delimiter=',')
df_X_test = genfromtxt('X_test.csv', delimiter=',')
df_Y_test = genfromtxt('Y_test.csv', delimiter=',')

#############################################################################
def WRR(X_train , Y_train , Lam_range, feature_count):
    
    I = np.identity(feature_count)
    
    WRR= [0 for row in range(Lam_range)]
    dof = []
    X_train_trans = X_train.transpose()
    
    for  Lam in range(0, Lam_range):
      
      Lam_Identity = np.dot(Lam , I)
      
      Inverse =   inv( Lam_Identity + np.dot(X_train_trans ,X_train))
      part1 = np.dot(Inverse ,X_train_trans)
      part2 = np.dot(part1 , X_train)

      WRR[Lam] = np.dot(part1 , df_Y_train)
      
      dof.append ( np.trace( part2))

    return np.asarray(WRR) , np.asarray(dof)
    
#############################################################################
W_RR , DOF= WRR (df_X_train ,df_Y_train ,5000, 7)
print W_RR[1]

plt.figure(1)
label = [ 'cylinders' , 'displacement' ,
        'horsepower'  ,'weight' , 'acceleration' ,'car yr' ,'W0']
for  i in range(0,7 ):
    plt.plot(DOF ,W_RR[:,i],label = label[i]) 


plt.xlabel("Degrees of Freedom")
plt.ylabel("WRR - Coefficient")
plt.legend()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


#### ADD LEGEND 
#############################################################################
def predict( WRR , X_test , Lam_range):
    
    Y_predict= [0 for row in range(Lam_range)]

    for  i in range(0, Lam_range):
        Y_predict[i] = np.dot(WRR[i], X_test.transpose())

    return Y_predict

#############################################################################

## Root Mean  Square Error
def RMSE (Y_predict ,Y_test ,Lam_range, test_rec_count ):
 
 RMSE= [0 for row in range(Lam_range)] 
 

 for  i in range(0, Lam_range):
     square_error  =  np.square(Y_predict[i] - Y_test)

     RMSE[i] =  np.sqrt((np.sum(square_error)/test_rec_count))
 
 return RMSE

#############################################################################
Y_predict = predict (W_RR ,df_X_test , 50 )

RMSE_X1 =  RMSE(Y_predict , df_Y_test ,50, 42)
plt.figure(2)


plt.plot(range(50), np.asarray(RMSE_X1)) 

plt.xlabel("Lambda")
plt.ylabel("RMSE")
plt.legend()
plt.show()

#############################################################################
# add 2 and 3 order terms in the input matrix 
def get_pth_order ( X  ):

    X_1 = X
    X_2 = []
    X_3 = []
    
    for i in range (len(X_1)):
        X_2.append([])
        X_3.append([])
        for j in range(len(X_1[0])):
            X_2[i].append((X_1[i][j]))
            X_3[i].append((X_1[i][j]))
            
        for j in range(len(X_1[0])-1):
            X_2[i].append((X_1[i][j])**2)
            X_3[i].append((X_1[i][j])**2)

        for j in range(len(X_1[0])-1):
            X_3[i].append((X_1[i][j])**3)


    return X_1 , np.asarray(X_2) , np.asarray(X_3)


X1_train , X2_train , X3_train = get_pth_order (df_X_train )
X1_test , X2_test , X3_test = get_pth_order (df_X_test )

#############################################################################

W_RR_X_1 , DOF= WRR (X1_train ,df_Y_train ,500, 7)
Y_predict_X_1 = predict (W_RR_X_1 ,X1_test , 500 )
RMSE_X1 =  RMSE(Y_predict_X_1 , df_Y_test ,500, 42)

W_RR_X_2 , DOF= WRR (X2_train ,df_Y_train ,500, 13)
Y_predict_X_2 = predict (W_RR_X_2 ,X2_test , 500 )
RMSE_X2 =  RMSE(Y_predict_X_2 , df_Y_test ,500, 42)


W_RR_X_3 , DOF= WRR (X3_train ,df_Y_train ,500, 19)
Y_predict_X_3 = predict (W_RR_X_3 ,X3_test , 500 )
RMSE_X3 =  RMSE(Y_predict_X_3 , df_Y_test ,500, 42)

plt.figure(3)

plt.plot(range(500), np.asarray(RMSE_X1) ,label = "X1") 
plt.plot(range(500), np.asarray(RMSE_X2), label = "X2")  
plt.plot(range(500), np.asarray(RMSE_X3), label = "X3") 

plt.xlabel("Lambda")
plt.ylabel("RMSE")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
           
plt.show() 

#############################################################################



 