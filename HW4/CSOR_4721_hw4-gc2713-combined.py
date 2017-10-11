# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

    
##################3##################3##################3##################3
def genMixGaussian ( N ):
# generate data using 3 gaussians
    xy = []
        #y= [0 for row in range(N)]
        
    mean1 = [0, 0]
    cov1 = [[1, 0], [0, 1]]  
    
    mean2 = [3, 0]
    cov2 = [[1, 0], [0, 1]]  
    
    mean3 = [0, 3]
    cov3 = [[1, 0], [0, 1]]  
    
    x1, y1  =  np.random.multivariate_normal(mean1, cov1, int(0.2*500)).T
    x1y1 = np.vstack((x1,y1))
    x2, y2  =  np.random.multivariate_normal(mean2, cov2, int(0.5*500)).T
    x2y2 = np.vstack((x2,y2))
    x3, y3  =  np.random.multivariate_normal(mean3, cov3, int(0.3*500)).T
    x3y3 = np.vstack((x3,y3))    
    
    xy = np.vstack((x1y1.T ,x2y2.T  ,x3y3.T))
    return xy
    
##################3##################3##################3##################3

def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]



def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

##################3##################3##################3##################3
def objective(points, closest, centroids):
    
    tot_RMSE =0 
    for k in range( centroids.shape[0]):
        distances = []
        for  p in  points[closest==k]:
             dist = distance.euclidean(p,centroids[k,:])   
        
             distances.append(dist)
        RMSE =  (np.asanyarray(distances)).sum(axis = 0)
        tot_RMSE = tot_RMSE + RMSE
    return tot_RMSE
       

K = [2 ,3, 4,5]

XY = genMixGaussian(500)
centroids = initialize_centroids(XY , 3)




closest = closest_centroid(XY, centroids)
centroids = move_centroids(XY, closest, centroids)


C = ['r' , 'm' , 'b' , 'g' , 'c' , 'k']

for k in K:
   plt.figure(1)
   centroids = initialize_centroids(XY , k)
   obj = []
   for i in range(20):  
    closest = closest_centroid(XY, centroids)
    centroids = move_centroids(XY, closest, centroids)
    obj.append(objective(XY , closest , centroids))
#    print k , obj    
   plt.plot(range(20) , obj , label = ("K=",k ) , c=C[k])
   plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)
   if ((k == 3) or (k == 5)):
        plt.figure(k)
        for j in range(k):
            plt.scatter(XY[closest==j][:, 0], XY[closest==j][:, 1],marker='x' , c=C[j])
            plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
           # print closest


plt.show()




##################################################################
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

from scipy.spatial import distance
import heapq , itertools


from numpy import genfromtxt

ratings = genfromtxt('./COMS4721_hw4-data/ratings.csv', delimiter=',')
ratings_test = genfromtxt('./COMS4721_hw4-data/ratings_test.csv', delimiter=',')
movies = np.loadtxt('./COMS4721_hw4-data/movies.txt',delimiter='~' , dtype = 'string')


def prepareM ( ratings): 
    N1 =  np.max(ratings[:,0]) # users 
    N2 = np.max(ratings[:,1])  # movies
    M = np.empty([N1,N2]) 
    
    for rating in ratings:
        Mrow = rating[0] - 1
        Mcol = rating[1] - 1
        Mval = rating[2]
        M[Mrow, Mcol] = Mval
    return M , int(N1) , int(N2)

def getOmgU(M):
    N1 = 943
    omg_u = []
    omg_u.append([])
    for i in range(1, N1):
        omg_u.append(np.nonzero(M[i])[0])
    return omg_u

def getOmgV(M):
    N2 = 1682
    Mt = M.T
    omg_v = []
    omg_v.append([])
    for j in range(1, N2):
        omg_v.append(np.nonzero(Mt[j])[0])
    return omg_v



def updateUserLoc(v, M):

    lam_sig_I = np.identity(10)*sig_2

    u= np.zeros((944,10))

    for i in range(943):#range(1,944):
        v_j2 = np.zeros((10,10))
        M_vj = np.zeros((1,10))
        for j in omg_u[i]:
            i = int(i)
            j = int(j)
            v_j = np.matrix(v[j])
            v_j2 += v_j.T*v_j
            M_vj += M[i,j] * v_j


        lam_vj_inv = np.linalg.inv(lam_sig_I + v_j2)
        u_i = np.matmul(lam_vj_inv,M_vj.T)
        u[i] = u_i.T

    return u

def updateObjLoc(M, u):

    lam_sig_I = np.identity(10)*sig_2
    v = np.zeros((1682,10))
    for j in range(1682):#range(1,1683):

        u_i2 = np.zeros((10,10))
        M_ui = np.zeros((1,10))

        for i in omg_v[j]:
            i = int(i)
            j = int(j)
            u_i = np.matrix(u[i])
            u_i2 += u_i.T*u_i
            M_ui += M[i,j] * u_i

        lam_ui_inv = np.linalg.inv(lam_sig_I + u_i2)
        v_j = np.matmul(lam_ui_inv, M_ui.T)
        v[j] = v_j.T

    return v

def RMSE(ratings_test, pred):
    diff = 0
    for i in range(len(ratings_test)):
        diff += (ratings_test[i,2] - pred[int(ratings_test[i,0]),int(ratings_test[i,1])])**2
    return (diff/(len(ratings_test) * 1.0))**.5
    

def getObjVal(u,v,M, pred):

    p_m = 0
    p_u = 0
    p_v = 0

    for i in range(943): #range(1, 944):
        p_u += np.sum(np.abs(u[i])**2)
        for j in omg_u[i]:
            p_m += (M[i,j] - pred[i,j])**2
            p_v += np.sum(np.abs(v[j])**2)

    obj = -1 * (1/(2.0*sig_2)) * (p_m) - .5*p_u -.5*p_v
    return obj






d = 10
lam = 1
sig_2 = .25
M , N1 , N2= prepareM(ratings)
omg_u = getOmgU(M)
omg_v = getOmgV(M)                    
lam_sig_I = np.identity(d)*sig_2
    
v_all = []
u_all = []
#objval_all =np.zeros((2,100))
#rmse_all =np.zeros((2,100))

objval_all =np.zeros((10,100))
rmse_all =np.zeros((10,100))

label = [ 'step1' , 'step2' ,
        'step3'  ,'step4' , 'step5' ,'step6' ,'step7' , 'step8' , 'step9' , 'step10']
        
plt.figure(1)
#for step in xrange(100):
f = open("./COMS4721_hw4-data/q2-out.txt", 'wt')
writer = csv.writer(f)
for step in xrange(10):
    v = np.random.multivariate_normal(np.zeros(10),np.identity(10), N2)
    u = np.zeros((944,10))
    for i in xrange(N2):
        v[i] = np.random.multivariate_normal(np.zeros(10), np.identity(10))
     
#        obj = np.zeros(100)
#        rmse = np.zeros(100)

    for iter in range(100):
    #for iter in range(11):
     ## upate user and object location
        u = updateUserLoc(v,M)
        v = updateObjLoc(M,u)          
     
      ## get the objectve function value
        pred =np.matmul(u, v.T)
        objval_all[step, iter] = getObjVal(u,v,M,pred)
        rmse_all[step, iter] = RMSE(ratings_test, pred)
        #print step, iter, objval_all[step, iter] ,rmse_all[step, iter]
        writer.writerow( (step, iter, objval_all[step, iter] ,rmse_all[step, iter]) )
    #plt.plot(xrange(11), np.asarray(objval_all[step,:]) ,label = label[step])
    plt.plot(xrange(100), np.asarray(objval_all[step,:]) ,label = label[step])     
    v_all.append(v)
    u_all.append(u)
f.close()
#return    objval_all ,rmse_all


def closest_movie (p , Index_highest_obj):
    dist_all = []
    for v in v_all[Index_highest_obj]:
        dist = distance.euclidean(p,v)
        dist_all.append(dist)
    for tup in heapq.nsmallest(11, zip(dist_all, itertools.count())):
        print tup [0],tup[1] , movies[tup[1]]
    
Idx_highest_obj = np.argmax(objval_all[:,99])
#Idx_highest_obj = np.argmax(objval_all[:,10])

query_movies = ["GoodFellas (1990)" ,"Star Wars (1977)" , "My Fair Lady (1964)" ]
idx_goodfellas = list(movies).index("GoodFellas (1990)")
idx_star_wars = list(movies).index("Star Wars (1977)")
idx_my_fair_lady = list(movies).index("My Fair Lady (1964)") 

for movie in query_movies:
    idx = list(movies).index(movie)
    p=v_all[Idx_highest_obj][idx,:]
    print "****" , movie , "****"
    closest_movie(p , Idx_highest_obj)




 

 

