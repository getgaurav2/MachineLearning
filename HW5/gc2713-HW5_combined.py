import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp_linalg


from numpy import genfromtxt


Scores = genfromtxt('CFB2016_scores.csv', delimiter=',')
Teams = np.loadtxt('teamNames.txt',delimiter='~' , dtype = 'string')


def init_RandomWalkMatrix (Scores , Teams):
    M_hat = np.zeros((len(Teams),len(Teams)))
    for s in Scores:
        idx_A =  int(s[0] - 1)
        idx_B =  int( s[2] - 1 )
        
        if ( s[1] >= s[3]):# Team A wins
            Team_A = 1.0
            Team_B = 0.0 
        if ( s[1] < s[3]): # team B wins
            Team_A = 0.0 
            Team_B = 1.0           
        
            M_hat[idx_A,idx_A] = M_hat[idx_A,idx_A] + Team_A +  s[1]/(s[1]+s[3])
            
            M_hat[idx_B,idx_B] = M_hat[idx_B,idx_B] + Team_B +  s[3]/(s[1]+s[3])
            
            M_hat[idx_B,idx_A] = M_hat[idx_B,idx_A] + Team_A +  s[1]/(s[1]+s[3])
            
            M_hat[idx_A,idx_B] = M_hat[idx_A,idx_B] + Team_B +  s[3]/(s[1]+s[3]) 
            
    return  M_hat



def Norm ( M_hat ): 
    row_sums = M_hat.sum(axis=1)
    M = M_hat / row_sums[:, np.newaxis]
    return M     

def markovChain(M,T):
    wt=[]
    error_all = []
    w0=np.full((760), 1.00/760.00,dtype=np.float64)
    wt.append(w0)
    Eig = (sp_linalg.eigs(M.T , k=1,sigma=1.0)[1]).real
    w_inf = Eig.T/sum(Eig)
    print sum(w_inf)
    for t in range(1,T):        
        wt.append(wt[t - 1].dot(M))
        error_all.append (sum(abs(wt[t-1]  - w_inf)))
    return np.argsort(wt[t])[-25:]  , error_all , w_inf , wt , Eig

M_hat = init_RandomWalkMatrix (Scores , Teams)    
M_gaurav= Norm (M_hat)
W = np.full_like(M_gaurav ,float(1.0/(len(Teams))))


T = [ 10, 100  , 1000  ,  10000]
  
for t in T: 
    W , error_all , w_inf , wt  , Eig= markovChain(M_gaurav,t)
    print "*****" , t , "*****"
    for i,team in enumerate(reversed(W)):
       
       print(i+1,team+1,  Teams[team] )  # ,games[team],wins[team],pts[team])        
       plt.figure(t)
       plt.plot(range(t-1), error_all)
       plt.xlabel("Iteration")
       plt.ylabel("||  Wt - W_inf  ||")
       plt.title("Markov Chain - Team Ranking")
       
       
###############################################################################################
# question  2) 

import pandas as pd


from tabulate import tabulate


nyt_data = np.loadtxt('nyt_data.txt' , dtype = 'string')

def getDocData():
    words = pd.read_table('./nyt_vocab.dat',header=None)
    words.columns = ["Word"]
    fileData = "./nyt_data.txt"

    lines = []

    with open(fileData) as f:
        for line in f:
            lines.append(line)
    X = np.zeros((words.shape[0],len(lines)),dtype=int)
    for doc,line in enumerate(lines):
        for word_doc_ct in line.split(","):
            #print(doc,word_doc_ct)
            word_ct = word_doc_ct.split(":")
            word=int(word_ct[0])-1
            ct =int(word_ct[1])
            X[word,doc] = ct

    return X,words
               
def F_update(X, W, H):
    #d, n = X.shape
    WH = np.dot(W, H) + e
    F = - (X * np.log(WH) - WH).sum()
    return F

def Norm ( Mat ): 
    row_sums = Mat.sum(axis=1)
    M = Mat / row_sums[:, np.newaxis]
    return M  

def approx (X , W , H ):
     WH = np.dot(W, H) + e 
     #print WH.shape
     #print "**apprx**"
     #print np.true_divide(X , WH)
     return np.true_divide(X , WH)



def update_H (X ,H , W  ):
    WH = np.dot(W, H) + e
    H_new = H  * np.dot(W.T , X/ WH)
    W_col_sums = W.sum(axis=0)
    H_new = H_new / W_col_sums[:, np.newaxis]
    #print "*** H *** "
    #print H_new
    
    return H_new
    

def update_W (X ,H , W ):
    WH = np.dot(W, H) + e
    W_new = W * np.dot(X / WH , H.T)
    H_row_sums = H.sum(axis=1)
    W_new = W_new / H_row_sums[np.newaxis, :]
    #print "*** W *** " 
    #print W_new 
    
    return W_new
    
Rank = 25 
X , words = getDocData ()
W = np.random.rand(X.shape[0],Rank )
H = np.random.rand(Rank,X.shape[1])
e =  .00000000000000001
All_F = []
for iter in range (0,100):
 #   print "****" , iter, "****"
    H = update_H (X ,H , W  )
    W = update_W(X ,H , W)
    All_F.append(F_update(X , W , H))

plt.plot(range(100) , All_F)
plt.xlabel("iterations")
plt.ylabel("Divergence")
plt.title("NMF Objective function")

col_sums = W.sum(axis=0)
W = W / col_sums[np.newaxis,:]

table =  [[0 for x in range(10)] for x in range(25)] 
headers = []
for t in range(25):
        print "***  topic # ***" , t
        headers.append("Topic-"+str(t+1))
        top_words_indices = np.argsort(W[:,t])[-10:]
        for i,index in enumerate(reversed(top_words_indices)):
            print "**index" , i ,  index
            table[t][i] = words.iloc[int(index)]["Word"] + " ~~" + str(round(W[int(index),t],4))+""

for t in range(1,6):
        print_table =  [[0 for x in range(5)] for x in range(10)] 
        for c in range(5):
            for r in range(10):
                #print(r,c,r,c + (t - 1) * 5)
                print_table[r][c] = table[c + (t - 1) * 5][r]

        print (tabulate(print_table, headers=headers[(t-1)*5:t*5],tablefmt='orgtbl'))
        print("")    
    
    

        
        
       