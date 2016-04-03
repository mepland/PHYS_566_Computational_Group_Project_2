################################Problem 1######################################
#################################Part A########################################

import numpy as np                        #Import Needed Libraries
import matplotlib.pyplot as plt
import random

#Create Function checkEndCondition to Determine if the While Loop Should be Stopped
#x is the cluster array
#N is the dimension of the cluster array
#If the Check is True the Cluster Has Reached All 4 Edges
#If the Check is Flase the Cluster Has Not Reached All 4 Edges
def checkEndCondition(x,N):  
    countLeft=0;
    countRight=0;
    countTop=0;
    countBottom=0;
    
    check=True

    for i in range(N):    
        if x[i,0]==1:
            countLeft=countLeft+1
        if x[i,N-1]==1:
            countRight=countRight+1
        if x[0,i]==1:
            countTop=countTop+1
        if x[N-1,0]==1:
            countBottom=countBottom+1

    if countLeft>=1 and countRight>=1 and countTop>=1 and countBottom>=1:
        check=True
    else:
        check=False        
    return check

#Create a function inbBounds to check if the neighboring index is in the lattice
#Returns true if the neighbor is inbounds and false if the neighbor is out of bounds
#When the index is negative or greater the lattice dimension the neighbor is out out of bounds
def inBounds(x,y,N):
    inCheck=True
    if x==-1 or y==-1 or x>N-1 or y>N-1:
        inCheck=False
    else:
        inCheck=True
    return inCheck
   
   
#Create a function to check if an individual lattice point is a neighbor to the point of interests.
#Returns true if the point is a neighbor and false if the point is not a neighbor.
def checkNeighbor(x,y,N):
    neighbor=True
    if N[x,y]!=0:
        neighbor=True
    else:
        neighbor=False
    return neighbor
    
    

#Set the Lattice Size
N_5=5

m=50                               #Number of Simulations
random.seed()                      #Initialize Random Number Generator


#Start Values-One Value for the x-direction another for the y-diretion
S_5x=random.randrange(0,N_5)
S_5y=random.randrange(0,N_5)

#Create Each Size Lattice
L_5=np.zeros((N_5,N_5))

#Set The Initial Random Occupied Site
L_5[S_5x,S_5y]=1

#Keep track of the number of clusters
clusterCount=1


#Generate a New Random Number
newx=random.randrange(0,N_5)
newy=random.randrange(0,N_5)


#Check if Each of the Neighbors is Inside the Lattice
left=inBounds(newx-1,newy,N_5)
right=inBounds(newx+1,newy,N_5)
top=inBounds(newx,newy-1,N_5)
bottom=inBounds(newx,newy+1,N_5)


#Initialize these variables which determine whether the neighbors are occupied
leftNeigh=False
rightNeigh=False
topNeigh=False
bottomNeigh=False


if left==True:
    leftNeigh=checkNeighbor(newx-1,newy,L_5)
if right==True:
    rightNeigh=checkNeighbor(newx+1,newy,L_5)
if top==True:
    topNeigh=checkNeighbor(newx,newy-1,L_5)
if bottom==True:
    bottomNeigh=checkNeighbor(newx,newy+1,L_5)

#Case 1, No Neighbors are Occupied, Make a New Cluster
if leftNeigh==False and rightNeigh==False and topNeigh==False and bottomNeigh==False:
    L_5[xnew,ynew]=clusterCount+1   #Make a new cluster
    clusterCount=clusterCount+1     #Increase the total number of clusters by 1
    
#Case 2, One Common Neighbor, Add to an Existing Cluster
    
#Case 3, Multiple Common Neighbors, Merge Clusters



    
    




    

