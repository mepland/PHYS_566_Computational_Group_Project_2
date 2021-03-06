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
   
#Create a function to check if an individual lattice point is an occupied neighbor to the point of interests.
#Returns true if the point is a neighbor and false if the point is not an occupied neighbor.
def checkNeighbor(x,y,N):
    neighbor=True
    if N[x,y]!=0:
        neighbor=True
    else:
        neighbor=False
    return neighbor
 
#Create a function to see if all occupied neighbors are the same.
#Returns true if all the occupied neighbors are the same and false if they are not.
def isIdentical(testList):
    for i in range(len(testList)):
        first=testList[0]
        if first!=testList[i]:
            return False
    return True   

#Create a function to take in different lattice sizes 
#This is the biggest function in the program
#N is the lattice size

N=5
L=np.zeros((N,N))

def findCluster(N):
    m=50                               #Number of Simulations
    random.seed()                      #Initialize Random Number Generator
    
    #Start Values-One Value for the x-direction (row number) another for the y-diretion(column number)
    S_x=random.randrange(0,N)
    S_y=random.randrange(0,N)

    #Create the lattice using the N dimension
    L=np.zeros((N,N))

    #Set The Initial Random Occupied Site
    L[S_x,S_y]=1

    #Keep track of the number of clusters
    clusterCount=1

    #Generate a New Random Number
    newx=random.randrange(0,N)
    newy=random.randrange(0,N)


    #Check if Each of the Neighbors is Inside the Lattice
    left=inBounds(newx-1,newy,N)
    right=inBounds(newx+1,newy,N)
    top=inBounds(newx,newy-1,N)
    bottom=inBounds(newx,newy+1,N)

    #Initialize these variables which determine whether the neighbors are occupied
    leftNeigh=False
    rightNeigh=False
    topNeigh=False
    bottomNeigh=False

    #Check if the neighbors are occupied
    if left==True:
        leftNeigh=checkNeighbor(newx-1,newy,L)
    if right==True:
        rightNeigh=checkNeighbor(newx+1,newy,L)
    if top==True:
        topNeigh=checkNeighbor(newx,newy-1,L)
    if bottom==True:
        bottomNeigh=checkNeighbor(newx,newy+1,L)

    #Case 1, No Neighbors are Occupied, Make a New Cluster
    if leftNeigh==False and rightNeigh==False and topNeigh==False and bottomNeigh==False:
        L[newx,newy]=clusterCount+1   #Make a new cluster
        clusterCount=clusterCount+1     #Increase the total number of clusters by 1
    
    #Case 2, One Common Neighbor, Add to an Existing Cluster
    #Check if each neighbor is occupied using if statements
    elif leftNeigh==True or rightNeigh==True or topNeigh==True or bottomNeigh==True:
        neighCount=0                 #The number of total occupied neighbors
        occupyNeigh=[0.0]*4           #An array with all the occupied neighbors
        if leftNeigh==True:
            occupyNeigh[neighCount]=L[newx-1,newy]
            neighCount=neighCount+1
        if rightNeigh==True:
            occupyNeigh[neighCount]=L[newx+1,newy]
            neighCount=neighCount+1
        if topNeigh==True:
            occupyNeigh[neighCount]=L[newx,newy-1]
            neighCount=neighCount+1
        if bottomNeigh==True:
            occupyNeigh[neighCount]=L[newx,newy-1]
            neighCount=neighCount+1
            
        occupyNeigh=occupyNeigh[0:neighCount]      #Resize the array to the number of occpuied neighbors
        sameNeigh=isIdentical(occupyNeigh)         #Use isIdentical to determine if all occupied neighbors are occupied by the same number
     
        #If the occupied neighbors are all the same, assign the current point the same value
        if sameNeigh==True:
            L[newx-1,newy]=occupyNeigh[0]
    
#Case 3, Multiple Common Neighbors, Merge Clusters


    return L             #The function findCluster returns the lattice with cluster 
#Find the cluster for different lattice sizes
N_5=findCluster(5)
