## ----------------------------------------------- ##
## funcs.py -------------------------------------- ##
## Functions for ST 790 - Machine Learning Project ##
## Authors: Caleb Weaver, Peter Norwood ---------- ##
## cjweave2@ncsu.edu, pnorwoo@ncsu.edu ----------- ## 
## North Carolina State Univeristy --------------- ##
## Instructor: Dr. Wenbin Lu --------------------- ##
## ----------------------------------------------- ##


### all functions for the project

## loading proper packages
import numpy as np
import numpy.linalg as la
from numpy import matrix
import random
import pandas as pd
from PIL import Image 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

### Grayscale
def gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

### Plot Image
def plot_img(x):
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(x, cmap='gray')

### Get mask matrix
def get_mask(rate,m,n):
    mask = np.zeros((m,n))
    mask[:] = 1
    grid = [(a,b) for a in range(m) for b in range(n)]
    s = int((1-rate) * len(grid))
    idx = random.sample(range(m*n), s)
    points = [grid[i] for i in idx]
    for p in points:
        mask[p] = 0
    return(mask)

### Apply Mask
def apply_mask(x,mask):
    masked = np.zeros_like(x)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == 0:
                masked[i,j] = 0
            else:
                masked[i,j] = x[i,j]
    return(masked)
    
## Apply mask with missing values instead of zeros
def apply_mask_na(x,mask):
    masked = np.zeros_like(x)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == 0:
                masked[i,j] = np.nan
            else:
                masked[i,j] = x[i,j]
    return(masked)
           
### Get parameter estimates
def mle(x):
    n = x.shape[1]
    mu = np.nanmean(x,0).T
    sig = np.array((pd.DataFrame(x)).cov())
    return([mu,sig])

### Get MAP Estimates
def mapest(x, mask, mu, sig):
    recov = np.zeros_like(x)
    num_patches = x.shape[0]
    
    bigmu = pd.concat([pd.DataFrame(mu).T]*num_patches, ignore_index=True)
    resid = bigmu - x
    sigw = np.array(pd.DataFrame(resid).cov())
    
    for k in range(num_patches):
        patch1 = x[k,]
        mask1 = mask[k,]
        maskbool = [False if i==1 else True for i in mask1]
        notmaskbool = [not i for i in maskbool]
        A = sig[notmaskbool,:]
        A = A[:,notmaskbool]
        
        patch1_rec = np.zeros_like(patch1);
        patch1_rec[:] = patch1
        
        B = sig[maskbool,:]
        B = B[:,notmaskbool]
        
        C = sigw[notmaskbool,:]
        C = C[:,notmaskbool]
        
        tmp = mu[maskbool] + B @ la.inv(A + C) @ (patch1[notmaskbool] - mu[notmaskbool])
        patch1_rec[maskbool] = tmp;
        recov[k,:] = patch1_rec;
            
    return(recov)

### Model
def fit(masked,mask,max_iter,tolerance=.0001):
    diff = []
    recov = np.zeros_like(masked)
    recov[:] = masked
    for i in range(max_iter):
        print(i)
        #plot_img(recov)
        old = np.zeros_like(recov)
        old[:] = recov
        # M Step
        mu, sig = mle(recov)
        # E Step                                
        recov = np.array(mapest(recov,
                                mask,
                                mu,
                                sig));
        recov = 255 * ((recov - np.min(recov)) / (np.max(recov)))
        diffk = np.nanmean(abs(recov-old))
        diff.append(diffk)
        if(normf(recov,old)<tolerance):
            break
    return([recov,diff])



### Posterior Function
## mu0 -- prior row
## sigma0 -- prior cov, for that row
## sigma1 -- cov for masked row
## y -- masked row   
def posterior(mu0,sig0,sig1,y):
  n = 500
  xbar = y
  sig0i = la.inv(sig0)
  sig1i = la.inv(sig1)
  postmu = la.inv(sig0i + n*sig1i) @ (sig0i @ mu0 + n*sig1i @ xbar)
  postsig = la.inv(sig0i + n*sig1i)
  return([postmu,postsig])
  
   
