## ----------------------------------------------- ##
## app.py ---------------------------------------- ##
## Applying the CBBMC method on the NCSU stats --- ##
## graduate student headshots -------------------- ##
## Authors: Caleb Weaver, Peter Norwood ---------- ##
## cjweave2@ncsu.edu, pnorwoo@ncsu.edu ----------- ## 
## North Carolina State Univeristy --------------- ##
## Instructor: Dr. Wenbin Lu --------------------- ##
## ----------------------------------------------- ##

### import functions
## loading proper packages
import numpy as np
import numpy.linalg as la
import numpy.matrix as mat
import random
import pandas as pd
from PIL import Image 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer
import os
## set the working directory
os.chdir("PATH")


## different levels of masking
masks =[0.2,0.5,0.8]

## Loading the image data
# Create Testing Data
## total sample size
N = 157
ims = []
ppl = []
for i in range(N):
    s = "PATH" + str(i) + ".jpg"
    im = Image.open(s)
    im = gray(np.array(im))
    ims.append(im)



## alex:  1
## charlie: 17
## ethan: 33
    
## creating training and testing sets
test = [ims[i] for i in [1,17,33]]
train = ims
del train[1],train[17],train[33]

train = np.array(train)   
k, m, n = train.shape
Xall = np.zeros((k, m*n))
for k in range(train.shape[0]):
    Xall[k,:] = train[k,:,:].reshape(1,m*n) 
        
## create the 8 groups
bestk = 8
mod = KMeans(n_clusters=bestk,random_state=10)
mod.fit(Xall)
n_components=10
pca=PCA(n_components=n_components, whiten=True)

## looping through the individuals
for i in range(len(test)):

    pic = np.zeros_like(test[i])
    pic[:] = test[i]
    str1 = "raw_pic_i_"+str(i)+".csv"
    np.savetxt(str1,pic,delimiter=",")
    
    i_ind = i
    
    ## looping through the different mask levels
    for j in masks:
        
        str2 = "gm_pic_i_"+str(i_ind)+"_j_"+str(j)+".csv"
        str3 = "cbbmc_pic_i_"+str(i_ind)+"_j_"+str(j)+".csv"
        str4 = "maskNA_pic_i_"+str(i_ind)+"_j_"+str(j)+".csv"
        str5 = "mask_pic_i_"+str(i_ind)+"_j_"+str(j)+".csv"
        str6 = "mask_i_"+str(i_ind)+"_j_"+str(j)+".csv"
        
        ## damage the image
        mask = get_mask(j,500,500)
        masked_pic = apply_mask(pic.reshape(500,500), mask)
        masked_picNA = apply_mask_na(pic.reshape(500,500), mask)
        
        ## classify the damaged image to a group
        pic_imputed = np.zeros_like(masked_pic)
        imp = Imputer()
        impfit = imp.fit(pic_imputed.reshape(500,500))
        temp = impfit.transform(masked_picNA.reshape(500,500)).reshape(1,500*500)
        group = mod.predict(temp)[0]
        
        ## grabbing the prior image based on the fit
        groupidx = [True if x==group else False for x in mod.labels_]
        Xgroup = Xall[groupidx,]
        pca.fit(Xgroup)
        covarr = []
        for r in range(500):
            X = np.zeros((Xgroup.shape[0], 500))
            for i in range(Xgroup.shape[0]):
                im = Xgroup[i,:].reshape(500,500)
                X[i,] = im[r,:]
            y = pd.DataFrame(X).cov()
            covarr.append(y)

        prior = pca.mean_.reshape((500,500))
        
        ## grey scaling the prior image
        prior = 255 * ((prior - np.min(prior)) / (np.max(prior)))
        
        ## re-create image using the gm method
        gm = fit(masked_picNA,mask,20)
        gm_pic = gm[0]

        ## developing the posterior distribution to update the 
        ## masked image with
        update = np.empty((500,500))
        sig1 = np.array(pd.DataFrame(gm_pic).cov())
        for k in range(500):
            sig0e = covarr[k]/np.max(covarr[k])
            np.fill_diagonal(sig0e.values,np.diag(covarr[k]))
            sig1e = sig1/np.max(sig1)
            np.fill_diagonal(sig1e,np.diag(sig1))
            pst = posterior(prior[k,:],sig0e,sig1e,gm_pic[k,:])
            pstmu = pst[0]
            update[k,] = pstmu
        
        ## scaling the updated matrix
        update = 255 * ((update - np.min(update)) / (np.max(update)))
        
        ## cbbmc update
        new = np.zeros_like(masked_pic)
        for p in range(masked_pic.shape[0]):
            for q in range(masked_pic.shape[0]):
                if mask[p,q]==1:
                    new[p,q] = masked_pic[p,q]
                else:
                    new[p,q] = update[p,q]
        
        h, w = new.shape
        cbbmc_pic =  np.array(new)

        ## smoothing out the cbbmc image
        for y in range(0, w-2):
            for x in range(0, h-2):     
                    px1 = new[x][y] #0/0
                    px2 = new[x][y+1] #0/1
                    px3 = new[x][y+2] #0/2
                    px4 = new[x+1][y] #1/0
                    px5 = new[x+1][y+1] #1/1
                    px6 = new[x+1][y+2] #1/2
                    px7 = new[x+2][y] #2/0
                    px8 = new[x+2][y+1] #2/1
                    px9 = new[x+2][y+2] #2/2
                    average = px1/9. + px2/9. + px3/9. + px4/9. + px5/9. + px6/9. + px7/9. + px8/9. + px9/9.
                    cbbmc_pic[x+1][y+1] = average  
                    
        ## scaling the pictures
        gm_pic = 255 * ((gm_pic - np.min(gm_pic)) / (np.max(gm_pic)))
        cbbmc_pic = 255 * ((cbbmc_pic - np.min(cbbmc_pic)) / (np.max(cbbmc_pic)))
        
        ## plotting the images
        #plot_img(masked_picNA)
        #plot_img(gm_pic)
        #plot_img(cbbmc_pic)
        #plt.pause(0.1)
        
        ## saving the images
        np.savetxt(str2,gm_pic,delimiter=",")
        np.savetxt(str3,cbbmc_pic,delimiter=",") 
        np.savetxt(str4,masked_picNA,delimiter=",")
        np.savetxt(str5,masked_pic,delimiter=",")
        np.savetxt(str6,mask,delimiter=",")
        
        
        