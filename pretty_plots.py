## ----------------------------------------------- ##
## pretty_plots.py ------------------------------- ##
## pretty looking plots for our machine learning - ##
## project --------------------------------------- ##
## Authors: Caleb Weaver, Peter Norwood ---------- ##
## cjweave2@ncsu.edu, pnorwoo@ncsu.edu ----------- ## 
## North Carolina State Univeristy --------------- ##
## Instructor: Dr. Wenbin Lu --------------------- ##
## ----------------------------------------------- ##

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


## useful functions
### Plot Image
def plot_img(x):
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(x, cmap='gray')

## alex 0.8

alex_og = pd.read_csv("C:/Users/peter/OneDrive/Documents/Classes/ST 790 - Fall 2019/Matrix Completion/raw_pic_i_0.csv")
alex_mask = pd.read_csv("C:/Users/peter/OneDrive/Documents/Classes/ST 790 - Fall 2019/Matrix Completion/maskNA_pic_i_0_j_0.2.csv")
alex_gm = pd.read_csv("C:/Users/peter/OneDrive/Documents/Classes/ST 790 - Fall 2019/Matrix Completion/gm_pic_i_0_j_0.2.csv")
alex_cbbmc = pd.read_csv("C:/Users/peter/OneDrive/Documents/Classes/ST 790 - Fall 2019/Matrix Completion/cbbmc_pic_i_0_j_0.2.csv")

alex_plots = [alex_og,
              alex_mask,
              alex_gm,
              alex_cbbmc]

## one row
fig, ax =plt.subplots(nrows=1, ncols=4,figsize = [24,12] )
names = ["Original","Damaged","GM Method","CBBMC Method"]
for i in range(0,4):
    ax[i].imshow(alex_plots[i], cmap="gray")
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title(names[i])
    
## square   
fig, ax =plt.subplots(nrows=2, ncols=2,figsize = [12,12] )
names = ["Original","Damaged","GM Method","CBBMC Method"]
## i should really just think and write a for loop, but it's late
ax[0,0].imshow(alex_plots[0], cmap="gray")
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
ax[0,0].set_title(names[0]) 
    
ax[0,1].imshow(alex_plots[1], cmap="gray")
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[0,1].set_title(names[1])   

ax[1,0].imshow(alex_plots[2], cmap="gray")
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])
ax[1,0].set_title(names[2]) 
    
ax[1,1].imshow(alex_plots[3], cmap="gray")
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])
ax[1,1].set_title(names[3])    
 


## charlie 0.5
    
charlie_og = pd.read_csv("C:/Users/peter/OneDrive/Documents/Classes/ST 790 - Fall 2019/Matrix Completion/raw_pic_i_1.csv")
charlie_mask = pd.read_csv("C:/Users/peter/OneDrive/Documents/Classes/ST 790 - Fall 2019/Matrix Completion/maskNA_pic_i_1_j_0.5.csv")
charlie_gm = pd.read_csv("C:/Users/peter/OneDrive/Documents/Classes/ST 790 - Fall 2019/Matrix Completion/gm_pic_i_1_j_0.5.csv")
charlie_cbbmc = pd.read_csv("C:/Users/peter/OneDrive/Documents/Classes/ST 790 - Fall 2019/Matrix Completion/cbbmc_pic_i_1_j_0.5.csv")

charlie_plots = [charlie_og,
                 charlie_mask,
                 charlie_gm,
                 charlie_cbbmc]

## one row
fig, ax =plt.subplots(nrows=1, ncols=4,figsize = [24,12] )
names = ["Original","Damaged","GM Method","CBBMC Method"]
for i in range(0,4):
    ax[i].imshow(charlie_plots[i], cmap="gray")
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title(names[i])
   
    
## square   
fig, ax =plt.subplots(nrows=2, ncols=2,figsize = [12,12] )
names = ["Original","Damaged","GM Method","CBBMC Method"]
## i should really just think and write a for loop, but it's late
ax[0,0].imshow(charlie_plots[0], cmap="gray")
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
ax[0,0].set_title(names[0]) 
    
ax[0,1].imshow(charlie_plots[1], cmap="gray")
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[0,1].set_title(names[1])   

ax[1,0].imshow(charlie_plots[2], cmap="gray")
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])
ax[1,0].set_title(names[2]) 
    
ax[1,1].imshow(charlie_plots[3], cmap="gray")
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])
ax[1,1].set_title(names[3])    
  
    
## ethan 0.2
    
    
ethan_og = pd.read_csv("C:/Users/peter/OneDrive/Documents/Classes/ST 790 - Fall 2019/Matrix Completion/raw_pic_i_2.csv")
ethan_mask = pd.read_csv("C:/Users/peter/OneDrive/Documents/Classes/ST 790 - Fall 2019/Matrix Completion/maskNA_pic_i_2_j_0.8.csv")
ethan_gm = pd.read_csv("C:/Users/peter/OneDrive/Documents/Classes/ST 790 - Fall 2019/Matrix Completion/gm_pic_i_2_j_0.8.csv")
ethan_cbbmc = pd.read_csv("C:/Users/peter/OneDrive/Documents/Classes/ST 790 - Fall 2019/Matrix Completion/cbbmc_pic_i_2_j_0.8.csv")

ethan_plots = [ethan_og,
               ethan_mask,
               ethan_gm,
               ethan_cbbmc]

## one row
fig, ax =plt.subplots(nrows=1, ncols=4,figsize = [24,12] )
names = ["Original","Damaged","GM Method","CBBMC Method"]
for i in range(0,4):
    ax[i].imshow(ethan_plots[i], cmap="gray")
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title(names[i])    
    
  ## square   
fig, ax =plt.subplots(nrows=2, ncols=2,figsize = [12,12] )
names = ["Original","Damaged","GM Method","CBBMC Method"]
## i should really just think and write a for loop, but it's late
ax[0,0].imshow(ethan_plots[0], cmap="gray")
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
ax[0,0].set_title(names[0]) 
    
ax[0,1].imshow(ethan_plots[1], cmap="gray")
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[0,1].set_title(names[1])   

ax[1,0].imshow(ethan_plots[2], cmap="gray")
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])
ax[1,0].set_title(names[2]) 
    
ax[1,1].imshow(ethan_plots[3], cmap="gray")
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])
ax[1,1].set_title(names[3])    
   
    
    