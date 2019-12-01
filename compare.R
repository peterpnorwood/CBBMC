## ----------------------------------------------- ##
## compare.R ------------------------------------- ##
## Comparing the cbbmc, gm, and other methods ---- ##
## on the NCSU grad student headshot dataset ----- ##
## Authors: Caleb Weaver, Peter Norwood ---------- ##
## cjweave2@ncsu.edu, pnorwoo@ncsu.edu ----------- ## 
## North Carolina State Univeristy --------------- ##
## Instructor: Dr. Wenbin Lu --------------------- ##
## ----------------------------------------------- ##

## loading proper packages
library(softImpute)
library(dplyr)

## plot the image
plotim = function(im){
  im <- t(im)
  im <- im[, ncol(im):1]
  image(z=im, col=gray(0:256/256),xaxt='n',yaxt='n', ann=FALSE)
}

## other matrix completion method
nucpen = function(masked,mask,rank=20,lambda=30){
  m = nrow(masked)
  n = ncol(masked)
  A = masked
  for(i in 1:m){
    for(j in 1:n){
      if(mask[i,j] == 0) A[i,j] = NA
    }
  }
  sol = softImpute(A,20,lambda = 30)
  return(sol$u %*% diag(sol$d) %*% t(sol$v))
}

## calculate mse
mse = function(x,recov) mean((x-recov)^2)

## calculate peak signal to noise ratio
psnr = function(x,recov){
  a = mse(x,recov)
  20*log(255,base=10) - 10*log(a,base=10)
}



## analyzing our data
info <- data.frame()
for(i in 0:2){
  for(j in c(0.2,0.5,0.8)){
    
    ## read in the datasets
    str_gm <- paste0("~/gm_pic_i_",i,"_j_",j,".csv")
    str_cbbmc <- paste0("~/cbbmc_pic_i_",i,"_j_",j,".csv")
    str_masked <- paste0("~/maskNA_pic_i_",i,"_j_",j,".csv")
    str_og <- paste0("~/raw_pic_i_",i,".csv")
    str_mask <- paste0("~/mask_i_",i,"_j_",j,".csv")
    gm <- as.matrix(read.csv(str_gm))
    cbbmc <- as.matrix(read.csv(str_cbbmc))
    masked <- as.matrix(read.csv(str_masked))
    mask <- as.matrix(read.csv(str_mask))
    og <- as.matrix(read.csv(str_og))
    
    ## other methods
    nuc <- nucpen(masked,mask,lambda=0)
    svt <- nucpen(masked,mask,lambda=30)
    
    ## scaling (gm and cbbmc are scaled in the python script)
    nuc <- 255*(nuc - min(nuc))/max(nuc)
    svt <- 255*(svt - min(svt))/max(svt)
    og <- 255*(og - min(og))/max(og)
    
    ## calculate mse for each method
    mse_vec <- c(mse(og,gm),
                 mse(og,cbbmc),
                 mse(og,nuc),
                 mse(og,svt))
    
    ## calculate psnr for each method
    psnr_vec <- c(psnr(og,gm),
                  psnr(og,cbbmc),
                  psnr(og,nuc),
                  psnr(og,svt))
    
    ## combine the info
    dat <- data.frame(mse=mse_vec,psnr=psnr_vec)
    dat$person <- i
    dat$mask_level <- j
    dat$method <- c("gm","cbbmc","nuc","svt")
    
    ## nothing better than building dataframes within the loop
    info <- rbind(info,dat)
    
  }
}

## analyzing the data
info %>% 
  group_by(mask_level,method) %>%
  summarise(mean(mse),
            mean(psnr))


