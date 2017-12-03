#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 00:13:47 2017

@author: eti
"""

def prepare_discriminator_data(tot_batches , gen ) :
    
    
    #pick 100 batches each  in general out of total_num_batches
    pos_sample =  np.random.randint(100, size=(1, tot_batches))          #6400
    neg_sample =            #6400
    
    #load batches with particular indexes for pos and neg samples
    
    
    pos_feat , pcap , ptarget , plengths  =                                     
    neg_feat , ncap , ntarget , nlengths =
    
    
    
    #run generator for negative samples :
    neg_cap = gen.forward(pos_feat) #generate captions
    
    #concatenate everything
    feat = torch.cat((pos_feat, neg_feat), 0).type(torch.FloatTensor)
    samples = [ pcap , neg_sample ]
    targets = torch.cat((pos , neg ), 0).type(torch.FloatTensor)
    lengths = torch.cat((plengths , nlengths ), 0).type(torch.FloatTensor)
    #shuffle them 
    
    
    #create batches and return
    
    
    
    return feat , samples, targets , lengths
    