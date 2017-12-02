#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 22:48:59 2017

@author: eti
"""

#splitting data into train , split , valid



num_batches = 517
val_id = np.load('val.npy')
test_id = np.load('test.npy')

val_data ={}
train_batch = {}
test_data = { }

for b in num_batches : 
    ft , obj , att , vids = load_batch(b)
    vids = np.array(vids)
    #get indexes of common test data
    ind = np.nonzero(np.in1d(vids, val_id))[0]    
    for i in ind :   
        #get index of ones
        if np.any(obj[i,:]==1) :
            obj_l = np.where(obj[i,:]==1)
        if np.any(att[i,:]==1) :   
            att_l = np.where(att[i,:]==1)           
            
        test_data['features'].append(ft[i,:])    
        test_data['object'].append(obj_l)    
        test_data['atts'].append(att[i,:])
        
    #get indexes of common valid data
    ind = np.nonzero(np.in1d(vids, val_id))[0]
    for i in ind :   
        #get index of ones
        if np.any(obj[i,:]==1) :
            obj_l = np.where(obj[i,:]==1)
        if np.any(att[i,:]==1) :   
            att_l = np.where(att[i,:]==1)           
            
        val_data['features'].append(ft[i,:])    
        val_data['object'].append(obj_l)    
        val_data['atts'].append(att[i,:])
        
    #get  indexes of train data
    ind =     