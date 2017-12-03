#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 02:40:28 2017

@author: eti
"""

from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers


CUDA = True
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20
START_LETTER = 0
BATCH_SIZE = 64
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 50
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

#oracle_samples_path = './oracle_samples.trc'
#oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
#pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
#pretrained_dis_path = './dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'


def train_generator_MLE(gen, gen_opt, epochs , num_batches) : #oracle, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    
    
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0.0

        #for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
        for i in range(num_batches) :
            #get batch_data
            inp, target = #helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                          #                                gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data[0]

            if (i / BATCH_SIZE)  == 10 : #% ceil(
            #                ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
           #     print('.', end='')
                sys.stdout.flush()

        mle_train_loss.append((total_loss / (i+1)))
        # each loss in a batch is loss per sample
        #total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        # sample from generator and compute oracle NLL
        #oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
        #                                            start_letter=START_LETTER, gpu=CUDA)
        
        #run_validation
        vinp, vtarget =
        vloss = gen.batchNLLLoss(vinp, vtarget)
        mle_valid_loss.append(vloss.data[0])
        
        #save losses
        np.save('mle_train_loss.npy' , mle_train_loss)
        np.save('mle_valid_loss.npy' , mle_valid_loss)
        
        
        print(' average_train_NLL = %.4f'  % (mle_train_loss[-1]) , 
              ' average_valid_NLL = %.4f'  % (mle_valid_loss[-1])) #oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))


def train_generator_PG(gen, gen_opt, dis, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    tloss =  0.0
    for batch in range(num_batches):
        #s = gen.sample(BATCH_SIZE)        # 64 works best
        inp, target =  #helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        
        rewards = dis.batchClassify( target)
        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

        tloss = tloss + pg_loss.data[0]
       
    # sample from generator and compute oracle NLL
    #oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
    #                                               start_letter=START_LETTER, gpu=CUDA)

    #print(' oracle_sample_NLL = %.4f' % oracle_loss)
    tloss = tloss / ( batch + 1 )
    #run_validation
    vinp, vtarget =
    rewards = dis.batchClassify( vtarget)
    vloss =  gen.batchPGLoss(vinp, vtarget, rewards)
    #mle_valid_loss.append(vloss.data[0])   


    return tloss, vloss




def train_discriminator(discriminator, dis_opt, d_steps, epochs , tot_batches , gen ):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    #pos_val = oracle.sample(100)
    #neg_val = generator.sample(100)
    #val_inp, val_target = #helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        #s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        #dis_inp, dis_target = #helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        
        dis_ft , dis_inp, dis_target =  helpers.prepare_discriminator_data(tot_batches , gen )        
        
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, len(dis_inp)) :   #2 * POS_NEG_SAMPLES, BATCH_SIZE):
                feat , inp, target = disp_ft[i] , dis_inp[i] , dis_target[i]         
                #dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify( feat , inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data[0]
                total_acc += torch.sum((out>0.5)==(target>0.5)).data[0]

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data[0]/200.))

# MAIN
if __name__ == '__main__':
    
    #oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    #oracle.load_state_dict(torch.load(oracle_state_dict_path))
    #oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)
    # a new oracle can be generated by passing oracle_init=True in the generator constructor
    # samples for the new oracle can be generated using helpers.batchwise_sample()

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

    if CUDA:
        #oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()
        #oracle_samples = oracle_samples.cuda()

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)    
    train_generator_MLE(gen, gen_optimizer, MLE_TRAIN_EPOCHS)


    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(dis, dis_optimizer, gen, 50, 3 , tot_batches , gen)

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                               start_letter=START_LETTER, gpu=CUDA)
    print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)


    genpg_valid_loss = list()
    genpg_train_loss = list()
    
    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        tloss , vloss = train_generator_PG(gen, gen_optimizer,dis,num_batches)
        
        #save losses
        genpg_train_loss.append(tloss)
        genpg_valid_loss.append(vloss)
        
        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer,gen, oracle, 5, 3 , tot_batches , gen)
        
        
np.save('genpg_train_loss.npy' , genpg_train_loss)
np.save('genpg_valid_loss.npy' , genpg_valid_loss)
        