'''
Created on 2016-02-11

@author: Niranjan
'''

import numpy as np
import sys
from datetime import datetime

# set for testing purpose. Generates the same random number
np.random.seed(0)

# a class that trains the data using probabilistic latent sematic indexing
class PlsaRecc(object):    

    def __init__(self,n_topic):
        '''
        Initialize empty document list.
        '''
        self.documents = []
        self.usr_itm_gtruth = np.load('path_to_data')
        self.usr_itm_gtruth = self.usr_itm_gtruth.astype('int64')
        self.t_usrs = self.usr_itm_gtruth.shape[0]
        self.t_itms = self.usr_itm_gtruth.shape[1]
        # total topics
        self.t_tpic = n_topic
        # p(z)  topic probability
        self.tp_prb = np.zeros([self.t_tpic,], dtype = np.float)
        # P(z | u) user topic probability
        self.usr_tp_prb = np.zeros([self.t_usrs, self.t_tpic], dtype=np.float)
        # P(z | i) item topic probability
        self.tp_itm_prb = np.zeros([self.t_tpic, self.t_itms], dtype=np.float)
        # posterior P(z | u, i)  the item-topic matrix for every user
        self.postr_tp_prb = np.zeros([self.t_usrs, self.t_itms, self.t_tpic], dtype=np.float)
        # randomly initialize the parameters
        self.init_random_param()
    
    # method to normalize 2-D array
    def __normalize_mat(self,mat):
        
        if (isinstance(mat, list)):
            for m in mat:
                norm_sum = m.sum(axis=1)
                m /= norm_sum[:, np.newaxis]
        else:
            if(mat.ndim == 2):
                norm_sum = mat.sum(axis=1)
            else:
                norm_sum = mat.sum()
            mat /= norm_sum[:, np.newaxis]
        
    # method that assigning random initial values to the model parameters
    def init_random_param(self):
        self.usr_tp_prb = np.random.random(size = (self.t_usrs, self.t_tpic))
        self.tp_itm_prb = np.random.random(size = (self.t_tpic, self.t_itms))
        # uniform probability for all users
        self.tp_prb += (1/float(self.t_tpic))
        # normalize all parameters
        self.__normalize_mat([self.usr_tp_prb, self.tp_itm_prb])
        print 'ok'
        

    def add_document(self, document):
        '''
        Add a document to the corpus.
        '''
        self.documents.append(document)
        

    def plsa(self,max_iter):
        
        for itr in range(max_iter):
            print 'done with {} iteration'.format(itr)
            # sample a topic for z for every user and items the E step
            for usr in range(self.t_usrs):
                for itm in range(self.t_itms):
                    p_z_ui = self.usr_tp_prb[usr] * self.tp_itm_prb[:,itm] * self.tp_prb
                    if sum(p_z_ui) == 0.0:
                        pass
                    else:
                        if(np.sum(p_z_ui) > 1):
                            self.__normalize_mat(p_z_ui)
                    # update the posterior
                    self.postr_tp_prb[usr][itm] = p_z_ui
            

            print "M step:"
            # obtain p(z)
            for z in range(self.t_tpic):
                tmp_p_z = 0
                for usr in range(self.t_usrs):
                    for itm in range(self.t_itms):
                        n_i_u = self.usr_itm_gtruth[usr][itm]
                        tmp_p_z += n_i_u * self.postr_tp_prb[usr, itm, z]
                self.tp_prb[z] = tmp_p_z

            
            # obtain the item topic probability P(i | z)
            for z in range(self.t_tpic):
                for itm in range(self.t_itms):
                    tmp_p_zi = 0
                    for usr in range(self.t_usrs):
                        # get the count the number of times the user u picked the item i
                        n_i_u = self.usr_itm_gtruth[usr][itm]
                        tmp_p_zi += n_i_u * self.postr_tp_prb[usr, itm, z]
                    self.tp_itm_prb[z][itm] = tmp_p_zi

             
            # update P(u | z)
            for z in range(self.t_tpic):
                for usr in range(self.t_usrs):
                    tmp_p_zu = 0
                    for itm in range(self.t_itms):
                        # get the count the number of times the user u picked the item i
                        n_i_u = self.usr_itm_gtruth[usr][itm]
                        tmp_p_zu += n_i_u * self.postr_tp_prb[usr, itm, z]
                    self.usr_tp_prb[usr][z] = tmp_p_zu


if __name__ == '__main__':
    
    recc_obj = PlsaRecc(5)
    recc_obj.plsa(1)
