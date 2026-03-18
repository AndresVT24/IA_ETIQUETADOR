__authors__ = ['1782649', '1745722', '1691333']
__group__ = '10'

import numpy as np
import math
import operator
import os
import utils
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        train_data = train_data.astype(float)
        self.train_data = train_data.reshape(train_data.shape[0], -1)


    def get_k_neighbours(self, test_data, k):
        """
        Funció que pren com a entrada el conjunt de test que volem etiquetar
        (test_data) i fa el següent:
        1. Canvia les dimensions de les imatges de la mateixa manera que ho hem fet amb
        el conjunt de entrenament.
        2. Calcula la distància entre les mostres del test_data i les del train_data.
        3. Guarda a la variable de classe self.neighbors les K etiquetes de les imatges més
        pròximes per a cada mostra del test.

        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        test_data = test_data.astype(float)
        test_data = test_data.reshape(test_data.shape[0], -1)

        distancia = cdist(test_data, self.train_data)
        cercanos =  np.argsort(distancia, axis=1)[:,:k]
        self.neighbors = self.labels[cercanos]

    def get_class(self):
        print(self.neighbors)
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets 
                the most voted value (i.e. the class at which that row belongs)
        """
        clases = []

        for fila in self.neighbors:
            valores, idx, counts = np.unique(fila, return_index=True, return_counts=True)

            orden = np.argsort(idx)
            valores = valores[orden] 
            idx = idx[orden]
            counts = counts[orden] 
            
            max_count = np.max(counts)
            pos = np.argmax(counts)
            clases.append(valores[pos])

        return np.array(clases)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
    
