# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:24:51 2019

@author: sergi
"""

import pickle
import gzip
caminho = './base de dados'

# Só mudar o número 0 para outro, de 1 a 4, para acessar as outras bases
with gzip.open(caminho + ' 0', 'rb') as arquivo:
    treino, validacao, teste = pickle.load(arquivo)
