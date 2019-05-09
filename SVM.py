# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:24:51 2019

@author: sergi
"""
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score

import scikitplot as skplt
import matplotlib
import matplotlib.pyplot as plt

import pickle
import gzip
from sklearn.svm import SVC
caminho = './base de dados'
    
# %%

def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        plt.show()
        y_pred_scores = y_pred_scores[:, 1]
        auroc = roc_auc_score(y, y_pred_scores)
        aupr = average_precision_score(y, y_pred_scores)
        performance_metrics = performance_metrics + (auroc, aupr)
    return performance_metrics

def print_metrics_summary(accuracy, recall, precision, f1, auroc=None, aupr=None):
    print()
    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))
    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))
    if auroc is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
    if aupr is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))

def print_metrics_summary2(accuracy, recall, precision, f1, arq, auroc=None, aupr=None):
    arq.write('\n')
    arq.write("{metric:<18}{value:.4f}\n".format(metric="Accuracy:", value=accuracy))
    arq.write("{metric:<18}{value:.4f}\n".format(metric="Recall:", value=recall))
    arq.write("{metric:<18}{value:.4f}\n".format(metric="Precision:", value=precision))
    arq.write("{metric:<18}{value:.4f}\n".format(metric="F1:", value=f1))
    if auroc is not None:
        arq.write("{metric:<18}{value:.4f}\n".format(metric="AUROC:", value=auroc))
    if aupr is not None:
        arq.write("{metric:<18}{value:.4f}\n".format(metric="AUPR:", value=aupr))

# %%
for n in range(5):
    
    with gzip.open(caminho + ' ' + str(n), 'rb') as arquivo:
        treino, validacao, teste = pickle.load(arquivo)
    
    svc_clf = SVC(probability = True, verbose = True, random_state = n)  # Modifique aqui os hyperparâmetros
    svc_clf.fit(treino.iloc[:, :-2], treino.iloc[:, -2])
    svc_pred_class = svc_clf.predict(validacao.iloc[:, :-2])
    svc_pred_scores = svc_clf.predict_proba(validacao.iloc[:, :-2])
    
    accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(validacao.iloc[:, -2], svc_pred_class, svc_pred_scores)
    print('Performance no conjunto de validação:')
    print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
    
    with open('resultados.txt', 'a+') as arq_resul:
        arq_resul.write('Resultado utilizando o banco de dados ' + str(n) + ':\n')
        print_metrics_summary2(accuracy, recall, precision, f1, arq_resul, auroc, aupr)
        arq_resul.write('\n')