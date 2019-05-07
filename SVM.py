# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:24:51 2019

@author: sergi
"""

import pickle
import gzip
from sklearn.svm import SVC
caminho = './base de dados'



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



# Só mudar o número 0 para outro, de 1 a 4, para acessar as outras bases
with gzip.open(caminho + ' 0', 'rb') as arquivo:
    treino, validacao, teste = pickle.load(arquivo)


# %%
#treino.iloc[:1000, :-2]
#treino.iloc[:, -2]
svc_clf = SVC(probability=True)  # Modifique aqui os hyperparâmetros
svc_clf.fit(treino.iloc[:10000, :-2], treino.iloc[:10000, -2])
svc_pred_class = svc_clf.predict(validacao.iloc[:10000, :-2])
svc_pred_scores = svc_clf.predict_proba(validacao.iloc[:10000, :-2])


accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(validacao.iloc[:10000, -2], svc_pred_class, svc_pred_scores)
print('Performance no conjunto de validação:')
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)


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