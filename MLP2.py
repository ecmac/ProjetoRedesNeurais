# -*- coding: utf-8 -*-
"""
Created on Sat May  4 10:47:11 2019

@author: sergi
"""

import numpy as np
import pandas as pd
import tensorflow as tf
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
caminho = './base de dados'

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier


# %%

def extract_final_losses(history):
    """Função para extrair o melhor loss de treino e validação.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    Dicionário contendo o melhor loss de treino e de validação baseado 
    no menor loss de validação.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    idx_min_val_loss = np.argmin(val_loss)
    return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}

def plot_training_error_curves(history):
    """Função para plotar as curvas de erro do treinamento da rede neural.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    A função gera o gráfico do treino da rede e retorna None.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    fig, ax = plt.subplots()
    ax.plot(train_loss, label='Train')
    ax.plot(val_loss, label='Validation')
    ax.set(title='Training and Validation Error Curves', xlabel='Epochs', ylabel='Loss (MSE)')
    ax.legend()
    plt.show()

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
# %% mudança da topologia do MLP

def create_sklearn_compatible_model():
    model = Sequential()
    model.add(Dense(200, activation='tanh', input_dim=244))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='SGD', loss='mean_squared_error')
    return model
    
#%% apenas rodar em uma base
for n in range(1):
    
    with gzip.open(caminho + ' ' + str(n), 'rb') as arquivo:
        treino, validacao, teste = pickle.load(arquivo)
# %%
mlp_clf = KerasClassifier(build_fn=create_sklearn_compatible_model, 
                      batch_size=64, epochs=100,
                      verbose=0)

mlp_clf.fit(treino.iloc[:10000, :-2], treino.iloc[:10000, -2])
mlp_pred_class = mlp_clf.predict(validacao.iloc[:10000, :-2])
mlp_pred_scores = mlp_clf.predict_proba(validacao.iloc[:10000, :-2])

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(validacao.iloc[:10000, -2], mlp_pred_class, mlp_pred_scores)

print('Performance no conjunto de validação:')
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)

with open('resultados-mlp.txt', 'a+') as arq_resul:
    arq_resul.write('Resultado utilizando o banco de dados ' + str(n) + ':\n')
    print_metrics_summary2(accuracy, recall, precision, f1, arq_resul, auroc, aupr)
    arq_resul.write('\n')