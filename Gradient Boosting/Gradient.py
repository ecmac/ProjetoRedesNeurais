﻿# -*- coding: utf-8 -*-
"""mlp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KPnMi0_S7pdI-mQTJTF9xb-9-HfMLiNi
"""

# -*- coding: utf-8 -*-
import numpy as np
#import pandas as pd
#import tensorflow as tf

#from tensorflow.keras.models import Sequential, load_model
#from tensorflow.keras.layers import Dense, Dropout, Softmax
#from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

import scikitplot as skplt
import matplotlib.pyplot as plt

#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

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

def compute_performance_metrics(y, y_pred_class, y_pred_scores=None,
                                local_save='./', n_itera = 0, n_base = 0,
                                save = False):
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        axes = skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        fig = axes.get_figure()
        if save:
            fig.savefig(local_save + 'KS Statistic '+
                        str(n_itera) + '-' + str(n_base) + '.png',
                        bbox_inches="tight")
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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          local_save='./', n_itera = 0, n_base = 0, 
                          save = False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="grey" if cm[i, j] > thresh else "grey")
    fig.tight_layout()
    if save:
        fig.savefig(local_save + 'Confusion Matrix '+
                str(n_itera) + '-' + str(n_base) + '.png')
    plt.show()

    return cm


# %%
    
import pickle
import gzip
caminho = './Base de dados/'

def normalizar_base(caminho,n):        
    with gzip.open(caminho + 'base_de_dados_' + str(n) + '.pkl', 'rb') as arquivo:
          treino, validacao, teste = pickle.load(arquivo)
    
    treino_x = treino.iloc[:, 1:-2].values
    treino_y = treino.iloc[:, -2].values
    del treino
    
    validacao_x = validacao.iloc[:, 1:-2].values
    validacao_y = validacao.iloc[:, -2].values
    del validacao
    
    teste_x = teste.iloc[:, 1:-2].values
    teste_y = teste.iloc[:, -2].values
    del teste
    
    scaler = StandardScaler()
    treino_x = scaler.fit_transform(treino_x)
    validacao_x = scaler.transform(validacao_x)
    teste_x = scaler.transform(teste_x)
    
    return treino_x, treino_y, validacao_x, validacao_y, teste_x, teste_y

# %% Topologia do GD   
def criar_gradient_boosting(validation_fraction):
    return GradientBoostingClassifier(validation_fraction=validation_fraction, 
                                      random_state= 1, verbose=2)
# %%
scoring = ['roc_auc']
def criar_grid_search(model, param_grid):
    return GridSearchCV(model, param_grid, scoring=scoring, n_jobs= 3, 
                 iid=None, refit='roc_auc', cv=5,
                 verbose=2, pre_dispatch=6,
                 error_score='raise-deprecating', return_train_score=True)
# %%

n = 1

treino_x, treino_y, validacao_x, validacao_y, teste_x, teste_y = normalizar_base(caminho, n)

treino_x_expand = np.append(treino_x,validacao_x, axis = 0)
treino_y_expand = np.append(treino_y,validacao_y, axis = 0)
validation_fraction = validacao_x.shape[0] / (treino_x.shape[0] + validacao_x.shape[0])

del treino_x, validacao_x, treino_y, validacao_y


# %%
param_grid = {'learning_rate': [[0.8]], 
              'n_estimators': [[150]], 
              'max_depth': [[3]], 
              'max_features': [['auto']], 
              'n_iter_no_change': [[3]], 
              }
"""

loss=’deviance’, 
learning_rate=0.1,
n_estimators=100, 
subsample=1.0, 
min_samples_split=2, 
min_samples_leaf=1,
 min_weight_fraction_leaf=0.0, 
max_depth=3, 
min_impurity_decrease=0.0, 
min_impurity_split=None,
max_features=None,  
n_iter_no_change=None, 
tol=0.0001


param_grid = {'loss':[['deviance'], ['exponential']], 
              'learning_rate': [[0.001]], 
              'n_estimators': [[100],[1000]], 
              'subsample': [[1],[0.2]],
              'min_samples_split': [[2],[5]], 
              'min_samples_leaf': [[2],[5]], 
              'max_depth': [[3],[8]], 
              'min_impurity_decrease': [[0],[0.3]], 
              'max_features': [[None],['auto']], 
              'n_iter_no_change': [[3],[5]], 
              'tol': [[1e-4],[1e-6]]
              }

param_grid = {'loss':['deviance','exponential'], 
              'learning_rate': np.arange(0.01,0.2,0.1), 
              'n_estimators': np.arange(10,11), 
              'subsample': np.arange(0.1,0.2),
              'min_samples_split': np.arange(2,3), 
              'min_samples_leaf': np.arange(1,2), 
              'max_depth': np.arange(3,4), 
              'min_impurity_decrease': np.arange(0.0,0.1), 
              'max_features': [None,'auto'], 
              'n_iter_no_change': np.arange(3,4), 
              'tol': [1e-4]
              }

param_grid = {'loss':[['deviance'], ['exponential']], 
              'learning_rate': [[0.001],[0.01],[0.1]], 
              'n_estimators': [[100],[500],[1000]], 
              'subsample': [[1],[0.5],[0.2]],
              'min_samples_split': [[2],[5]], 
              'min_samples_leaf': [[2],[5]], 
              'max_depth': [[3],[8]], 
              'min_impurity_decrease': [[0],[0.3]], 
              'max_features': [[None],['auto']], 
              'n_iter_no_change': [[3],[5]], 
              'tol': [[1e-4],[1e-6]]
              }
"""
param_grid = ParameterGrid(param_grid)

# %%

gd = criar_gradient_boosting(validation_fraction)

grid_search = criar_grid_search(gd, param_grid)

grid_search.fit(treino_x_expand, treino_y_expand)


# %%

pred_class = grid_search.predict(teste_x)
pred_scores = grid_search.predict_proba(teste_x)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(
  teste_y, pred_class, pred_scores)

print('Performance no conjunto de validação:')
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)

cm = plot_confusion_matrix(teste_y,pred_class, classes=[0,1], normalize=True,
                      title='Normalized confusion matrix')

# %%
"""
    cm = plot_confusion_matrix(teste_y,mlp_pred_class, classes=[0,1], normalize=True,
                      title='Normalized confusion matrix',
                      local_save='./CM/', n_itera = cont, n_base = n)
  
    valor_salvo_ant = pd.DataFrame([[etiqueta,'accuracy', accuracy],
                           [etiqueta,'recall', recall],
                           [etiqueta,'precision', precision],
                           [etiqueta,'f1', f1],
                           [etiqueta,'auroc', auroc],
                           [etiqueta,'aupr', aupr],
                           [etiqueta,'confusion matrix', cm[0][0],cm[0][1], cm[1][0],cm[1][1]]
                           ])
    valor_salvo = valor_salvo.append(valor_salvo_ant, ignore_index=True)
    valor_salvo.to_excel('results-mlp.xlsx',header = None, index = False)

cont += 1
with open('cont.txt', 'w') as arq_cont:
    arq_cont.write(str(cont))
"""
# %%
from joblib import dump, load

dump(grid_search, './Gradient Boosting/modelo_GD_com_GS.joblib') 
grid_search = load('./Gradient Boosting/modelo_GD_com_GS.joblib')
# %%

grid_search.best_params_
