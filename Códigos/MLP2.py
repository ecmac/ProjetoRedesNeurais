# -*- coding: utf-8 -*-
"""mlp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KPnMi0_S7pdI-mQTJTF9xb-9-HfMLiNi
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Softmax
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

import scikitplot as skplt
import matplotlib.pyplot as plt

#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import pickle
import gzip
caminho = './Base de dados/'
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
                                local_save='./', n_itera = 0, n_base = 0):
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        axes = skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        fig = axes.get_figure()
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
                          local_save='./', n_itera = 0, n_base = 0):
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
    #fig = plt.get_figure()
    fig.savefig(local_save + 'Confusion Matrix '+
                str(n_itera) + '-' + str(n_base) + '.png')
    plt.show()

    return cm


# %%
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

# %% mudança da topologia do MLP     
earlystop = [EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3,
                           verbose=1, mode='auto')]

def create_sklearn_compatible_model(c1, c2, c3, c4, c5, ac, ac2, lr, decay):
    model = Sequential()
    model.add(Dense(220, activation=ac, input_dim=243))
    model.add(Dropout(rate = 0.8))
    model.add(Dense(170, activation=ac))
    model.add(Dropout(rate = 0.7))
    model.add(Dense(70, activation=ac))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(30, activation=ac))
    model.add(Dropout(rate = 0.2))
    model.add(Dense(10, activation=ac))
    model.add(Dropout(rate = 0.1))
    model.add(Dense(1, activation=ac2))
    model.compile(optimizer = adam_novo(lr, decay), loss='mean_squared_error',
                  metrics = ['accuracy'])
    return model

def adam_novo(lr, decay):
    return tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                                     epsilon=1e-08, decay=decay)
# %%

with open('cont.txt', 'r') as arq_cont:
    cont = int(arq_cont.read())
    
cont += 1
ativacao = ['relu']
ativacao_final = ['sigmoid']
learning_rate = [1e-3]
decay = [1e-8]
valor_salvo = pd.read_excel('./results-mlp.xlsx')

#c1_list = list(range(150,250,10))
#c2_list = list(range(100,150,5))
#c3_list = list(range(50,100,10))
#c4_list = list(range(20,50,5))
#c5_list = list(range(2,20,2))


c1_list = [230]
c3_list  = [60]
c5_list = [14]
#c1_list.remove(220)
c2_list = [1]
c4_list = [1]

np.random.seed(100)
np.random.shuffle(c1_list)
#np.random.shuffle(c2_list)
#np.random.shuffle(c3_list)
#np.random.shuffle(c4_list)
np.random.shuffle(c5_list)

for dc in decay:
    for lr in learning_rate:
        for ac in ativacao:
            for ac2 in ativacao_final:
                for c1 in c1_list:
                    for c2 in c2_list:
                        for c3 in c3_list:
                            for c4 in c4_list:
                                for c5 in c5_list:
                                    with open('parâmetros do modelo.txt', 'a+') as arq:
                                        texto = '{0} - c1:{1}, c2:{2}, c3:{3}, c4:{4}, c5:{5}, ac:{6}, ac2:{7}, lr:{8}, dc:{9}\n'.format(
                                                cont, c1, c2, c3, c4, c5, ac, ac2, lr, dc)
                                        arq.write(texto)
                                    for n in range(5): 
                                        
                                        etiqueta = str(cont) + '-' + str(n)
                                        
                                        treino_x, treino_y, validacao_x, validacao_y, teste_x, teste_y = normalizar_base(caminho, n)
                                        
                                        mlp = create_sklearn_compatible_model(c1,c2,c3,c4,c5,ac,ac2,lr,dc)
                                    
                                        mlp.fit(treino_x, treino_y, callbacks = earlystop,
                                                  validation_data = (validacao_x, validacao_y),
                                                  use_multiprocessing=True, workers=8,
                                                  batch_size=500, epochs=200, verbose=1)
                                        
                                        mlp.save('./Models/model ' + etiqueta + '.h5')
                                        
                                        mlp_pred_class = mlp.predict_classes(teste_x, verbose = 1, batch_size = len(teste_y))
                                        mlp_pred_scores = mlp.predict_proba(teste_x, verbose = 1, batch_size = len(teste_y))
                                        mlp_pred_scores = np.append(mlp_pred_scores, 
                                                                    1 - mlp_pred_scores, axis = -1)[:, [1, 0]]
                                        
                                        accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(
                                          teste_y, mlp_pred_class, mlp_pred_scores, 
                                          local_save='./KS/', n_itera = cont, n_base = n)
                                        
                                        print('Performance no conjunto de validação:')
                                        print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
                                    
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