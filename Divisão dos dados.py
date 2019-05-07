#!/usr/bin/env python
# coding: utf-8

# %%


import pandas as pd
import pickle
import gzip

# %%
def dividir_base(rand, tabela):
    bons = tabela.loc[tabela['IND_BOM_1_1'] == 1]
    maus = tabela.loc[tabela['IND_BOM_1_2'] == 1]
    
    testa_bons = bons[(3*bons.shape[0])//4:]
    testa_maus = maus[(3*maus.shape[0])//4:]
    
    bons = bons[:(3*bons.shape[0])//4]
    maus = maus[:(3*maus.shape[0])//4]
    
    bons_rand = bons.sample(frac = 1, random_state = rand)
    maus_rand = maus.sample(frac = 1, random_state = rand)
    
    treina_bons = bons_rand[:2*(bons_rand.shape[0])//3]
    valida_bons = bons_rand[2*(bons_rand.shape[0])//3:]
    
    treina_maus = maus_rand[:2*(maus_rand.shape[0])//3]
    valida_maus = maus_rand[2*(maus_rand.shape[0])//3:]
    
    treina_maus_over = pd.concat([treina_maus, treina_maus.sample(frac=treina_bons.shape[0]/treina_maus.shape[0] - 1)])
    valida_maus_over = pd.concat([valida_maus, valida_maus.sample(frac = valida_bons.shape[0]/valida_maus.shape[0] - 1)])
    
    treina_ambos = pd.concat([treina_bons, treina_maus_over])
    valida_ambos = pd.concat([valida_bons, valida_maus_over])
    testa_ambos = pd.concat([testa_bons, testa_maus])
    
    treina_ambos = treina_ambos.sample(frac = 1)
    valida_ambos = valida_ambos.sample(frac = 1)
    testa_ambos = testa_ambos.sample(frac = 1)

    return [treina_ambos, valida_ambos, testa_ambos]

# %%
tabela = pd.read_csv("TRN", sep="\t")
tabela = tabela.sample(frac = 1, random_state = 1)

for n in range(5):
    dados = dividir_base(n, tabela)
    with gzip.GzipFile('base de dados ' + str(n), 'wb') as arquivo:
        pickle.dump(dados, arquivo, 2)
        