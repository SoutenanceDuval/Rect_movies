# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:01:43 2023

@author: cleme
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from itertools import combinations
from time import perf_counter
import pickle

plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'

nb_cores = 64
worker_list = [i+1 for i in range(nb_cores)]
key_save = 'mp2'
segment_index = [i for i in range(nb_cores)]

radius = 0.02
N_r = 100
N_b = 100

nb_collisions_square = 2*10**3
nb_collisions_rect = 3*10**4
# nb_collisions_square = 10*N_r
# nb_collisions_rect = 10*N_r

temperature_r = 1
temperature_b = 10**-2


worker_min, worker_max = 1,1000

clef = '/users/champ/cduval/Documents/Scripts/KinTh/data1/'
clef = ''

sum_vr_tot = np.zeros(nb_collisions_rect + nb_collisions_square)
sum_vb_tot = np.zeros(nb_collisions_rect + nb_collisions_square)
sum_er_tot = np.zeros(nb_collisions_rect)
sum_eb_tot = np.zeros(nb_collisions_rect)
compteur_tot = 0
compteur_collect = 0

collect_er_tot = {}
collect_eb_tot = {}

for worker in range(worker_min, worker_max):
    try:
        with open(clef+key_save+'data_rectN%.0fradius%.3fTa%.3fTb%.3fworker%.0f.pickle'%(N_r + N_b, radius, temperature_r, temperature_b, worker), 'rb') as handle:
            collect_vr, collect_vb, collect_er, collect_eb, sum_vr, sum_vb, sum_er, sum_eb, compteur = pickle.load(handle)
        print(worker, compteur)
        compteur_tot += compteur
        sum_vr_tot += sum_vr
        sum_vb_tot += sum_vb
        sum_er_tot += sum_er
        sum_eb_tot += sum_eb
        
        for key in collect_er:
            if compteur_collect < 5000:
                compteur_collect += 1
                collect_er_tot[compteur_collect] = collect_er[key]
                collect_eb_tot[compteur_collect] = collect_eb[key]
    
    except:
        pass

sum_vr_tot /= compteur_tot
sum_vb_tot /= compteur_tot
sum_er_tot /= compteur_tot
sum_eb_tot /= compteur_tot

print('Nbre configurations:', compteur_tot)


fig2, axs2 = plt.subplots(1,1,figsize=(3, 2.5))
axs2.plot(sum_vr_tot, c='red')
axs2.plot(sum_vb_tot, c='blue')
# axs2.plot(sum_eb**0 * 0.5 * (sum_er[0] + sum_eb[0]), c='black')

axs2.set_ylim(-0.0, 1.505)
axs2.set_xlim(-5, 30000)

# for key in collect_vr:
#     if key < 140:
#         axs2.plot(collect_eb[key], c='blue', linewidth=0.1)
#         axs2.plot(collect_er[key], c='red', linewidth=0.1)




fig1, axs1 = plt.subplots(1,1,figsize=(3, 1.5))
x = np.array([ii - 2000 for ii in range(nb_collisions_rect)])

axs1.plot(x, sum_er_tot, c='red')
axs1.plot(x, sum_eb_tot, c='blue')
axs1.plot(x, sum_eb_tot**0 * (sum_er_tot[0]+sum_eb_tot[0]), c='black', linestyle='--', zorder=-1)
axs1.plot(x, sum_eb_tot**0 * 0.5 * (sum_er[0] + sum_eb[0]), c='black')

axs1.set_ylim(-5, 105)
axs1.set_xlim(-1500, 7000)

axs1.spines['top'].set_visible(False)
axs1.spines['right'].set_visible(False)

axs1.annotate('', xy=(1.1, 0), xycoords='axes fraction', xytext=(-0.01, 0),
              arrowprops=dict(arrowstyle='-|>', color='black'))
axs1.annotate('', xy=(0, 1.1), xycoords='axes fraction', xytext=(0, -0.015),
              arrowprops=dict(arrowstyle='-|>', color='black'))

mat_er = np.zeros((nb_collisions_rect, [key for key in collect_er_tot][-1]))
mat_eb = np.zeros((nb_collisions_rect, [key for key in collect_er_tot][-1]))

for key in collect_er_tot:
    i, j = 0, 0
    for uu in collect_er_tot[key]:
        mat_er[i, key-1] = uu
        i += 1
    for vv in collect_eb_tot[key]:
        mat_eb[j, key-1] = vv
        j += 1   
    # if key < 20:
    #     axs1.plot(collect_er_tot[key], c='black', linewidth=0.1, alpha=0.1, zorder=-100)
        # axs1.plot(, c='red', linewidth=0.1)

sigma_er = (np.array([np.std(mat_er[ii, :]) for ii in range(nb_collisions_rect)]))
axs1.fill_between(x, sum_er_tot-sigma_er, sum_er_tot+sigma_er, color='red', zorder=-10, alpha=0.2, edgecolor='white')

sigma_eb = (np.array([np.std(mat_eb[ii, :]) for ii in range(nb_collisions_rect)]))
axs1.fill_between(x, sum_eb_tot-sigma_eb, sum_eb_tot+sigma_eb, color='blue', zorder=-11, alpha=0.2, edgecolor='white')

chemin = 'C:\\Users\\cleme\\Documents\\LKB\\Boltzmann\\soutenance\\'
fig1.savefig(chemin+'two_gases_billiard.png', bbox_inches="tight", dpi=500)
fig1.savefig('two_gases_billiard.png', bbox_inches="tight", dpi=500)
