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
import multiprocessing as multipro
import os
import pickle

id_job = int(os.getenv('SLURM_JOBID'))
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

def overlaps(coord_particle1, coord_particle2):   
    return np.hypot(*(coord_particle1 - coord_particle2)) <= 2*radius

def generate_positions(N_loc, x_length_loc, y_length_loc):
    r0 = np.zeros((2, N_loc))
    for uu in range(N_loc):
        while 1:
            Ax = radius + (x_length_loc - 2*radius) * np.random.random()
            Ay = radius + (y_length_loc - 2*radius) * np.random.random()
            coord_particleA = np.array([Ax, Ay])
            overlap_loc = False
            for index in range(N_r):
                coord_particleB = r0[:, index]
                if overlaps(coord_particleA, coord_particleB):
                    overlap_loc = True
                    break
            if not overlap_loc:
                r0[:, uu] = coord_particleA
                break
    return r0

def generate_speeds(N_loc, temperature_loc):
    v0 = np.zeros((2, N_loc))
    v0x = (np.random.random(N_loc) - 0.5)
    v0y = (np.random.random(N_loc) - 0.5)
    norm_v = np.sqrt(v0x**2 + v0y**2)
    energy_per_particle = np.sum(0.5 * norm_v**2)/N_loc
    v0x *= np.sqrt(temperature_loc/energy_per_particle)
    v0y *= np.sqrt(temperature_loc/energy_per_particle)
    # norm_v = np.sqrt(v0x**2 + v0y**2)
    # energy_per_particle = np.sum(0.5 * norm_v**2)/N_loc
    v0[0][:], v0[1][:] = v0x, v0y
    return v0

def next_collision_pair(pos1, pos2, vit1, vit2, t0):
    Delta_pos = pos1 - pos2
    Delta_vit = vit1 - vit2
    Discriminant = (np.dot(Delta_pos, Delta_vit))**2 - np.dot(Delta_vit, Delta_vit)*(np.dot(Delta_pos, Delta_pos) - 4*radius**2)
    if np.dot(Delta_pos, Delta_vit) < 0:
        approaching = True
    else:
        approaching = False
    if approaching and Discriminant>0:
        t_pair = t0 - (np.dot(Delta_pos, Delta_vit) + np.sqrt(Discriminant))/np.dot(Delta_vit, Delta_vit)
    else:
        t_pair = 10**100
    return t_pair

def next_collision_wall_square(pos1, vit1, t0, x_length_loc, y_length_loc):
    posx, posy = pos1
    vitx, vity = vit1
    tposs = []
    if vitx != 0.0:
        tposs.append(np.abs((np.sign(vitx) + 1)*x_length_loc / 2 - posx - np.sign(vitx)*radius)/np.abs(vitx))
    if vity != 0.0:
        tposs.append(np.abs((np.sign(vity) + 1)*y_length_loc / 2 - posy - np.sign(vity)*radius)/np.abs(vity))
        
    return t0 + np.min(tposs)

def collision_pair(pos1, pos2, vit1, vit2):
    Delta_pos = pos1 - pos2
    Delta_vit = vit1 - vit2
    e_normal = Delta_pos / np.linalg.norm(Delta_pos)
    new_vit1 = vit1 - e_normal * (np.dot(Delta_vit, e_normal))
    new_vit2 = vit2 + e_normal * (np.dot(Delta_vit, e_normal))
    return new_vit1, new_vit2

def collision_wall_square(pos1, vit1, x_length_loc, y_length_loc):
    posx, posy = pos1
    vitx, vity = vit1
    digits_thresh = 8
    if np.round(posx - radius, digits_thresh) == 0.0 or np.round(posx + radius, digits_thresh) == x_length_loc:
        vitx *= -1
    elif np.round(posy - radius, digits_thresh) == 0.0 or np.round(posy + radius, digits_thresh) == y_length_loc:
        vity *= -1
    vit1 = np.array([vitx, vity])
    return vit1

def event_disks_square(r_loc, v_loc, t_loc, N_loc, x_length_loc, y_length_loc):
    
    tps_min_pair = 10**10
    for pair in combinations(np.arange(N_loc),2):
        ind1, ind2 = pair
        pos1 = r_loc[:, ind1]
        pos2 = r_loc[:, ind2]
        vit1 = v_loc[:, ind1]
        vit2 = v_loc[:, ind2]
        tps_pair = next_collision_pair(pos1, pos2, vit1, vit2, t_loc)
        if tps_pair < tps_min_pair:
            tps_min_pair = tps_pair
            next_pair = pair
            
    tps_min_wall = 10**10
    for ind1 in range(N_loc):
        pos1 = r_loc[:, ind1]
        vit1 = v_loc[:, ind1]
        tps_wall = next_collision_wall_square(pos1, vit1, t_loc, x_length_loc, y_length_loc)
        if tps_wall < tps_min_wall:
            tps_min_wall = tps_wall
            next_ind = ind1
            
    bool_wall, bool_pair = 0, 0
    t_next = min(tps_min_pair, tps_min_wall)
    r_loc += (t_next - t_loc)*v_loc
        
    if tps_min_wall < tps_min_pair:
        # print('Wall. t_loc', t_loc, 't_next', t_next, next_ind)
        pos1 = r_loc[:, next_ind]
        vit1 = v_loc[:, next_ind]
        v_loc[:, next_ind] = collision_wall_square(pos1, vit1, x_length_loc, y_length_loc)
        bool_wall = 1
        
    else:
        # print('Binary collision. t_loc', t_loc, 't_next', t_next)
        ind1_next, ind2_next = next_pair
        new_vit1, new_vit2 = collision_pair(r_loc[:, ind1_next], r_loc[:, ind2_next], v_loc[:, ind1_next], v_loc[:, ind2_next])
        v_loc[:, ind1_next] = new_vit1
        v_loc[:, ind2_next] = new_vit2
        bool_pair = 1
        
    return r_loc, v_loc, t_next, bool_wall, bool_pair

def next_collision_wall_rect(pos1, vit1, t0, x_length_loc, y_length_loc):
    posx, posy = pos1
    vitx, vity = vit1
    tposs = []
    if vitx != 0.0:
        tposs.append(np.abs((np.sign(vitx) + 1)*x_length_loc / 2 - posx - np.sign(vitx)*radius)/np.abs(vitx))
    if vity != 0.0:
        tposs.append(np.abs((np.sign(vity) + 1)*y_length_loc / 2 - posy - np.sign(vity)*radius)/np.abs(vity))
        
    return t0 + np.min(tposs)

def collision_wall_rect(pos1, vit1, x_length_loc, y_length_loc):
    posx, posy = pos1
    vitx, vity = vit1
    digits_thresh = 8
    if np.round(posx - radius, digits_thresh) == 0.0 or np.round(posx + radius, digits_thresh) == x_length_loc:
        vitx *= -1
    elif np.round(posy - radius, digits_thresh) == 0.0 or np.round(posy + radius, digits_thresh) == y_length_loc:
        vity *= -1
    vit1 = np.array([vitx, vity])
    return vit1

def event_disks_rect(r_loc, v_loc, t_loc, N_loc, x_length_loc, y_length_loc):
    
    tps_min_pair = 10**10
    for pair in combinations(np.arange(N_loc),2):
        ind1, ind2 = pair
        pos1 = r_loc[:, ind1]
        pos2 = r_loc[:, ind2]
        vit1 = v_loc[:, ind1]
        vit2 = v_loc[:, ind2]
        tps_pair = next_collision_pair(pos1, pos2, vit1, vit2, t_loc)
        if tps_pair < tps_min_pair:
            tps_min_pair = tps_pair
            next_pair = pair
            
    tps_min_wall = 10**10
    for ind1 in range(N_loc):
        pos1 = r_loc[:, ind1]
        vit1 = v_loc[:, ind1]
        tps_wall = next_collision_wall_rect(pos1, vit1, t_loc, x_length_loc, y_length_loc)
        if tps_wall < tps_min_wall:
            tps_min_wall = tps_wall
            next_ind = ind1
            
    bool_wall, bool_pair = 0, 0
    t_next = min(tps_min_pair, tps_min_wall)
    r_loc += (t_next - t_loc)*v_loc
        
    if tps_min_wall < tps_min_pair:
        # print('Wall. t_loc', t_loc, 't_next', t_next, next_ind)
        pos1 = r_loc[:, next_ind]
        vit1 = v_loc[:, next_ind]
        v_loc[:, next_ind] = collision_wall_rect(pos1, vit1, x_length_loc, y_length_loc)
        bool_wall = 1
        
    else:
        # print('Binary collision. t_loc', t_loc, 't_next', t_next)
        ind1_next, ind2_next = next_pair
        new_vit1, new_vit2 = collision_pair(r_loc[:, ind1_next], r_loc[:, ind2_next], v_loc[:, ind1_next], v_loc[:, ind2_next])
        v_loc[:, ind1_next] = new_vit1
        v_loc[:, ind2_next] = new_vit2
        bool_pair = 1
        
    return r_loc, v_loc, t_next, bool_wall, bool_pair



###Listener core does all the savings
def listener_core(output_qloc):
    compteur = 0
    collect_vr = {}
    collect_vb = {}
    collect_er = {}
    collect_eb = {}
    sum_vr = np.zeros(nb_collisions_rect + nb_collisions_square)
    sum_vb = np.zeros(nb_collisions_rect + nb_collisions_square)
    sum_er = np.zeros(nb_collisions_rect)
    sum_eb = np.zeros(nb_collisions_rect)
    
    while 1:
        v_r_moy, v_b_moy, energy_r, energy_b = output_qloc.get()
                
        compteur += 1
        
        sum_vr += v_r_moy
        sum_vb += v_b_moy
        sum_er += np.array(list(energy_r) + [10**5]*(nb_collisions_rect - len(energy_r)))
        sum_eb += np.array(list(energy_b) + [10**5]*(nb_collisions_rect - len(energy_b)))
        
        if compteur < 100:
            collect_vr[compteur] = v_r_moy
            collect_vb[compteur] = v_b_moy
            collect_er[compteur] = energy_r
            collect_eb[compteur] = energy_b
        
        worker_save = worker_list[0] + id_job % nb_cores
        to_be_saved = [collect_vr, collect_vb, collect_er, collect_eb, sum_vr, sum_vb, sum_er, sum_eb, compteur]
        
        if compteur % 5 == 0:
            with open(key_save+'data_rectN%.0fradius%.3fTa%.3fTb%.3fworker%.0f.pickle'%(N_r + N_b, radius, temperature_r, temperature_b, worker_save), 'wb') as handle:
                pickle.dump(to_be_saved, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    return 1

###Worker cores do all the work
def worker_core(xx, output_qloc):
    
    worker = worker_list[xx]+1
    seed0 = perf_counter()+1
    seed = int(seed0*worker % 2**31)
    np.random.seed(seed)
    
    while 1:
    
        v_r_moy = []
        v_b_moy = []
        energy_r = []
        energy_b = []
        
        #red box
        x_length = 1    
        y_length = 1        
        
        #Ensemble average: generate random initial position configurations
        r0 = generate_positions(N_r, x_length, y_length)
        #Random speed for given temperature (i.e., given energy)
        v0 = generate_speeds(N_r, temperature_r)
        
        t0 = 0
        nb_collisions_walls = 0
        nb_collisions_pairs = 0
                
        while nb_collisions_pairs < nb_collisions_square:
            R, V, T, bool_wall, bool_pair = event_disks_square(np.copy(r0), np.copy(v0), t0, N_r, x_length, y_length)
            nb_collisions_walls += bool_wall
            nb_collisions_pairs += bool_pair
                
            if bool_pair == 1:
                Vx, Vy = V
                V_red_moy = np.mean(np.sqrt(Vx**2 + Vy**2))
                v_r_moy.append(V_red_moy)
                Er_new = 0.5 * np.sum(Vx**2 + Vy**2)
                energy_r.append(Er_new)
            
            r0, v0, t0 = np.copy(R), np.copy(V), T
            
        r0_red, v0_red = np.copy(R), np.copy(V)
        n_events = nb_collisions_pairs + nb_collisions_walls
        
        print('Nb events: ', n_events)
        print('Nb wall collisions: ', nb_collisions_walls)
        print('Nb pair collisions: ', nb_collisions_pairs, '\t', 'Nb collisions par particules: ', nb_collisions_pairs/N_r)
        print('')
        
        
        #blue box
        x_length = 1    
        y_length = 1        
        
        #Ensemble average: generate random initial position configurations
        r0 = generate_positions(N_b, x_length, y_length)
        #Random speed for given temperature (i.e., given energy)
        v0 = generate_speeds(N_b, temperature_b)
        
        t0 = 0
        nb_collisions_walls = 0
        nb_collisions_pairs = 0
                
        while nb_collisions_pairs < nb_collisions_square:
            R, V, T, bool_wall, bool_pair = event_disks_square(np.copy(r0), np.copy(v0), t0, N_b, x_length, y_length)
            nb_collisions_walls += bool_wall
            nb_collisions_pairs += bool_pair
                
            if bool_pair == 1:
                Vx, Vy = V
                V_blue_moy = np.mean(np.sqrt(Vx**2 + Vy**2))
                v_b_moy.append(V_blue_moy)
                Eb_new = 0.5 * np.sum(Vx**2 + Vy**2)
                energy_b.append(Eb_new)
            
            r0, v0, t0 = np.copy(R), np.copy(V), T
            
        r0_blue, v0_blue = np.copy(R), np.copy(V)
        n_events = nb_collisions_pairs + nb_collisions_walls
        
        print('Nb events: ', n_events)
        print('Nb wall collisions: ', nb_collisions_walls)
        print('Nb pair collisions: ', nb_collisions_pairs, '\t', 'Nb collisions par particules: ', nb_collisions_pairs/N_b)
        print('')
        
        
        #rectangle box
        x_length = 2    
        y_length = 1
                
        N = N_r + N_b
        
        r0 = np.zeros((2, N))
        v0 = np.zeros((2, N))
        
        for uu in range(N):
            if uu % 2 == 0:
                #particules d'indice pair à gauche
                ind_pair = int(uu/2)
                Ax = r0_red[0][ind_pair]
                Ay = r0_red[1][ind_pair]
                Bx = v0_red[0][ind_pair]
                By = v0_red[1][ind_pair]
                
            else:
                #particules d'indice impair à droite
                ind_impair = int((uu-1)/2)
                Ax = r0_blue[0][ind_impair] + 1 
                Ay = r0_blue[1][ind_impair]
                Bx = v0_blue[0][ind_impair]
                By = v0_blue[1][ind_impair]
            
            coord_particleA = np.array([Ax, Ay])
            r0[:, uu] = coord_particleA
            vitesse_particleA = np.array([Bx, By])
            v0[:, uu] = vitesse_particleA
            
        
        t0 = 0
        nb_collisions_walls = 0
        nb_collisions_pairs = 0
        
        
        Er = 0.0
        
        while nb_collisions_pairs < nb_collisions_rect:
            R, V, T, bool_wall, bool_pair = event_disks_rect(np.copy(r0), np.copy(v0), t0, N, x_length, y_length)
            
            if bool_pair == 1:
                Vx, Vy = V
                Vx_pair = np.array([Vx[kk] for kk in range(len(Vx)) if kk%2==0])
                Vx_impair = np.array([Vx[kk] for kk in range(len(Vx)) if kk%2==1])
                Vy_pair = np.array([Vy[kk] for kk in range(len(Vx)) if kk%2==0])
                Vy_impair = np.array([Vy[kk] for kk in range(len(Vx)) if kk%2==1])
                V_red_moy = np.mean(np.sqrt(Vx_pair**2 + Vy_pair**2))
                V_blue_moy = np.mean(np.sqrt(Vx_impair**2 + Vy_impair**2))
                v_r_moy.append(V_red_moy)
                v_b_moy.append(V_blue_moy)
                
                Er_new = 0.5 * np.sum(Vx_pair**2 + Vy_pair**2)
                Eb_new = 0.5 * np.sum(Vx_impair**2 + Vy_impair**2)
                
                if np.round(Er_new, 7) != np.round(Er, 7):
                    energy_r.append(Er_new)
                    energy_b.append(Eb_new)
                    Er = Er_new
            
            
            nb_collisions_walls += bool_wall
            nb_collisions_pairs += bool_pair
            
            if nb_collisions_pairs % 20 == 0:
                print(t0, nb_collisions_pairs)
            
            r0, v0, t0 = np.copy(R), np.copy(V), T
        
        v_r_moy = np.array(v_r_moy)
        v_b_moy = np.array(v_b_moy)
        energy_r = np.array(energy_r)
        energy_b = np.array(energy_b)
        
        n_events = nb_collisions_pairs + nb_collisions_walls
        
        print('Nb events: ', n_events)
        print('Nb wall collisions: ', nb_collisions_walls)
        print('Nb pair collisions: ', nb_collisions_pairs, '\t', 'Nb collisions par particules: ', nb_collisions_pairs/N)
        print('')
        
        
        to_be_sent = [v_r_moy, v_b_moy, energy_r, energy_b]

        output_qloc.put(to_be_sent)
            
    return 1


if __name__ == "__main__":
    manager = multipro.Manager()
    output_q = manager.Queue()    
    
    pool = multipro.Pool(nb_cores)
    pool.apply_async(listener_core, (output_q,))
    
    jobs = []
    for i in range(nb_cores-1):
        job = pool.apply_async(worker_core, (segment_index[i], output_q))
        jobs.append(job)
    
    for job in jobs: 
        job.get()







