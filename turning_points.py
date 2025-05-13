#testing reaction network code
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, Kvaerno3, PIDController
import optax  
import pickle as pkl
#from reaction_nets import rxn_net
from functools import partial
import random
#from modified_reaction_nets import random_rxn_net
from reaction_nets import random_rxn_net
from rxn_nets_old import rxn_net
from flax import serialization

def profile(rxn, params, initial_conditions, all_features, solver, stepsize_controller, t_points, dt, max_steps):
    E, B, F=params
    solns=[]
    for F_a in all_features:
        sol_F_a=rxn.integrate(solver=solver, stepsize_controller=stepsize_controller, t_points=t_points, dt0=dt, initial_conditions=initial_conditions, args=(E, B, F, F_a,), max_steps=max_steps) 
        #print(jnp.sum(jnp.exp(sol_F_a.ys), axis=1))
        solns.append(sol_F_a.ys[-1].copy())
    return jnp.exp(jnp.array(solns))

def count_turning_points(arr: np.ndarray):
    # Assumes arr is a 1D NumPy array
    # Remove consecutive duplicates (flattened plateaus)
    diff_indices = np.where(np.diff(arr) != 0)[0] + 1
    unique_locs = np.concatenate(([0], diff_indices))

    arr_simple = arr[unique_locs]
    dx = np.diff(arr_simple)
    
    # Identify where the sign of the slope changes (i.e., turning points)
    turning_mask = dx[1:] * dx[:-1] < 0

    # Map back to original indices — use the first index of each plateau
    return np.count_nonzero(turning_mask)

def gen_rand_rxn_nets(seed, n_nets, n, m, n_second_order, n_inputs, test, A, second_order_edge_idxs, F_a_idxs):
    rxn_nets=[]
    for i in range(n_nets):
        rxn=random_rxn_net(n, m,seed,n_second_order, n_inputs, test, A, second_order_edge_idxs, F_a_idxs)
        rxn_nets.append(rxn)
   
    with open("rxn_nets.msgpack", "wb") as f:
        f.write(serialization.to_bytes(rxn_nets))

def gen_profiles(fname, params, all_features, solver, stepsize_controller, t_points, dt,max_steps):
    with open("rxn_nets.msgpack", "rb") as f:
        obj = serialization.from_bytes(rxn_nets, f.read())

    n_profiles=len(rxn_nets)
    dist_tps=np.zeros(n_profiles)
    for i, rxn in enumerate(rxn_nets):
        #gen reaction net
        
        solns=profile(rxn, params, initial_conditions, all_features, solver, stepsize_controller, t_points, dt, max_steps)
        n_tps=count_turning_points(solns.ys)
        dist_tps[i]=n_tps
