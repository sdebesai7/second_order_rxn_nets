#make this generalize better
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController, Kvaerno3
import optax  
import pickle as pkl
from reaction_nets import rxn_net
from functools import partial
import scipy.optimize
import os
import equinox as eqx
from jax import make_jaxpr

jax.config.update("jax_enable_x64", True)

#generalize this set up better? 
def initialize_rxn_net(network_type):
    rxn=rxn_net(network_type)
    #choose reaction network
    if network_type =='goldbeter_koshland':
        #initial conditions: parameters to learn and system initial conditions. 
        #2D feature are concatenated to form a full set of parameters for the ODE
        initial_params = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        # Initial concentrations of each species
        initial_conditions = jnp.array([1, 0.0, 0.0, 0.0, 1, 1])
        true_params=None
    elif network_type == 'triangle_a':
        initial_params=jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        initial_conditions = jnp.log(jnp.array([0.3, 0.4, 0.3]))
        true_params=jnp.array([np.log(1), 0, 0, np.log(10), 0, 0, np.log(1), 0, 0, np.log(0.1), 0, 0, np.log(0.05), 0, 0, np.log(4), 0, 0])
    elif network_type == 'triangle_b':
        initial_params=jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        initial_conditions = jnp.log(jnp.array([0.3, 0.4, 0.3]))
        true_params=jnp.array([np.log(1), 0, 0, np.log(10), 0, 0, np.log(1), 0, 0, np.log(0.1), 0, 0, np.log(0.05), 0, 0, np.log(4), 0, 0])
    elif network_type == 'triangle_c':
        initial_params=jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        initial_conditions = jnp.log(jnp.array([0.3, 0.4, 0.3]))
        true_params=jnp.array([np.log(1), 0, 0, np.log(10), 0, 0, np.log(1), 0, 0, np.log(0.1), 0, 0, np.log(0.05), 0, 0, np.log(4), 0, 0])
    
    return rxn, initial_params, initial_conditions, true_params

def profile(rxn, initial_params, initial_conditions, all_features, solver=Tsit5(), stepsize_controller=PIDController(0.005, 0.01), t_points=jnp.linspace(0.0, 10.0, 100), dt0=0.001, max_steps=10000):
    solns=[]
    for feature in all_features:
        all_params=jnp.append(initial_params, feature)
        solution = rxn.integrate(solver=solver, stepsize_controller=stepsize_controller, t_points=t_points, dt0=dt0, initial_conditions=initial_conditions, args=all_params, max_steps=max_steps)
        solns.append(solution.ys[-1].copy())
    return jnp.array(solns)

def gen_training_data(rxn, type, n_samples, true_params, initial_conditions, solver, stepsize_controller, t_points, dt0, max_steps):
    if type == 'simple_monotonic':
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        #sample driving force uniformly
        all_features=jax.random.uniform(subkey, (n_samples), minval=-5, maxval=10) 

        #compute associated labels
        all_labels=profile(rxn, true_params, initial_conditions, all_features, solver, stepsize_controller, t_points, dt0, max_steps)
        all_labels=jnp.exp(all_labels)
        
        '''
        a=(1 + jnp.tanh(0.4*(all_features - 7)))*0.4 + 0.1
        c=(1 + jnp.tanh(-0.4*(all_features - 3)))*0.45
        b=1-a-c
        all_labels=jnp.array([a, b, c]).T
        '''
        #print(f'all_features: {all_features}')

    elif type == '4 gaussians':
        samples_per_gaussian = n_samples // 4  # Integer division

        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)

        #features: 4 2d gaussians, 1 centered in each quadrant
        driving_dist_1 = jax.random.multivariate_normal(subkey, jnp.array([-10, -10]), 3 * jnp.eye(2), shape=(samples_per_gaussian,))
        key, subkey = jax.random.split(key)
        driving_dist_2 = jax.random.multivariate_normal(subkey, jnp.array([10, -10]), 3 * jnp.eye(2), shape=(samples_per_gaussian,))
        key, subkey = jax.random.split(key)
        driving_dist_3 = jax.random.multivariate_normal(subkey, jnp.array([-10, 10]), 3 * jnp.eye(2), shape=(samples_per_gaussian,))
        key, subkey = jax.random.split(key)
        driving_dist_4 = jax.random.multivariate_normal(subkey, jnp.array([10, 10]), 3 * jnp.eye(2), shape=(samples_per_gaussian,))

        #labels
        #W, WE1, W_star, W_star_E2, E1, E2
        label_dist_1 = jnp.zeros((samples_per_gaussian, 6))
        label_dist_2 = jnp.zeros((samples_per_gaussian, 6))
        label_dist_3 = jnp.zeros((samples_per_gaussian, 6))
        label_dist_4 = jnp.zeros((samples_per_gaussian, 6))

        #[0.5, 0, 0, 0, 0.25, 0.25]
        label_dist_1 = label_dist_1.at[:, 0].set(0.5)
        label_dist_1 = label_dist_1.at[:, 4].set(0.25)
        label_dist_1 = label_dist_1.at[:, 5].set(0.25)

        #[0, 0, 0.5, 0, 0.25, 0.25]
        label_dist_2 = label_dist_2.at[:, 2].set(0.5)
        label_dist_2 = label_dist_2.at[:, 4].set(0.25)
        label_dist_2 = label_dist_2.at[:, 5].set(0.25)

        #[0, 0.5, 0, 0, 0.5, 0]
        label_dist_3 = label_dist_3.at[:, 1].set(0.5)
        label_dist_3 = label_dist_3.at[:, 4].set(0.5)

        #[0.25, 0, 0.25, 0, 0.25, 0.25]
        label_dist_4 = label_dist_4.at[:, 0].set(0.25)
        label_dist_4 = label_dist_4.at[:, 2].set(0.25)
        label_dist_4 = label_dist_4.at[:, 4].set(0.25)
        label_dist_4 = label_dist_4.at[:, 5].set(0.25)

        #concatenate + randomize
        all_features = jnp.concatenate([driving_dist_1, driving_dist_2, driving_dist_3, driving_dist_4], axis=0)
        all_labels = jnp.concatenate([label_dist_1, label_dist_2, label_dist_3, label_dist_4], axis=0)
    

    #check that features all sum to 1:
    #print(jnp.sum(all_labels, axis=1))
    '''
    if jnp.all(jnp.sum(all_labels, axis=1) != 1.0):
        raise Exception(f'probabilities in labels do not all sum to 1: \n {jnp.sum(all_labels, axis=1)}')
    if not jnp.all(all_labels > 0):
        raise Exception(f'probabilities are not all greater than 0: \n {all_labels < 0}  \n {all_labels}')
    '''

    key, subkey = jax.random.split(key)
    indices = jax.random.permutation(subkey, n_samples)
    shuffled_features = all_features[indices]
    shuffled_labels = all_labels[indices]

    training_features = jnp.array(shuffled_features)
    training_labels = jnp.array(shuffled_labels)

    print(f"Features shape: {training_features.shape}")
    print(f"Labels shape: {training_labels.shape}")

    #train / test split
    train_size = int(0.8 * n_samples)
    train_features = training_features[:train_size]
    train_labels = training_labels[:train_size]
    val_features = training_features[train_size:]
    val_labels = training_labels[train_size:]

    return train_features, train_labels, val_features, val_labels


