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
import initialize_nets
import optimizing_params

jax.config.update("jax_enable_x64", True)

def test(rxn, optimized_params, val_features, t_points, initial_conditions, solver, stepsize_controller, dt0, max_steps):
    pred_labels = []
    final_states = []
    
    for feature in val_features:
        all_params = jnp.append(optimized_params, feature)
        solution=rxn.integrate(solver=solver, stepsize_controller=stepsize_controller, t_points=t_points, dt0=dt0, initial_conditions=initial_conditions, args=all_params, max_steps=max_steps)
        final_state = jnp.exp(solution.ys[-1])
        
        pred_labels.append(final_state)
        final_states.append(jnp.exp(solution.ys))
        
    return jnp.array(pred_labels), final_states

def model_accuracy(val_labels, pred_labels, threshold=0.01):
    correct = 0
    total = len(val_labels)
    
    for true_label, pred_label in zip(val_labels, pred_labels):
        if jnp.all(jnp.abs(true_label - pred_label) < threshold):
            correct += 1
    
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    #test + save 
    t_points = jnp.linspace(0.0, 10.0, 200) 
    solver=Tsit5()
    stepsize_controller=PIDController(0.005, 0.01)
    dt0=0.001
    max_steps=10000

    init_data_file='data/init_data/triangle_b_double_non_monotonic'
    #read in training and network info
    file = open(init_data_file, 'rb')
    init_data_dict=pkl.load(file)
    file.close()

    train_data_file='data/train/triangle_b_double_non_monotonic_training_data'
    
    file = open(train_data_file, 'rb')
    train_data_dict=pkl.load(file)
    file.close()

    val_features, val_labels, initial_conditions=init_data_dict['val_features'], init_data_dict['val_labels'], init_data_dict['initial_conditions']
    rxn=rxn_net(init_data_dict['network_type'])

    optimized_params =train_data_dict['optimized_params'] 
    jax.debug.print(f'optimized params: {optimized_params}')
    pred_labels, final_states = test(rxn, optimized_params, val_features, t_points, initial_conditions, solver, stepsize_controller, dt0, max_steps)
    accuracy = model_accuracy(val_labels, pred_labels)

    test_data={'val_features':val_features, 'val_labels':val_labels, 'pred_labels':pred_labels}

    test_data_file='data/test/triangle_b_double_non_monotonic_test_data'
    optimizing_params.save_data(test_data, test_data_file)

    print(f"Model accuracy: {accuracy:.4f}")


