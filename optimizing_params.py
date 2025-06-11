import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController, Kvaerno3
import optax  
import pickle as pkl
from reaction_nets import rxn_net, random_rxn_net
from functools import partial
import scipy.optimize
import os
import equinox as eqx
from jax import make_jaxpr
import initialize_nets

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

# Function to compute loss between labels and computed concentrations using kld loss
# inputs:
#   rxn (reaction_network object): reaction network
#   params (jndarray(float64)): parameters to optimize
#   solver (diffrax solver): solver to use for integration
#   stepsize_controller (diffrax stepsize controller): stepsize controller to use for integration
#   dt0 (float): initial time step size
#   max_steps (int): maximum number of steps to take in the integration
#   t_points (jndarray(float64)): time points to evaluate the solution at
#   feature (jndarray(float64)): feature vector to append to params
#   label (jndarray(float64)): target concentration profile to compare against
#   initial_conditions (jndarray(float64)): initial concentrations
# outputs:
#   loss (float): KLD loss between the predicted and target concentration profiles
@partial(jax.jit, static_argnames=['rxn', 'solver', 'stepsize_controller', 'dt0', 'max_steps'])
def kld_loss(params, rxn, solver, stepsize_controller, dt0, max_steps, t_points, feature, label, initial_conditions):
    all_params=jnp.append(params, feature)
    solution = rxn.integrate(solver=solver, stepsize_controller=stepsize_controller, t_points=t_points, dt0=dt0, initial_conditions=initial_conditions, args=all_params, max_steps=max_steps)
    loss = optax.losses.kl_divergence_with_log_targets(solution.ys[-1], jnp.log(label))
    return loss

# Function to generate concentration profile as a function of F_a for a single reaction network with given parameters using kld loss
# needs to be debugged 
@partial(jax.jit, static_argnames=['rxn'])
def cross_entropy_loss(params, rxn, t_points, feature, label, initial_conditions):
    #solve w/ current params + params that are fixed by the training example
    all_params=jnp.append(params, feature)
    solution = rxn.integrate(solver=Tsit5(), stepsize_controller=PIDController(0.5, 0.1), t_points=t_points, dt0=0.001, initial_conditions=initial_conditions, args=all_params, max_steps=1000000)
    y_pred_conc = jnp.exp(solution.ys[-1]) #compute loss based on equilibrated param solution (exponentiated)
    y_pred_logits = jax.scipy.special.logit(y_pred_conc)  

    #compute the loss 
    loss = optax.softmax_cross_entropy(logits=y_pred_logits,labels=label)
    return loss

# Function to optimize parameters of a reaction network using autodiff
# batch training has not been tested
# inputs:
#   rxn (reaction_network object): reaction network
#   online_training (bool): whether to use online training or batch training
#   initial_params (jndarray(float64)): initial parameters to optimize
#   t_points (jndarray(float64)): time points to evaluate the solution at
#   y_features (jndarray(float64)): feature vectors for each training example
#   y_labels (jndarray(float64)): target concentration profiles for each training example
#   initial_conditions (jndarray(float64)): initial concentrations
#   solver (diffrax solver): solver to use for integration
#   stepsize_controller (diffrax stepsize controller): stepsize controller to use for integration
#   dt0 (float): initial time step size
#   max_steps (int): maximum number of steps to take in the integration
#   learning_rate (float): learning rate for the optimizer
#   num_epochs (int): number of epochs to train for
#   batch_size (int): batch size for batch training
def optimize_ode_params(rxn, online_training, initial_params, t_points, y_features, y_labels, initial_conditions, solver, stepsize_controller, dt0, max_steps, learning_rate=0.01, num_epochs=10, batch_size=32):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_params)
    optimized_params_history = []
    params = initial_params.copy()
  
    loss_history = []

    n_samples = len(y_features)
    n_batches = (n_samples + batch_size - 1) // batch_size

    indices = jnp.arange(n_samples)

    total_iterations = 0
    grads_per_epoch_autodiff={}
    #grads_per_epoch_fde={}
    avg_epoch_losses=[]

    print('training')
    for epoch in range(num_epochs):
        key = jax.random.PRNGKey(epoch)
        #shuffle indices of training data 
        indices = jax.random.permutation(key, indices)
        
        epoch_loss = 0

        #tracking gradients
        grads_in_epoch_autodiff = []
        #grads_in_epoch_fde = []

        if online_training:
            for idx in indices:
                total_iterations += 1
                optimized_params_history.append(params.copy())
                
                feature = y_features[idx]
                label = y_labels[idx]
                
                loss_fxn = lambda p: kld_loss(p, rxn, solver, stepsize_controller, dt0, max_steps, t_points, feature, label, initial_conditions) #cross_entropy_loss(p, rxn, t_points, feature, label, initial_conditions)
                #grads_fde = scipy.optimize.approx_fprime(params, kld_loss, 1.4901161193847656e-08, rxn, solver, stepsize_controller, dt0, max_steps, t_points, feature, label, initial_conditions)
                #comptute loss and gradients using autodiff
                loss_value, grads_autodiff = jax.value_and_grad(loss_fxn)(params)
                
                epoch_loss += loss_value.item()
                
                grads_in_epoch_autodiff.append(grads_autodiff.copy())
                #grads_in_epoch_fde.append(grads_fde.copy())
                #update parameters
                updates, opt_state = optimizer.update(grads_autodiff, opt_state)
                params = optax.apply_updates(params, updates)
                params = jnp.maximum(params, 1e-5)  #no neg params
                #print(f'params after optimizing: {params} \n')
                
                if total_iterations % 10 == 0:
                    loss_history.append(loss_value.item())
               
            #track gradients using autodiff and fde
            #grads_per_epoch_fde[epoch]=grads_in_epoch_fde
            grads_per_epoch_autodiff[epoch]=grads_in_epoch_autodiff

            print('epoch complete')
        else:
            for batch_idx in range(n_batches):
                total_iterations += 1
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
            
                batch_features = y_features[batch_indices]
                batch_labels = y_labels[batch_indices]
            
                batch_loss = 0
                batch_grads = None
            
                for feature, label in zip(batch_features, batch_labels):
                    #loss + gradients
                    loss_fxn = lambda p: cross_entropy_loss(p, rxn, t_points, feature, label, initial_conditions)
                    loss_value, sample_grads = jax.value_and_grad(loss_fxn)(params)
                    batch_loss += loss_value.item()
                
                    if batch_grads is None:
                        batch_grads = sample_grads
                    else:
                        batch_grads = jax.tree_map(lambda a, b: a + b, batch_grads, sample_grads)
            
                #avg
                batch_size_actual = end_idx - start_idx
                batch_grads = jax.tree_map(lambda g: g / batch_size_actual, batch_grads)
            
                #update
                updates, opt_state = optimizer.update(batch_grads, opt_state)
                params = optax.apply_updates(params, updates)
                params = jnp.maximum(params, 1e-5)  # set param to 0 if it goes neg
            
                epoch_loss += batch_loss
            
                #avg_batch_loss
                if total_iterations % 10 == 0:
                    batch_avg_loss = batch_loss / batch_size_actual
                    loss_history.append(batch_avg_loss)
        
        avg_epoch_loss = epoch_loss / n_samples
        avg_epoch_losses.append(avg_epoch_loss)
    return params, avg_epoch_losses, loss_history, grads_per_epoch_autodiff, optimized_params_history#, grads_per_epoch_fde

def save_data(all_data, filename):
    os.makedirs('data', exist_ok=True)
    file = open(f'{filename}', 'wb')
    pkl.dump(all_data, file)
    file.close()

if __name__ == "__main__":
    #can make command line args 
    n=6
    m=n*(n-1)//2
    seed=0
    n_second_order=1
    n_inputs=1
    batch_size = 32  
    online_training=True
    num_epochs =  6 #20 
    t_points = jnp.linspace(0.0, 10.0, 200) 
    solver=Tsit5()
    stepsize_controller=PIDController(0.005, 0.01)
    dt0=0.001
    max_steps=10000

    init_data_file='data/init_data/triangle_b_double_non_monotonic'
    out_data_file='data/train/triangle_b_double_non_monotonic_training_data'
    #read in training and network info
    file = open(init_data_file, 'rb')
    init_data_dict=pkl.load(file)
    file.close()

    net_type, initial_params, t_points, train_features, train_labels, initial_conditions, true_params, training_data_type, train_features, train_labels=init_data_dict['network_type'], init_data_dict['initial_params'], init_data_dict['t_points'], init_data_dict['train_features'], init_data_dict['train_labels'], init_data_dict['initial_conditions'], init_data_dict['true_params'], init_data_dict['training_data_type'], init_data_dict['train_features'], init_data_dict['train_labels']
    #create reaction network 
    print(net_type)
    print(f'init concentrations: {np.exp(initial_conditions)}')
    print(f'init params: {initial_params}')
    print(f'true params: {true_params}')
    #rxn=rxn_net(net_type)

    
    rxn=random_rxn_net(n, m, seed, n_second_order, n_inputs)

    #optimize
    optimized_params, avg_epoch_losses, loss_history, grads_per_epoch_autodiff, optimized_params_history = optimize_ode_params(rxn, online_training, initial_params, t_points, train_features, train_labels, initial_conditions, solver, stepsize_controller, dt0, max_steps, learning_rate=0.01, num_epochs=num_epochs,batch_size=batch_size)

    #save 
    all_data={'train_features':train_features, 'train_labels': train_labels,'grads_per_epoch_autodiff':grads_per_epoch_autodiff, 'optimized_params':optimized_params, 'loss_history':loss_history, 'optimized_params_history':optimized_params_history}
    save_data(all_data, out_data_file)
    
 
    
