import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt
import optax  
import pickle as pkl
from reaction_nets import rxn_net
from functools import partial

jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnames=['rxn'])
def cross_entropy_loss(params, rxn, t_points, feature, label, initial_conditions):
    #solve w/ current params + params that are fixed by the training example

    all_params=jnp.append(params, feature)
    #jax.debug.print('params passed into solver: {all_params}', all_params=all_params)

    #solution = diffeqsolve(term, Tsit5(), t0=t_points[0], t1=t_points[-1], dt0=0.01, y0=initial_conditions, args=all_params, saveat=SaveAt(ts=t_points), max_steps=10000)
    solution = rxn.integrate(Tsit5(), t_points, dt0=0.01, initial_conditions=initial_conditions, args=all_params, max_steps=10000)
    
    y_pred_conc = solution.ys[-1] #compute loss based on equilibrated param solution
    #jax.debug.print('solution to diff-eq w/ params: {y_pred_conc}', y_pred_conc=y_pred_conc)

    #convert to counts
    #y_pred_counts = y_pred_conc * volume
    
    #compute the loss 
    y_pred_probs = y_pred_conc/(jnp.sum(y_pred_conc) + 1e-10) #y_pred_counts / (jnp.sum(y_pred_counts) + 1e-10)
    y_pred_probs = jnp.clip(y_pred_probs, 1e-10, 1.0 - 1e-10) #guarantee btwn 1 and 0
    y_pred_logits = jax.scipy.special.logit(y_pred_probs)

    #jax.debug.print('logits for features: {y_pred_logits}', y_pred_logits=y_pred_logits)
        
    loss = optax.softmax_cross_entropy(logits=y_pred_logits,labels=label)
    
    return loss

def optimize_ode_params(rxn, online_training, initial_params, t_points, y_features, y_labels, initial_conditions, learning_rate=0.01, num_epochs=10, batch_size=32):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_params)
    params = initial_params.copy()
    loss_history = []

    n_samples = len(y_features)
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

    indices = jnp.arange(n_samples)

    total_iterations = 0

    for epoch in range(num_epochs):
        key = jax.random.PRNGKey(epoch)
        indices = jax.random.permutation(key, indices)
        
        epoch_loss = 0
        
        if online_training:
            for idx in indices:
                total_iterations += 1
                
                feature = y_features[idx]
                label = y_labels[idx]
                print(f'params before optimizing: {params}')
                #loss + grad
                loss_fxn = lambda p: cross_entropy_loss(p, rxn, t_points, feature, label, initial_conditions)
                loss_value, grads = jax.value_and_grad(loss_fxn)(params)
                print(f'loss value: {loss_value}')
                print(f'grads: {grads}')
                epoch_loss += loss_value.item()
                
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                params = jnp.maximum(params, 1e-5)  #no neg params
                print(f'params after optimizing: {params} \n')
                
                if total_iterations % 10 == 0:
                    loss_history.append(loss_value.item())
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
    
    return params, avg_epoch_loss, loss_history

#make this generalize better
def gen_training_data(type, n_samples):
    if type == 'simple_monotonic':
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        #sample driving force uniformly
        all_features=jax.random.uniform(subkey, (n_samples), minval=-20, maxval=20) 

        #compute associated labels
        b=(1 + jnp.tanh(-0.4*all_features - 2))*0.4 + 0.1
        a=(1 + jnp.tanh(0.6*all_features + 2))*0.45
        c=1-a-b

        all_labels=jnp.array([a, b, c]).T
    elif type == 'simple_non_monotonic':
        #features
        all_features=jax.random.uniform(subkey, (n_samples), minval=-20, maxval=20) 

        #labels
        a=0.1*jnp.ones(n_samples) #constant
        b=(1 + jnp.tanh(-0.4*all_features - 2))*0.45
        c=(1 + jnp.tanh(0.4*all_features + 2))*0.45

        all_labels=jnp.array([a, b, c]).T

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
    if jnp.all(jnp.sum(all_labels, axis=1) != 1.0):
        raise Exception(f'probabilities in labels do not all sum to 1: \n {jnp.sum(all_labels, axis=1)}')

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

def test(rxn, optimized_params, val_features, t_points, initial_conditions):
    pred_labels = []
    final_states = []
    
    for feature in val_features:
        all_params = jnp.append(optimized_params, feature)
        #solution = diffeqsolve(ODETerm(reaction_nets.goldbeter_kohsland), Tsit5(), t0=t_points[0], t1=t_points[-1], dt0=0.01, y0=initial_conditions, args=all_params, saveat=SaveAt(ts=t_points), max_steps=10000)
        solution=rxn.integrate(Tsit5(), t_points, dt0=0.01, initial_conditions=initial_conditions, args=all_params, max_steps=10000)
        final_state = solution.ys[-1]
        final_state_probs = final_state / jnp.sum(final_state + 0.01)
        
        pred_labels.append(final_state_probs)
        final_states.append(solution.ys)
        
    return jnp.array(pred_labels), final_states

def model_accuracy(val_labels, pred_labels, threshold=0.01):
    correct = 0
    total = len(val_labels)
    
    for true_label, pred_label in zip(val_labels, pred_labels):
        if jnp.all(jnp.abs(true_label - pred_label) < threshold):
            correct += 1
    
    accuracy = correct / total
    return accuracy

def save_data(train_features, train_labels, val_features, val_labels, optimized_params, loss_history):
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    file = open('data/train_features', 'wb')
    pkl.dump(train_features, file)
    file.close()

    file = open('data/train_labels', 'wb')
    pkl.dump(train_labels, file)
    file.close()

    file = open('data/val_features', 'wb')
    pkl.dump(val_features, file)
    file.close()

    file = open('data/val_labels', 'wb')
    pkl.dump(val_labels, file)
    file.close()

    file = open('data/opt_params', 'wb')
    pkl.dump(optimized_params, file)
    file.close()

    file = open('data/loss_history', 'wb')
    pkl.dump(loss_history, file)
    file.close()

#generalize this set up better? 
def initialize_rxn_net(network_type):
    t_points = jnp.linspace(0.0, 10.0, 100)
    batch_size = 32  
    num_epochs = 1 #20  
    online_training=True
   
    rxn=rxn_net(network_type)
    #choose reaction network
    if network_type =='goldbeter_koshland':
        #initial conditions: parameters to learn and system initial conditions. 
        #2D feature are concatenated to form a full set of parameters for the ODE
        initial_params = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        # Initial concentrations of each species
        initial_conditions = jnp.array([1, 0.0, 0.0, 0.0, 1, 1])
    elif network_type == 'triangle_a':
        initial_params=jnp.array([0.5, 0.5, 0.5])
        initial_conditions = jnp.array([1, 2, 0.5])
    elif network_type == 'triangle_b':
        initial_params=jnp.array([0.5, 0.5, 0.5])
        initial_conditions = jnp.array([1, 2, 0.5])
    elif network_type == 'triangle_c':
        initial_params=jnp.array([0.5, 0.5, 0.5])
        initial_conditions = jnp.array([1, 2, 0.5])
    
    return rxn, initial_params, initial_conditions, t_points, batch_size, num_epochs, online_training 

if __name__ == "__main__":
    #generate training data + labels
    network_type='triangle_b'#'goldbeter_koshland'
    rxn, initial_params, initial_conditions, t_points, batch_size, num_epochs, online_training=initialize_rxn_net(network_type)

    n_samples = 1000
    type='simple_monotonic'#'4 gaussians'
    train_features, train_labels, val_features, val_labels = gen_training_data(type, n_samples)

    #optimize
    optimized_params, final_loss, loss_history = optimize_ode_params(rxn, online_training, initial_params, t_points, train_features, train_labels, initial_conditions,learning_rate=0.01, num_epochs=num_epochs,batch_size=batch_size)

    #test + save 
    pred_labels, final_states = test(rxn, optimized_params, val_features, t_points, initial_conditions)
    accuracy = model_accuracy(val_labels, pred_labels)
    save_data(train_features, train_labels, val_features, val_labels, optimized_params, loss_history)
    print(f"Model accuracy: {accuracy:.4f}")