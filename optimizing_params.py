import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt
import optax  
import pickle as pkl

jax.config.update("jax_enable_x64", True)

#rxn ode
def rxn(t, y, args):
    a1, a2, d1, d2, k1, k2 = args
    W, WE1, W_star, W_star_E2, E1, E2 = y
  
    dW_dt = -a1*W*E1 + d1*WE1 + k2*W_star_E2
    dWE1_dt = a1*W*E1 - (d1+k1)*WE1
    dW_star_dt = -a2*W_star*E2 + d2*W_star_E2 + k1*WE1
    dW_star_E2_dt = a2*W_star*E2 - (d2+k2)*W_star_E2
    dE1_dt = -dWE1_dt
    dE2_dt = -dW_star_E2_dt
    
    return jnp.array([dW_dt, dWE1_dt, dW_star_dt, dW_star_E2_dt, dE1_dt, dE2_dt])


@jax.jit
def cross_entropy_loss(params, t_points, y_data, initial_conditions, volume=1.0):
    #solve w/ current params
    solution = diffeqsolve(ODETerm(rxn), Tsit5(), t0=t_points[0], t1=t_points[-1], dt0=0.01, y0=initial_conditions, args=params, saveat=SaveAt(ts=t_points), max_steps=10000)
    y_pred_conc = solution.ys
    
    #convert to counts
    y_pred_counts = y_pred_conc * volume
    y_data_counts = y_data * volume
    
    #compute loss average across time
    losses = []
    for i in range(len(t_points)):
        y_pred_probs = y_pred_counts[i] / (jnp.sum(y_pred_counts[i]) + 1e-10)
        y_pred_probs = jnp.clip(y_pred_probs, 1e-10, 1.0 - 1e-10) #guarantee btwn 1 and 0
        y_pred_logits = jax.scipy.special.logit(y_pred_probs)
        y_data_probs = y_data_counts[i] / (jnp.sum(y_data_counts[i]) + 1e-10)
        
        loss = optax.softmax_cross_entropy(logits=y_pred_logits,labels=y_data_probs)
        losses.append(loss)
    
    return jnp.mean(jnp.array(losses))

def optimize_ode_params(initial_params, t_points, y_data, initial_conditions, learning_rate=0.01, num_iterations=500, volume=1.0):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_params)
    params = initial_params
    loss_history = []
    
    #optimize 
    for i in range(num_iterations):
        #compute loss
        loss_value = cross_entropy_loss(params, t_points, y_data, initial_conditions, volume)
        loss_history.append(loss_value.item())
        
        #compute gradient
        loss_with_volume = lambda p: cross_entropy_loss(p, t_points, y_data, initial_conditions, volume)
        grads = jax.grad(loss_with_volume)(params)
        
        #update params
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.maximum(params, 1e-5) #set param to 0 if it goes neg
        
        # Print progress
        if i % 50 == 0 or i == num_iterations - 1:
            print(f"Iteration {i}, Loss: {loss_value}, Params: {params}")
    
    return params, loss_value, loss_history

# Example usage
if __name__ == "__main__":
    #create training by picking a set of params and sampling from a multinomial defined by the concentrations given at equilibrium for this parameter set
    true_params = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    initial_conditions = jnp.array([50.0, 0.0, 0.0, 0.0, 50.0, 50.0])
    t_points = jnp.linspace(0.0, 10.0, 100)
    volume = 1.0
    
    solution = diffeqsolve(ODETerm(rxn),Tsit5(),t0=t_points[0],t1=t_points[-1],dt0=0.01,y0=initial_conditions,args=true_params,saveat=SaveAt(ts=t_points),max_steps=10000)
    y_data_conc = solution.ys
    
    key = jax.random.PRNGKey(0)
    y_data_noisy = []
    
    #conservation laws
    substrate_count = initial_conditions[0] + initial_conditions[1] + initial_conditions[2] + initial_conditions[3] #W + W* + WE1 + W*E2 so W + W* total

    E1_count = initial_conditions[2] + initial_conditions[4]
    E2_count = initial_conditions[3] + initial_conditions[5]
    
    for i in range(len(t_points)):
        sol_i = y_data_conc[i]
        W, WE1, W_star, W_star_E2, E1, E2 = sol_i
        
        #conserved quantities (in concentrations)
        total_substrate = W + W_star + WE1 + W_star_E2
        total_E1 = E1 + WE1
        total_E2 = E2 + W_star_E2

        print(f'i={i}, W + W* tot: {total_substrate}, E1_T: {total_E1}, E2_T: {total_E2}')
        
        #compute ratios
        ratio_W_to_Wstar = W / (W + W_star)  
        ratio_E1_bound = WE1 / (total_E1)    
        ratio_E2_bound = W_star_E2 / (total_E2) 
        
        key, subkey = jax.random.split(key)
        
        #sample ratio of W to W1
        free_substrate_count = substrate_count - int(substrate_count * (ratio_E1_bound + ratio_E2_bound))
        W_vs_Wstar_prob = jnp.array([ratio_W_to_Wstar, 1 - ratio_W_to_Wstar])
        
        #multinomial sampling function was removed due to errors 
        free_substrate_split = jax.random.categorical(subkey, jnp.log(W_vs_Wstar_prob), shape=(int(free_substrate_count),))
        W_free_count = jnp.sum(free_substrate_split == 0)
        W_star_free_count = jnp.sum(free_substrate_split == 1)
        
        #calc WE1 and W*E2
        WE1_count = int(E1_count * ratio_E1_bound)
        W_star_E2_count = int(E2_count * ratio_E2_bound)
        
        #calc E1 and E2 (free)
        E1_free_count = E1_count - WE1_count
        E2_free_count = E2_count - W_star_E2_count
        
        #calc W and W* counts 
        W_total_count = W_free_count + WE1_count
        W_star_total_count = W_star_free_count + W_star_E2_count

        #convert to concentration
        scaled_noisy_data = jnp.array([W_free_count / substrate_count, WE1_count / substrate_count, W_star_free_count / substrate_count, W_star_E2_count / substrate_count, E1_free_count / E1_count, E2_free_count / E2_count])
        y_data_noisy.append(scaled_noisy_data)
    
    y_data_noisy = jnp.array(y_data_noisy)
    
    initial_params = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
   
    optimized_params, final_loss, loss_history = optimize_ode_params(initial_params, t_points, y_data_noisy, initial_conditions,learning_rate=0.01, num_iterations=300, volume=volume)
    
    print("\nOptimization Results:")
    print(f"True Parameters: {true_params}")
    print(f"Initial Parameters: {initial_params}")
    print(f"Optimized Parameters: {optimized_params}")
    print(f"Final Loss: {final_loss}")

    #run simulation with optimized params
    optimized_solution = diffeqsolve(ODETerm(rxn),Tsit5(),t0=t_points[0],t1=t_points[-1],dt0=0.01,y0=initial_conditions,args=optimized_params,saveat=SaveAt(ts=t_points),max_steps=10000)
    
    #dump outputs
    file = open('data/true_params', 'wb')
    pkl.dump(true_params, file)
    file.close()

    file = open('data/opt_params', 'wb')
    pkl.dump(optimized_params, file)
    file.close()

    file = open('data/loss_history', 'wb')
    pkl.dump(loss_history, file)
    file.close()

    file=open('data/opt_data_ys', 'wb')
    pkl.dump(optimized_solution.ys, file)
    file.close()

    file=open('data/true_data_ys', 'wb')
    pkl.dump(solution.ys, file)
    file.close()

    file=open('data/true_data_ts', 'wb')
    pkl.dump(solution.ts, file)
    file.close()

    file=open('data/opt_data_ts', 'wb')
    pkl.dump(optimized_solution.ts, file)
    file.close()