#testing reaction network code
import numpy as np
import pickle as pkl
#from reaction_nets import rxn_net
from functools import partial
import random
#from modified_reaction_nets import random_rxn_net
from reaction_nets_numpy import random_rxn_net
import scipy
from scipy.signal import savgol_filter



def profile(rxn, params, initial_conditions, all_features, solver, t_points, dt):
    E, B, F=params
    solns=[]
    for F_a in all_features:
        sol_F_a=rxn.integrate(solver=solver, t_points=t_points, initial_conditions=initial_conditions, args=(E, B, F, F_a,), dt0=dt) 
        solns.append(sol_F_a.y.T[-1].copy())
        #jax.debug.print('{x}', x=sol_F_a.result)
    return np.exp(np.array(solns))

def count_turning_points(data,window_length=11, polyorder=2, min_width=5):
    
    if len(data) < window_length:
        window_length = max(min(len(data) - 2, 7), 3)
    window_length = window_length if window_length % 2 == 1 else window_length - 1
    polyorder = min(polyorder, window_length - 2)
    
    # Apply Savitzky-Golay filter to smooth the data
    data = savgol_filter(data, window_length, polyorder)
    min_prominence=0.1*np.max(data)
    
    # Find peaks based on prominence criterion
    prominence_peaks, _ = scipy.signal.find_peaks(data, plateau_size=1, prominence=min_prominence)
    prominence_troughs, _ = scipy.signal.find_peaks(-data, plateau_size=1, prominence=min_prominence)
    
    # Find peaks based on width criterion
    width_peaks, _ = scipy.signal.find_peaks(data, plateau_size=1, width=min_width)
    width_troughs, _ = scipy.signal.find_peaks(-data, plateau_size=1, width=min_width)

    # Combine unique peaks and troughs from both criteria
    all_peaks = np.unique(np.concatenate((prominence_peaks, width_peaks)))
    all_troughs = np.unique(np.concatenate((prominence_troughs, width_troughs)))

    return all_peaks.shape[0] + all_troughs.shape[0]

def gen_profiles(fname, n, m, seeds, n_second_order, n_inputs, second_order_edge_idxs, initial_conditions, all_features, solver, stepsize_controller, t_points, dt,max_steps):
    with open(fname, "rb") as f:
        params_rand = pkl.load(f)
    f.close()

    n_profiles=6#len(params_rand)
    dist_tps=np.zeros(n_profiles * n)
    solns_all=np.zeros((n_profiles, all_features.shape[0], n))
    counter=0
    for i, params in enumerate(params_rand[0:n_profiles]):
        #gen reaction net
        #seed=int(seeds[i])
        print(f'seed {i}')
        rxn=random_rxn_net(n, m, int(seeds[i]), n_second_order, n_inputs, test=False, A=None, second_order_edge_idxs=second_order_edge_idxs, F_a_idxs=None)
        solns=profile(rxn, params, initial_conditions, all_features, solver, stepsize_controller, t_points, dt, max_steps)
        solns_all = solns_all.at[i].set(solns.copy())

        for j, species_prof in enumerate(solns.T):
            n_tps=count_turning_points(species_prof)
            dist_tps=dist_tps.at[counter].set(n_tps)
            counter+=1

    return dist_tps, solns_all

def main():
    n=6
    m=n*(n-1)//2
    seeds=np.arange(1, 4001)
    seeds = seeds.reshape(4, 1000)
    initial_conditions=np.log(np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]))
    all_features=np.linspace(-20, 20, 100)
    t_points=np.linspace(0, 20, 200)
    solver= 'LSODA'
    dt=0.1
    
    second_order_edge_idxs=[np.array([[0, 0]]), np.array([[0, 1]]), np.array([[0, 1], [4, 1]]), np.array([[0, 1], [4, 1], [2, 3]]), np.array([[0, 1], [4, 1], [2, 3], [4, 5]])]
    n_second_order_all=np.array([0,1,2,3])

    n_inputs=1

    params_file=f'data/turning_points/params_all_N0'

    rel=0
    for i, (n_second_order, second_order_edge) in enumerate(zip(n_second_order_all, second_order_edge_idxs)):
        print(f'n_second_order: {n_second_order}')
        print(f'second order edges: {second_order_edge}')
        dist, solns_all=gen_profiles(params_file, n, m, seeds[i], n_second_order, n_inputs,second_order_edge, initial_conditions, all_features, solver, t_points, dt)

        #save result
        f=open(f'data/turning_points/dist_N{n}_M{m}_S{n_second_order}_distributions', 'wb')
        pkl.dump(dist, f)
        f.close()

        #save result
        f=open(f'data/turning_points/dist_N{n}_M{m}_S{n_second_order}_profiles', 'wb')
        pkl.dump(solns_all, f)
        f.close()
        rel+=1

if __name__ == "__main__":
    main()



