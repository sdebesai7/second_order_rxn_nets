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
import gc
import os
import sys
import argparse
#from memory_profiler import profile

def profile(rxn, params, initial_conditions, all_features, solver, stepsize_controller, t_points, dt, max_steps):
    E, B, F=params
    solns=[]
    for F_a in all_features:
        sol_F_a=rxn.integrate(solver=solver, stepsize_controller=stepsize_controller, t_points=t_points, dt0=dt, initial_conditions=initial_conditions, args=(E, B, F, F_a,), max_steps=max_steps) 
        solns.append(sol_F_a.ys[-1].copy())
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
    prominence_peaks, _ = scipy.signal.find_peaks(data, prominence=min_prominence)
    prominence_troughs, _ = scipy.signal.find_peaks(-data, prominence=min_prominence)
    
    # Find peaks based on width criterion
    width_peaks, _ = scipy.signal.find_peaks(data, width=min_width)
    width_troughs, _ = scipy.signal.find_peaks(-data, width=min_width)

    # Combine unique peaks and troughs from both criteria
    all_peaks = np.unique(np.concatenate((prominence_peaks, width_peaks)))
    all_troughs = np.unique(np.concatenate((prominence_troughs, width_troughs)))

    bad_points = ~np.isfinite(data)
    
    # Expand the mask to include neighbors of bad points
    bad_points_expanded = bad_points.copy()
    
    # Mark neighbors of bad points as bad too
    for i in range(len(bad_points)):
        if bad_points[i]:
            # Mark left neighbor as bad
            if i > 0:
                bad_points_expanded[i-1] = True
            # Mark right neighbor as bad  
            if i < len(bad_points) - 1:
                bad_points_expanded[i+1] = True
    
    # FILTER PEAKS AND TROUGHS
    # Remove peaks that are at bad points or next to bad points
    valid_peaks = all_peaks[~bad_points_expanded[all_peaks]]
    valid_troughs = all_troughs[~bad_points_expanded[all_troughs]]

    finite_peaks = np.isfinite(data[valid_peaks])  # boolean array, True for finite values
    count_finite_peaks = np.sum(finite_peaks)

    # Count finite values in all_troughs
    finite_troughs = np.isfinite(data[valid_troughs])
    count_finite_troughs = np.sum(finite_troughs)

    #    Total finite values in both arrays
    total_finite = count_finite_peaks + count_finite_troughs

    return total_finite, prominence_peaks, prominence_troughs, width_peaks, width_troughs

def gen_profiles(fname, n, m, seeds, n_second_order, n_inputs, second_order_edge_idxs, initial_conditions, all_features, solver, stepsize_controller, t_points, dt,max_steps, output_file, turning_points_file):
    with open(fname, "rb") as f:
        params_rand = pkl.load(f)
    f.close()

    n_profiles=5#len(params_rand)
    #jax.debug.print('num profs: {x}', x=n_profiles)
    #dist_tps=jnp.zeros(n_profiles * n)
    #solns_all=jnp.zeros((n_profiles, all_features.shape[0], n))
    counter=0

    with open(output_file, 'w') as profile_file, open(turning_points_file, 'w') as tp_file:
        for i, params in enumerate(params_rand[0:n_profiles]):
            if i%50 == 0:
                #jax.debug.print('seed {i}', i=int(seeds[i]))
                print(f'seed {i}')
        
            rxn=random_rxn_net(n, m, int(seeds[i]), n_second_order, n_inputs, test=False, A=None, second_order_edge_idxs=second_order_edge_idxs, F_a_idxs=None)
            solns=profile(rxn, params, initial_conditions, all_features, solver, stepsize_controller, t_points, dt, max_steps)
            
            profile_data_flat = solns.flatten()  # Shape: (n_features * n_species,)
            profile_line = f"{i}," + ",".join(map(str, profile_data_flat)) + "\n"
            profile_file.write(profile_line)
            
            #solns_all = solns_all.at[i].set(solns.copy())
            turning_points_for_profile = []
            for j, species_prof in enumerate(solns.T):
                n_tps=count_turning_points(species_prof)
                turning_points_for_profile.append(n_tps) 
                #dist_tps=dist_tps.at[counter].set(n_tps)
                counter+=1

            tp_line = f"{i}," + ",".join(map(str, turning_points_for_profile)) + "\n"
            tp_file.write(tp_line)
    print('integrated all profiles for this set of edges')

def main():
# Parse command line arguments
    parser = argparse.ArgumentParser(description='Run turning points analysis with different parameters')
    parser.add_argument('--n_second_order_edges', type=str, default="[[0,0]]",
                       help='Second order edges as 2D list string, e.g., "[[0,1],[4,1]]"')
    #parser.add_argument('--n_profiles', type=int, default=6, help='Number of profiles to process')
    #parser.add_argument('--chunk_size', type=int, default=20, help='Chunk size for processing')
    parser.add_argument('--task_id', type=int, default=0, help='Task ID for job array')
    #parser.add_argument('--output_suffix', type=str, default='', help='Suffix for output files')

    args = parser.parse_args()

    try:
        import ast
        edges_list = ast.literal_eval(args.n_second_order_edges)
        second_order_edges = np.array(edges_list)

        # Calculate n_second_order from the edges (for compatibility)
        if len(edges_list) == 1 and edges_list[0] == [0, 0]:
            n_second_order = 0  # Special case for dummy edge
        else:
            n_second_order = len(edges_list)

    except (ValueError, SyntaxError) as e:
        print(f"Error parsing n_second_order_edges: {e}")
        print(f"Input was: {args.n_second_order_edges}")
        sys.exit(1)

    n = 6
    m = n * (n - 1) // 2
    seeds = np.arange(1, 4001)
    seeds = seeds.reshape(4, 1000)
    initial_conditions = np.log(np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]))
    all_features = np.linspace(-20, 20, 100)
    t_points = np.linspace(0, 20, 200)
    solver = 'LSODA'

    dt = 0.1

    max_steps = 1000#0000

    n_inputs = 1

    params_file=f'data/turning_points/params_all_N0'
    print(f'n_second_order: {n_second_order}')
    print('second order edges: {second_order_edges}')
    seed_set = seeds[min(args.task_id, len(seeds)-1)]

    suffix =f"task{args.task_id}"
    dist_filename = f'data/turning_points/N{n}_M{m}_S{n_second_order}_distributions_{suffix}.txt'
    profiles_filename = f'data/turning_points/N{n}_M{m}_S{n_second_order}_profiles_{suffix}.txt'

    gen_profiles(params_file, n, m, seed_set, n_second_order, n_inputs, second_order_edges, initial_conditions, all_features, solver, t_points, dt, profiles_filename, dist_filename)

    # Save results
    #with open(dist_filename, 'wb') as f:
    #    pkl.dump(dist, f)

    #if solns_all is not None:
    #    with open(profiles_filename, 'wb') as f:
    #        pkl.dump(solns_all, f)


if __name__ == "__main__":
    main()




