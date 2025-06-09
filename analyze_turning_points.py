import jax
import jax.numpy as jnp
import numpy as np
import sys
import turning_points_inputs
jax.config.update("jax_enable_x64", True)

def load_turning_points_from_text(filename, n_profiles, n_species):
    """
    Load turning points from text file written by gen_profiles()
    
    Returns:
        turning_points: JAX array of shape (n_profiles, n_species)
    """
    turning_points = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Skip comment lines
            if line.startswith('#'):
                continue
                
            # Parse data line
            parts = line.strip().split(',')
            profile_id = int(parts[0])
            tp_data = [int(x) for x in parts[1:]]  # Turning points are integers
            
            turning_points.append(tp_data)
    
    return jnp.array(turning_points)

def load_profiles_from_text(filename, n_features, n_species):
    """
    Load profiles from text file written by gen_profiles()
    
    Returns:
        profiles: JAX array of shape (n_profiles, n_features, n_species)
    """
    profiles = []
    seeds = []
    with open(filename, 'r') as f:
        for line in f:
            print(line)
            # Skip comment lines
            if line.startswith('#'):
                continue
                
            # Parse data line
            parts = line.strip().split(',')
            profile_id = int(parts[0])
            seeds.append(profile_id)
            profile_data = [float(x) for x in parts[1:]]
            # Reshape from flat to (n_features, n_species)
            profile_array = jnp.array(profile_data).reshape(n_features, n_species)
            profiles.append(profile_array)
    
    return jnp.array(profiles), jnp.array(seeds)

def profiles_to_turning_points(seed, turning_points_file, solns):
    with open(turning_points_file, 'w') as tp_file, open(turning_points_file.strip().split('.')[0]+'_peak_types'+'.txt', 'w') as peaks_file:
        turning_points_for_profile = []
        peaks_file.write("seed,species,prominence_peaks,prominence_troughs,width_peaks,width_troughs\n")
        print(solns.T.shape)
        for j, species_prof in enumerate(jnp.array(solns).T):
            print(j)
            n_tps, prominence_peaks, prominence_troughs, width_peaks, width_troughs=turning_points_inputs.count_turning_points(species_prof)
            turning_points_for_profile.append(n_tps) 
            peaks_line = f"{seed},{j},{prominence_peaks},{prominence_troughs},{width_peaks},{width_troughs}\n"
            peaks_file.write(peaks_line)
        tp_line = f"{seed}," + ",".join(map(str, turning_points_for_profile)) + "\n"
        tp_file.write(tp_line)
    return turning_points_for_profile, seed 

def main():
    n=6
    m=int(n*(n-1)/2)
    n_second_order=0
    suffix='task0'
    profiles_filename = f'data/turning_points/N{n}_M{m}_S{n_second_order}_profiles_{suffix}.txt'
    turning_points_filename = f'data/turning_points/N{n}_M{m}_S{n_second_order}_distributions_{suffix}.txt'

    n_profiles=5#10000
    n_features=100
    n_species=6
    
    #turning_points=load_profiles_from_text(profiles_filename, n_profiles, n_features, n_species)
    profiles, seeds=load_turning_points_from_text(turning_points_filename, n_profiles, n_species)
    
    tp=[]
    for profile, seed in zip(profiles, seeds):
        tp.append(profiles_to_turning_points(profiles, profiles_filename, seeds))

    jax.debug.print('tps: {x}', x=(jnp.array(tp)).shape)
    #sys.stderr.flush() 
    #sys.stdout.flush()  # Force flush stdout
    jax.debug.print('profiles: {x}', x=profiles.shape)
    #sys.stdout.flush()  # Force flush stdout
    #sys.stderr.flush() 
if __name__ == "__main__":
    main()
