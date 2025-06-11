#code to define reaction networks
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle as pkl
import scipy
from scipy.integrate import solve_ivp

#a general reaction net object. 
#different methods corresponding to different types of reaction nets
#integrate function numerically solves given parameters

class random_rxn_net:
    def __init__(self, n, m, seed, n_second_order, n_inputs, test=False, A=None, second_order_edge_idxs=None, F_a_idxs=None):

        np.random.seed(seed)
       
        if test:
            self.A = A  # adjacency matrix for triangle topology
            self.second_order_edge_idxs = second_order_edge_idxs  # np.array([[0, 1]]) #list of second order edges
            self.F_a_idxs = F_a_idxs  # np.array([[2, 1]]) #C to B
            
        else:
            self.G = nx.gnm_random_graph(n, m, seed)  # look at uniform distribution over graphs with given set of nodes and edges
            self.A = np.array(nx.adjacency_matrix(self.G).toarray())  # adjacency matrix
        self.n = n
        self.edge_idxs = np.argwhere(self.A != 0)
        # self.num_edges = self.edge_idxs.shape[0]
        self.n_second_order = n_second_order
        
        # get idxs for all non-zero edges in the upper triangle of the adjacency matrix
        # these are the edges that can have an anti-symmetric non-eq driving force
        # they are also the edges that second order reactions can occur along
        upper_triangle_mask = np.triu(np.ones_like(self.A, dtype=bool), k=1)
        nonzero_mask = (self.A != 0) & upper_triangle_mask
        self.F_edge_idxs = np.array(np.nonzero(nonzero_mask)).T
        nonzero_mask = (self.A != 0)  # & upper_triangle_mask
        self.F_a_edge_idxs = np.array(np.nonzero(nonzero_mask)).T
        
        # setting up inputs and second order edges
        
        if test:
            self.second_order_edges = np.array([[0, 1]])  # second order reactions
            self.second_order_edge_reactants = np.array([[0, 2]])  # np.array([[1, 0]]) #
            self.second_order_edge_prods = np.array([[1, 2]]) #np.array([[1, 1]])  # np.array([[1, 2]]) #np.array([[0, 0]])
        else:
            # edges for inputs
            idxs = np.random.choice(np.arange(len(self.F_a_edge_idxs)), size=n_inputs, replace=False)
            self.F_a_idxs = np.array(self.F_a_edge_idxs[idxs])
        
            if self.n_second_order > 0:
                if second_order_edge_idxs is None:
                    idxs = np.random.choice(np.arange(len(self.edge_idxs)), size=n_second_order, replace=False)
                    self.second_order_edges = np.array(self.edge_idxs[idxs])
                else:
                    self.second_order_edges = second_order_edge_idxs
                
                self.second_order_edge_prods = np.zeros((self.n_second_order, 2), dtype=int)
                self.second_order_edge_reactants = np.zeros((self.n_second_order, 2), dtype=int)
            
                for i, edge in enumerate(self.second_order_edges):
                    # map to some indices
                    reactant2 = np.random.randint(0, self.n-1)
                    prod2 = np.random.randint(0, self.n-1)
                    # randomly select the second species in the second order reaction
                    reactants = np.array([edge[0], reactant2])
                    prods = np.array([edge[1], prod2])
                    self.second_order_edge_reactants[i] = reactants.copy()
                    self.second_order_edge_prods[i] = prods.copy()
            else:
                self.second_order_edges = np.array([[0, 0]])
                self.second_order_edge_reactants = np.array([[0, 0]])
                self.second_order_edge_prods = np.array([[0, 0]])
    

    def rxn_net_dynamics(self, t, y, params):
    
        # Convert from log concentrations to concentrations
        y = np.exp(y)
        E, B, F, F_a_in = params

        B_params = np.zeros((self.n, self.n))
        F_params = np.zeros((self.n, self.n))
        F_a_params = np.zeros((self.n, self.n))

        # Reshape params to create B matrix
        B_params[self.edge_idxs[:, 0], self.edge_idxs[:, 1]] = B
    
        # Reshape params to create F matrix 
        rows, cols = self.F_edge_idxs.T
        F_params[rows, cols] = F
        F_params = F_params - F_params.T

        # Reshape params to create F_a matrix
        rows, cols = self.F_a_idxs.T
        F_a_params[rows, cols] = F_a_in
        F_a_params = F_a_params - F_a_params.T

        # Reshape arrays accordingly to create rates matrix 
        W = np.exp(E.reshape(-1, 1) - B_params + 0.5*F_params + 0.5*F_a_params)
        rows, cols = self.edge_idxs.T
        mask = np.zeros((self.n, self.n), dtype=bool)
        mask[rows, cols] = True
        W = W * mask
    
        # Compute W first order 
        W_first_order = W.copy()
        if self.n_second_order > 0:
            # Set second order edges to 0
            cols, rows = self.second_order_edges.T
            W_first_order[rows, cols] = 0

        # Helper function to process reactants in second order reactions
        def process_reactant(i, j, products, reactants, idx_i, idx_f, dydt):
            term = 1
        
            # Check if it's not a chaperone
            reactant_count = np.sum(reactants == j)
            product_count = np.sum(products == j)
        
            # Determine multiplier based on chaperone status
            not_chaperone = ((reactant_count == 1) & (product_count == 2)) | \
                       (reactant_count == 2) | \
                       (~np.any(np.isin(reactants, products))) | \
                       ((reactant_count == 1) & (product_count == 0))
        
            c = -1 if not_chaperone else 0
                        
            for k in reactants:
                term = term * y[k]  # multiply by the concentration of the other species in the reaction 
            dydt[i] = dydt[i] + c * term * W[idx_f, idx_i]
      
            return dydt 
        
        def process_product(i, j, products, reactants, idx_i, idx_f, dydt):
            term = 1
           
            # Check if it's not a chaperone
            reactant_count = np.sum(reactants == j)
            product_count = np.sum(products == j)
        
            not_chaperone = ((reactant_count == 2) & (product_count == 1)) | \
                       (product_count == 2) | \
                       (~np.any(np.isin(reactants, products))) | \
                       ((product_count == 1) & (reactant_count == 0))
       
            c = 1 if not_chaperone else 0
        
            for k in reactants:
                term = term * y[k]  # multiply by the concentration of the other species in the reaction
                        
            dydt[i] = dydt[i] + c * term * W[idx_f, idx_i]
        
            return dydt

        def second_order_rxns(i, second_order_edges, second_order_edge_reactants, second_order_edge_prods, dydt):
            # Second order reactions
            for e, edge in enumerate(second_order_edges):
                idx_i = edge[0]
                idx_f = edge[1]
          
                reactants = second_order_edge_reactants[e]
                products = second_order_edge_prods[e]

                # Iterate through reactants 
                for j in reactants:
                    # If the species we are considering is a reactant in the second order term, we process it
                    if i == j:
                        dydt = process_reactant(i, j, products, reactants, idx_i, idx_f, dydt)
       
                # There is term for each product in the reaction if the reactant is the same as the species being considered
                for j in products:   
                    if i == j:
                        dydt = process_product(i, j, products, reactants, idx_i, idx_f, dydt)

            return dydt
    
        # Iterate through each species and compute dydt for that species
        dydt = np.zeros(self.n)
        for i in range(self.n):  
            # Contributions from first order for that species
            dydt[i] = dydt[i] + W_first_order[i] @ y - np.sum(W_first_order[:, i] * y[i])  # first order reactions
        
            # Get the second order contribution to that species dynamics
            if self.n_second_order > 0:
                dydt = second_order_rxns(i, self.second_order_edges, self.second_order_edge_reactants, self.second_order_edge_prods, dydt)  # second order reactions
          
        return dydt / y

    def integrate(self, solver, t_points, dt0, initial_conditions, args):

        def wrapped_dynamics(t, y):
            return self.rxn_net_dynamics(t, y, args)
    
        # Map solver names to scipy methods if needed
        # Common mappings might be:
        # 'Dopri5' -> 'RK45', 'Tsit5' -> 'DOP853', etc.
        # 'LSODA' -> 'LSODA' (automatic stiff/non-stiff switching)
        method_map = {
            'Dopri5': 'RK45',
            'Tsit5': 'DOP853', 
            'Euler': 'RK23',
            'Heun': 'RK23',
            'LSODA': 'LSODA'
        }
    
        # LSODA is particularly useful for reaction networks as it automatically
        # switches between stiff and non-stiff methods depending on the problem characteristics
    
        scipy_method = method_map.get(solver, solver)
    
        # Set up solver options
        solver_options = {}
    
        # Handle initial step size
        if dt0 is not None:
            solver_options['first_step'] = dt0
    
        # Set tolerances for adaptive step sizing
        solver_options['rtol'] = 1e-6
        solver_options['atol'] = 1e-9
    
        # Solve the ODE
        solution = solve_ivp(
            fun=wrapped_dynamics,
            t_span=(t_points[0], t_points[-1]),
            y0=initial_conditions,
            t_eval=t_points,
            method=scipy_method,
            dense_output=False,
            **solver_options
        )
    
        return solution