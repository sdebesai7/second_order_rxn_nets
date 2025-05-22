#code to define reaction networks
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax  
import networkx as nx
import pickle as pkl
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, Kvaerno3, PIDController
import optax  

#a general reaction net object. 
#different methods corresponding to different types of reaction nets
#integrate function numerically solves given parameters

jax.config.update("jax_enable_x64", True)

class random_rxn_net:
    def __init__(self, n, m, seed, n_second_order, n_inputs, test=False, A=None, second_order_edge_idxs=None, F_a_idxs=None):
        key = jax.random.PRNGKey(seed)
        if test:
            self.A=A #adjacency matrix for triangle topology
            self.second_order_edge_idxs=second_order_edge_idxs#jnp.array([[0, 1]]) #list of second order edges
            self.F_a_idxs=F_a_idxs#jnp.array([[2, 1]]) #C to B
        
        else:
            self.n=n #number of possible species / nodes 
            self.G = nx.gnm_random_graph(n, m, seed) #look at uniform distribution over graphs with given set of nodes and edges 
            self.A=jnp.array(nx.adjacency_matrix(self.G).toarray()) #adjacency matrix 
        self.n=n
        self.edge_idxs=jnp.argwhere(self.A != 0)
        self.num_edges=self.edge_idxs.shape[0]
        self.n_second_order=n_second_order

        #get idxs for all non-zero edges in the upper triangle of the adjacency matrix
        #these are the edges that can have an anti-symmetric non-eq driving force
        #they are also the edges that second order reactions can occur along 
        upper_triangle_mask = jnp.triu(jnp.ones_like(self.A, dtype=bool), k=1)
        nonzero_mask = (self.A != 0) & upper_triangle_mask
        self.F_edge_idxs = jnp.array(jnp.nonzero(nonzero_mask)).T  

        nonzero_mask = (self.A != 0) #& upper_triangle_mask
        self.F_a_edge_idxs = jnp.array(jnp.nonzero(nonzero_mask)).T 

        #setting up inputs and second order edges
        if test:
            self.second_order_edges = jnp.array([[0, 1]]) #second order reactions
            self.second_order_edge_reactants = jnp.array([[0, 2]])
            self.second_order_edge_prods = jnp.array([[1, 2]]) #jnp.array([[1, 1]]) 
        else:
            #edges for inputs
            key, subkey = jax.random.split(key)
            idxs = jax.random.choice(subkey, jnp.arange(len(self.F_a_edge_idxs)), shape=(n_inputs, ), replace=False)
            self.F_a_idxs = jnp.array(self.F_a_edge_idxs[idxs])
            if self.n_second_order>0:
                if second_order_edge_idxs == None:
                    key, subkey = jax.random.split(key)
                    idxs = jax.random.choice(subkey, jnp.arange(len(self.edge_idxs)), shape=(n_second_order, ), replace=False)
                    self.second_order_edges = jnp.array(self.edge_idxs[idxs])
                else:
                    self.second_order_edges=second_order_edge_idxs

                self.second_order_edge_prods = jnp.zeros((self.n_second_order, 2), dtype=int)
                self.second_order_edge_reactants = jnp.zeros((self.n_second_order, 2), dtype=int)
                
                for i, edge in enumerate(self.second_order_edges):
                    #map to some indices
                    #will always generate the same second order edges right now
                    
                    key, subkey1, subkey2 = jax.random.split(key, 3)
                    reactant2=jax.random.randint(key=subkey1, shape=(1,), minval=0, maxval=self.n-1, dtype=int)[0]
                    prod2=jax.random.randint(key=subkey2, shape=(1,), minval=0, maxval=self.n-1, dtype=int)[0]

                    #randomly select the second species in the second order reaction 
                    reactants=jnp.array([edge[0], reactant2])
                    prods=jnp.array([edge[1], prod2])

                    self.second_order_edge_reactants=self.second_order_edge_reactants.at[i].set(reactants.copy())
                    self.second_order_edge_prods=self.second_order_edge_prods.at[i].set(prods.copy())
            else:
                self.second_order_edges=jnp.array([[0, 0]])
                self.second_order_edge_reactants=jnp.array([[0, 0]])
                self.second_order_edge_prods=jnp.array([[0, 0]])
    #right now this does NOT assume that the second order edges are bidirectional
    @partial(jax.jit, static_argnames=['self'])
    def rxn_net_dynamics(self, t, y, params):
        #jax.debug.print('log concentrations: {x}', x=y)
        y=jnp.exp(y)
        #jax.debug.print('concentrations: {x}', x=y)
        E, B, F, F_a_in=params

        B_params=jnp.zeros((self.n, self.n))
        F_params=jnp.zeros((self.n, self.n))
        F_a_params=jnp.zeros((self.n, self.n))

        #reshape params to create B matrix
        B_params=B_params.at[self.edge_idxs[:, 0], self.edge_idxs[:, 1]].set(B)
        
        #reshape params to create F matrix 
        rows, cols = self.F_edge_idxs.T
        # Then use them as separate indices
        F_params = F_params.at[rows, cols].set(F)
        #jax.debug.print('F_params: {x}', x=F_params)
        F_params = F_params - F_params.T

        #reshape params to create F_a matrix
        rows, cols = self.F_a_idxs.T

        F_a_params=F_a_params.at[rows, cols].set(F_a_in)
        F_a_params = F_a_params - F_a_params.T

        #jax.debug.print('B_params: {x}', x=B_params)
        #jax.debug.print('F_params: {x}', x=F_params)
        #jax.debug.print('F_a_params: {x}', x=F_a_params)
        #jax.debug.print('E_params: {x}', x=E)

        #reshape arrays accordingly to create rates matrix 
        W=jnp.exp(E - B_params + 0.5*F_params+0.5*F_a_params)
        rows, cols=self.edge_idxs.T
        mask = jnp.zeros((self.n, self.n), dtype=bool)
        mask = mask.at[rows, cols].set(True)
        W=W * mask
        
        #compute W first order 
        def compute_W_first_order(inputs):
            second_order_edges, W_first_order=inputs
            #jax.debug.print('computing W first order')
            #jax.debug.print('edges: {x}', x=second_order_edges)
            cols, rows=second_order_edges.T
            #jax.debug.print('rows: {x}', x=rows)
            #jax.debug.print('cols: {x}', x=cols)
            W_first_order=W_first_order.at[rows, cols].set(0)

            return W_first_order
        
        def skip_W_first_order(inputs):
            second_order_edges, W_first_order=inputs
            #jax.debug.print('skipping W first order')
            #jax.debug.print('edges: {x}', x=second_order_edges)
            return W_first_order
        
        #set second order edges to 0
        W_first_order=W.copy()
        comp_W_cond=self.n_second_order > 0 
        W_first_order=jax.lax.cond(comp_W_cond, compute_W_first_order, skip_W_first_order, (self.second_order_edges, W_first_order))

        #jax.debug.print('W_first_order: {x}', x=W_first_order)
        #jax.debug.print('W: {x}', x=W_first_order)
        
        #functions to process second order reactions
        def process_reactant(inputs):
            i, j, reactants, idx_i, idx_f, dydt = inputs
            term = 1
                        
            def i_eq_j_branch():
                return -1
            def i_not_eq_j_branch():
                return 0 
                        
            c=jax.lax.cond(i==j, i_eq_j_branch, i_not_eq_j_branch)
                            
            for k in reactants:
                term=term*y[k] #multiply by the concentration of the other species in the reaction 
            dydt=dydt.at[i].set(dydt[i] + c*term*W[idx_f, idx_i])
            #jax.debug.print('second order prod contribution: {x}*{y}*{z}={u}', x=c, y=term, z=W[idx_f, idx_i], u=c*term*W[idx_f, idx_i])
            return dydt
            
        #skip over chaperones
        def skip_reactant(inputs):
            i, j, reactants, idx_i, idx_f, dydt=inputs
            return dydt
            
        def process_product(inputs):
            i, j, reactants, idx_i, idx_f, dydt = inputs
            term = 1
                            
            def i_eq_j_branch():
                return 1
            def i_not_eq_j_branch():
                return 0 
                        
            c=jax.lax.cond(i==j, i_eq_j_branch, i_not_eq_j_branch)

            for k in reactants:
                term=term*y[k] #multiply by the concentration of the other species in the reaction
                            
            dydt=dydt.at[i].set(dydt[i] + c*term*W[idx_f, idx_i])

            #jax.debug.print('second order reactant contribution: {x}*{y}*{z}={u}', x=c, y=term, z=W[idx_f, idx_i], u=c*term*W[idx_f, idx_i])

            return dydt
            
        def skip_product(inputs):
            i, j, reactants, idx_i, idx_f, dydt=inputs
            return dydt
            
        def second_order_rxns(inputs):
            i, second_order_edges, second_order_edge_reactants, second_order_edge_prods, dydt=inputs
            #second order reactions
            #jax.debug.print('before second order: {x}', x=dydt)
            #jax.debug.print('computing second order contributions')
            for e, edge in enumerate(second_order_edges):
                idx_i=edge[0]
                idx_f=edge[1]
              
                #there is a term for each reactant in the reaction if the reactant is the same as the species being considered 

                reactants=second_order_edge_reactants[e]
                products=second_order_edge_prods[e]

                #jax.debug.print('second order reactants {x}:', x=reactants)
                #jax.debug.print('second order prods {x}:', x=products)
                
                #iterate through reactants 
                for j in reactants:
                    skip=jnp.where(products == j, 1, 0).sum() > 0
                    #if the reactant is also a product then it's a chaperone, in which case we skip it
                    #otherwise, we compute a term 
                    dydt=jax.lax.cond(skip, skip_reactant, process_reactant, (i, j, reactants, idx_i, idx_f, dydt))
                    
                #there is term for each product in the reaction if the reactant is the same as the species being considered
                for j in products:   
                    skip=jnp.where(reactants == j, 1, 0).sum() > 0
                    dydt=jax.lax.cond(skip, skip_product, process_product, (i, j, reactants, idx_i, idx_f, dydt))
                #jax.debug.print('updated dydt: {x}', x=dydt)
                #jax.debug.print('dydt:{x}',x=dydt)
            return dydt
            
        def skip_second_order(inputs):
            i, second_order_edges, second_order_edge_reactants, second_order_edge_prods, dydt=inputs
            return dydt
        
        #iterate through each species and compute dydt
        dydt = jnp.zeros(self.n)
        for i in range(self.n):  
            #jax.debug.print('i: {x}', x=i)
            #jax.debug.print('positive first order: {x}', x=W_first_order[i] @ y)
            #jax.debug.print('negative first order: {x}', x=jnp.sum(W_first_order[:, i] * y[i]))
            #jax.debug.print('first order contribution: {x}', x=dydt[i] + W_first_order[i] @ y - jnp.sum(W_first_order[:, i] * y[i]))

            #contributions from first order 
            dydt=dydt.at[i].set(dydt[i] + W_first_order[i] @ y - jnp.sum(W_first_order[:, i] * y[i])) #first order reactions
            process_second_order=self.n_second_order > 0
            dydt=jax.lax.cond(process_second_order, second_order_rxns, skip_second_order, (i, self.second_order_edges, self.second_order_edge_reactants, self.second_order_edge_prods, dydt)) #second order reactions

            #jax.debug.print('dy[i]dt:{x}',x=dydt)      
        return dydt / y
    
    @partial(jax.jit, static_argnames=['self', 'solver', 'stepsize_controller', 'dt0', 'max_steps'])
    def integrate(self, solver, stepsize_controller, t_points, dt0, initial_conditions, args, max_steps):
        def wrapped_dynamics(t, y, args):
                return self.rxn_net_dynamics(t, y, args)
        term=ODETerm(wrapped_dynamics)

        solution = diffeqsolve(term, solver=solver, stepsize_controller=stepsize_controller, t0=t_points[0], t1=t_points[-1], dt0=dt0, y0=initial_conditions, args=args, saveat=SaveAt(ts=t_points), throw=False, max_steps=max_steps)
        #solution = diffeqsolve(term, solver=solver, t0=t_points[0], t1=t_points[-1], dt0=dt0, y0=initial_conditions, args=args, saveat=SaveAt(ts=t_points), max_steps=max_steps)
        return solution

