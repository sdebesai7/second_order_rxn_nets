#code to define reaction networks
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt
import optax  
import networkx as nx

#a general reaction net object. 
#different methods corresponding to different types of reaction nets
#integrate function numerically solves given parameters

jax.config.update("jax_enable_x64", True)
class random_rxn_net:
    def __init__(self, n, m, seed, n_second_order, n_inputs, test=False, A=None, second_order_edge_idxs=None, F_a_idxs=None):

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

        #setting up inputs and second order edges
        if test:
            self.second_order_edges = jnp.array([[0, 1]]) #second order reactions
            self.second_order_edge_reactants = jnp.array([[0, 2]])
            self.second_order_edge_prods = jnp.array([[1, 2]]) #jnp.array([[1, 1]]) 
        else:
            #edges for inputs
            idxs = np.random.choice(len(self.F_edge_idxs), size=n_inputs, replace=False)
            self.F_a_idxs = jnp.array(self.F_edge_idxs[idxs])
            if self.n_second_order>0:
                idxs = np.random.choice(len(self.edge_idxs), size=n_second_order, replace=False)
                self.second_order_edges = jnp.array(self.edge_idxs[idxs])

                self.second_order_edge_prods = jnp.zeros((self.n_second_order, 2), dtype=int)
                self.second_order_edge_reactants = jnp.zeros((self.n_second_order, 2), dtype=int)
                
                for i, edge in enumerate(self.second_order_edges):
                    #map to some indices
                    key = jax.random.PRNGKey(0)
                    key, subkey1, subkey2 = jax.random.split(key, 3)
                    reactant2=jax.random.randint(key=subkey1, shape=(1,), minval=0, maxval=self.n-1, dtype=int)[0]
                    key, subkey=jax.random.split(key)
                    prod2=jax.random.randint(key=subkey2, shape=(1,), minval=0, maxval=self.n-1, dtype=int)[0]

                    #randomly select the second species in the second order reaction 
                    reactants=jnp.array([edge[0], reactant2])
                    prods=jnp.array([edge[1], prod2])
                   
                    self.second_order_edge_reactants=self.second_order_edge_reactants.at[i].set(reactants.copy())
                    self.second_order_edge_prods=self.second_order_edge_prods.at[i].set(prods.copy())
    
    #right now this does NOT assume that the second order edges are bidirectional
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

        #set second order edges to 0
        W_first_order=W.copy()
        
        if self.n_second_order > 0:
            cols, rows=self.second_order_edges.T
            #jax.debug.print('rows: {x}', x=rows)
            #jax.debug.print('cols: {x}', x=cols)
            W_first_order=W_first_order.at[rows, cols].set(0)

        #jax.debug.print('W_first_order: {x}', x=W_first_order)

        dydt = jnp.zeros(self.n)
        for i in range(self.n):  
            #jax.debug.print('i: {x}', x=i)
            #jax.debug.print('first order contribution: {x}', x=dydt[i] + W_first_order[i] @ y - jnp.sum(W_first_order[:, i] * y[i]))
            #contributions from first order 
            dydt=dydt.at[i].set(dydt[i] + W_first_order[i] @ y - jnp.sum(W_first_order[:, i] * y[i]))
            if self.n_second_order > 0:
                #second order reactions
                #jax.debug.print('before second order: {x}', x=dydt)
                #jax.debug.print('computing second order contributions')
                for e, edge in enumerate(self.second_order_edges):
                    idx_i=edge[0]
                    idx_f=edge[1]
                
                    c=0
                
                    #there is a term for each reactant in the reaction if the reactant is the same as the species being considered 

                    reactants=self.second_order_edge_reactants[e]
                    products=self.second_order_edge_prods[e]
  
                    def process_reactant(inputs):
                            j, dydt = inputs
                            term = 1
                        
                            def i_eq_j_branch():
                                return -1
                            def i_not_eq_j_branch():
                                return 0 
                        
                            c=jax.lax.cond(i==j, i_eq_j_branch, i_not_eq_j_branch)
                            
                            for k in reactants:
                                term=term*y[k] #multiply by the concentration of the other species in the reaction 
                            dydt=dydt.at[i].set(dydt[i] + c*term*W[idx_f, idx_i])

                            return dydt
                    #skip over chaperones
                    def skip_reactant(inputs):
                        j, dydt=inputs
                        return dydt

                    def process_product(inputs):
                            j, dydt = inputs
                            term = 1
                            #c = 1 if i==j else 0
                            def i_eq_j_branch():
                                return 1
                            def i_not_eq_j_branch():
                                return 0 
                        
                            c=jax.lax.cond(i==j, i_eq_j_branch, i_not_eq_j_branch)

                            for k in reactants:
                                term=term*y[k] #multiply by the concentration of the other species in the reaction
                            
                            dydt=dydt.at[i].set(dydt[i] + c*term*W[idx_f, idx_i])

                            return dydt

                    def skip_product(inputs):
                        j, dydt=inputs
                        return dydt
             
                    for j in reactants:
                        skip=jnp.where(products == j, 1, 0).sum() > 0
                        dydt=jax.lax.cond(skip, skip_reactant, process_reactant, (j, dydt))
                    
                    #there is term for each product in the reaction if the reactant is the same as the species being considered
                    for j in products:   
               
                        #jax.debug.print('j: {x}', x=j)    
                        #if jnp.sum((jnp.isin(j,reactants)).astype(jnp.int32)) > 0:
                        skip=jnp.where(reactants == j, 1, 0).sum() > 0
                        dydt=jax.lax.cond(skip, skip_product, process_product, (j, dydt))

                    #jax.debug.print('updated dydt: {x}', x=dydt)
            #jax.debug.print('dydt:{x}',x=dydt)
        #jax.debug.print('dlog(y)dt:{x}\n',x=dydt/y)
                
        return dydt / y
    
    def integrate(self, solver, stepsize_controller, t_points, dt0, initial_conditions, args, max_steps):
        def wrapped_dynamics(t, y, args):
                return self.rxn_net_dynamics(t, y, args)
        term=ODETerm(wrapped_dynamics)

        solution = diffeqsolve(term, solver=solver, stepsize_controller=stepsize_controller, t0=t_points[0], t1=t_points[-1], dt0=dt0, y0=initial_conditions, args=args, saveat=SaveAt(ts=t_points), max_steps=max_steps)
        #solution = diffeqsolve(term, solver=solver, t0=t_points[0], t1=t_points[-1], dt0=dt0, y0=initial_conditions, args=args, saveat=SaveAt(ts=t_points), max_steps=max_steps)
        return solution


