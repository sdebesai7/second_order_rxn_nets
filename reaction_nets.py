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
    def __init__(self, n, p, n_second_order, n_inputs, init_concentrations, test=False, A=None, second_order_edge_idxs=None, F_a_idxs=None):

        if test:
            self.A=A#jnp.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) #adjacency matrix for triangle topology
            self.second_order_edge_idxs=second_order_edge_idxs#jnp.array([[0, 1]]) #list of second order edges
            self.F_a_idxs=F_a_idxs#jnp.array([[2, 1]]) #C to B
        
        else:
            self.n=n #number of possible species / nodes 
            self.G = nx.erdos_renyi_graph(self.n, p)
            self.A=nx.adjacency_matrix(self.G) #adjacency matrix 
        
        self.y=init_concentrations
        self.n=n
        self.edge_idxs=jnp.argwhere(self.A != 0)
        self.num_edges=self.edge_idxs.shape[0]

        #get idxs for all non-zero edges in the upper triangle of the adjacency matrix
        #these are the edges that can have an anti-symmetric non-eq driving force
        #they are also the edges that second order reactions can occur along 
        upper_triangle_mask = jnp.triu(jnp.ones_like(self.A, dtype=bool), k=1)
        nonzero_mask = (self.A != 0) & upper_triangle_mask
        self.F_edge_idxs = jnp.array(jnp.nonzero(nonzero_mask)).T  

        #setting up inputs and second order edges
        if test:
            self.second_order_edges = jnp.array([[0, 1]]) #second order reactions
            self.second_order_edge_idxs = {}

            for edge in self.second_order_edges:
                #map to some indices
                #for triangle toplogy b 
                reactant2=2
                prod2=2

                #for triangle topology c
                #reactant2=2 
                #prod2=1

                #randomly select the second species in the second order reaction 
                if reactant2 >= 0:
                    reactants=jnp.array([edge[0], reactant2])
                else:
                    prods=jnp.array([edge[1]]) 
            
                if prod2 >= 0:
                    prods=jnp.array([edge[1], prod2])
                else:
                    prods=jnp.array([edge[1]]) 
                self.second_order_edge_idxs[tuple(edge.tolist())]= jnp.array([reactants, prods])
    
        else:
            #edges for inputs
            self.F_a_idxs= jnp.array(np.random.choice(self.F_edge_idxs,  size=n_inputs, replace=False)) #list of input edges
    
            self.second_order_edges = jnp.array(np.random.choice(self.edge_idxs, size=n_second_order))#jnp.array(np.random.choice(self.F_edge_idxs, size=n_second_order))#randomly select some fraction of the edges that can have second order reactions 

            self.second_order_edge_idxs = {}
            for edge in self.second_order_edges:
                #map to some indices
                reactant2=np.random.randint(-1, self.n)
                prod2=np.random.randint(-1, self.n)

                #randomly select the second species in the second order reaction 
                if reactant2 >= 0:
                    reactants=jnp.array([edge[0], reactant2])
                else:
                    prods=jnp.array([edge[1]]) 
            
                if prod2 >= 0:
                    prods=jnp.array([edge[1], prod2])
                else:
                    prods=jnp.array([edge[1]]) 
                self.second_order_edge_idxs[tuple(edge.tolist())]= jnp.array([reactants, prods])

        
    def construct_init_params(self):
        E=jnp.ones(self.n) * 0.5
        B=jnp.ones(self.num_edges) * 0.5
        F=jnp.ones(self.F_edge_idxs.shape[0]) * 0.5

        return E, B, F
    
    #right now this does NOT assume that the second order edges are bidirectional
    def rxn_net_dynamics(self, y, params):
        y=jnp.exp(y)
        jax.debug.print('initial concentrations: {x}', x=y)
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
        jax.debug.print('F_params: {x}', x=F_params)
        F_params = F_params - F_params.T

        #reshape params to create F_a matrix
        rows, cols = self.F_a_idxs.T

        F_a_params=F_a_params.at[rows, cols].set(F_a_in)
        F_a_params = F_a_params - F_a_params.T

        jax.debug.print('B_params: {x}', x=B_params)
        jax.debug.print('F_params: {x}', x=F_params)
        jax.debug.print('F_a_params: {x}', x=F_a_params)

        #reshape arrays accordingly to create rates matrix 
        W=jnp.exp(E + B_params + F_params+F_a_params)
        rows, cols=self.edge_idxs.T

        mask = jnp.zeros((self.n, self.n), dtype=bool)

        mask = mask.at[rows, cols].set(True)

        W=W * mask

        jax.debug.print('W: {x}', x=W)

        #set second order edges to 0
        W_first_order=W.copy()
        
        cols, rows=self.second_order_edges.T
        jax.debug.print('rows: {x}', x=rows)
        jax.debug.print('cols: {x}', x=cols)
        W_first_order=W_first_order.at[rows, cols].set(0)

        jax.debug.print('W_first_order: {x}', x=W_first_order)

        dydt = jnp.zeros(self.n)
        for i in range(self.n):
            jax.debug.print('i: {x}', x=i)
            jax.debug.print('first order terms (positive): {x} * {y}',  x=W_first_order[i], y=y)
            jax.debug.print('first order terms (negative): {x} * {y}', x=W_first_order[:, i], y=y[i])
            jax.debug.print('first order contribution: {x}', x=dydt[i] + W_first_order[i] @ y - jnp.sum(W_first_order[:, i] * y[i]))

            dydt=dydt.at[i].set(dydt[i] + W_first_order[i] @ y - jnp.sum(W_first_order[:, i] * y[i]))
            
            #second order reactions
            jax.debug.print('before second order: {x}', x=dydt)
            for edge in self.second_order_edges:
                idx_i=edge[0]
                idx_f=edge[1]
                
                c=0
                
                #there is a term for each reactant in the reaction if the reactant is the same as the species being considered 
                for j in self.second_order_edge_idxs[tuple(edge.tolist())][0]:
                    jax.debug.print('j: {x}', x=j)
                    #skip over chaperones
                    if j in self.second_order_edge_idxs[tuple(edge.tolist())][1]:
                        jax.debug.print('not consumed')
                        continue
                    term=1
                    
                    if i ==j:
                        c=-1  
                        for k in self.second_order_edge_idxs[tuple(edge.tolist())][0]:
                            jax.debug.print('k: {x}', x=k)
                            jax.debug.print('y[k]: {x}', x=y[k])
                            term=term*y[k] #multiply by the concentration of the other species in the reaction
                        
                        jax.debug.print('second order rate term: {x}', x=W[idx_f, idx_i])  
                        jax.debug.print('second order: {x}', x=c*term*W[idx_f, idx_i])

                        dydt=dydt.at[i].set(dydt[i] + c*term*W[idx_f, idx_i])
            
                #there is term for each product in the reaction if the reactant is the same as the species being considered
                for j in  self.second_order_edge_idxs[tuple(edge.tolist())][1]:   
                    jax.debug.print('j: {x}', x=j)    
                    if j in self.second_order_edge_idxs[tuple(edge.tolist())][0]:
                        jax.debug.print('not consumed')
                        continue 
                    term=1
                    if i ==j:
                        c=1
                        for k in self.second_order_edge_idxs[tuple(edge.tolist())][0]:
                            jax.debug.print('k: {x}', x=k)
                            jax.debug.print('y[k]: {x}', x=y[k])
                            term=term*y[k] #multiply by the concentration of the other species in the reaction
                        jax.debug.print('second order rate term: {x}', x=W[idx_f, idx_i])  
                        jax.debug.print('second order: {x}', x=c*term*W[idx_f, idx_i])

                        dydt=dydt.at[i].set(dydt[i] + c*term*W[idx_f, idx_i]) 
            
                jax.debug.print('updated dydt: {x}', x=dydt)
        jax.debug.print('dydt:{x}',x=dydt)
        jax.debug.print('dlog(y)dt:{x}',x=dydt/y)
                
        return dydt / y

class rxn_net:
    def __init__(self, net_type):
        self.net_type=net_type

    #def random_net(self, t, y, init_E_params, init_B_params, init_F_params, F_a_idxs, init_F_a_params):
        
        #return dydt
        
    def gbk_dynamics(self, t, y, params):
        y=jnp.clip(y, 0)

        Ea1, Ba1, Fa1, Ea2, Ba2, Fa2, Ed1, Bd1, Fd1, Ed2, Bd2, Fd2, Fd2_in, Ek1, Bk1, Fk1, Ek2, Bk2, Fk2, Fa1_in, Fa2_in=params
    
        Fd1_in=0
        Fd2_in=0
        Fk1_in=0
        Fk2_in=0

        #rates
        a1=jnp.exp(Ea1-Ba1+0.5*Fa1+0.5*Fa1_in)
        a2=jnp.exp(Ea2-Ba2+0.5*Fa2+0.5*Fa2_in)
        d1=jnp.exp(Ed1-Bd1+0.5*Fd1+0.5*Fd1_in)
        d2=jnp.exp(Ed2-Bd2+0.5*Fd2+0.5*Fd2_in)
        k1=jnp.exp(Ek1-Bk1+0.5*Fk1+0.5*Fk1_in)
        k2=jnp.exp(Ek2-Bk2+0.5*Fk2+0.5*Fk2_in)
        
        W, WE1, W_star, W_star_E2, E1, E2 = y

        dW_dt = -a1*W*E1 + d1*WE1 + k2*W_star_E2
        dWE1_dt = a1*W*E1 - (d1+k1)*WE1
        dW_star_dt = -a2*W_star*E2 + d2*W_star_E2 + k1*WE1
        dW_star_E2_dt = a2*W_star*E2 - (d2+k2)*W_star_E2
        dE1_dt = -dWE1_dt
        dE2_dt = -dW_star_E2_dt
   
        return jnp.array([dW_dt, dWE1_dt, dW_star_dt, dW_star_E2_dt, dE1_dt, dE2_dt])
    
    #from Figure 13 of Cal's SI revisions
    #log transformed dynamics
    def triangle_topology_a(self, t, y, params):
        A, B, C=jnp.exp(y)
        
        E_A, E_B, E_C, B_AB, F_AB, B_BA, B_AC, F_AC, B_CA, B_BC, F_BC, B_CB, F_BC_in=params
        F_CB_in=-F_BC_in
        F_AB_in, F_BA_in, F_AC_in,F_CA_in = 0, 0, 0, 0

        F_BA=-F_AB
        F_CA=-F_AC
        F_CB=-F_BC

        #jax.debug.print('params:{params}', params=params)
        W_AB=jnp.exp(E_A-B_AB+0.5*F_AB + 0.5*F_AB_in)
        W_BA=jnp.exp(E_B-B_BA+0.5*F_BA + 0.5*F_BA_in)
        W_AC=jnp.exp(E_A-B_AC+0.5*F_AC + 0.5*F_AC_in)
        W_CA=jnp.exp(E_C-B_CA+0.5*F_CA + 0.5*F_CA_in)
        W_BC=jnp.exp(E_B-B_BC+0.5*F_BC + 0.5*F_BC_in)
        W_CB=jnp.exp(E_C-B_CB+0.5*F_CB + 0.5*F_CB_in)

        #mass action kinetics (for log(A), log(B), log(C))
        dAdt=W_AB*B + W_AC*C - W_CA*A - W_BA*A
        dBdt=W_BA*A + W_BC*C - W_AB*B - W_CB*B
        dCdt=W_CA*A + W_CB*B - W_AC*C - W_BC*C
        
        return jnp.array([dAdt/A, dBdt/B, dCdt/C])
 
    def triangle_topology_b(self, t, y, params):
        A, B, C=jnp.exp(y)
        
        E_A, E_B, E_C, B_AB, F_AB, B_BA, B_AC, F_AC, B_CA, B_BC, F_BC, B_CB, F_BC_in=params
        F_CB_in=-F_BC_in.copy()
        F_AB_in, F_BA_in, F_AC_in,F_CA_in = 0, 0, 0, 0

        F_BA=-F_AB.copy()
        F_CA=-F_AC.copy()
        F_CB=-F_BC.copy()

        #jax.debug.print('params:{params}', params=params)
        W_AB=jnp.exp(E_A-B_AB+0.5*F_AB + 0.5*F_AB_in)
        W_BA=jnp.exp(E_B-B_BA+0.5*F_BA + 0.5*F_BA_in)
        W_AC=jnp.exp(E_A-B_AC+0.5*F_AC + 0.5*F_AC_in)
        W_CA=jnp.exp(E_C-B_CA+0.5*F_CA + 0.5*F_CA_in)
        W_BC=jnp.exp(E_B-B_BC+0.5*F_BC + 0.5*F_BC_in)
        W_CB=jnp.exp(E_C-B_CB+0.5*F_CB + 0.5*F_CB_in)

        #mass action kinetics (for log(A), log(B), log(C))
        dAdt=W_AB*B + W_AC*C - W_CA*A - W_BA*A*C
        dBdt=W_BA*A * C + W_BC*C - W_AB*B - W_CB*B
        dCdt=W_CA*A + W_CB*B - W_AC*C - W_BC*C

        return jnp.array([dAdt/A, dBdt/B, dCdt/C])
    
    def triangle_topology_c(self, t, y, params):
        A, B, C=jnp.exp(y)

        E_A, E_B, E_C, B_AB, F_AB, B_BA, B_AC, F_AC, B_CA, B_BC, F_BC, B_CB, F_BC_in=params
        F_CB_in=-F_BC_in
        F_AB_in, F_BA_in, F_AC_in,F_CA_in = 0, 0, 0, 0

        F_BA=-F_AB
        F_CA=-F_AC
        F_CB=-F_BC

        #jax.debug.print('params:{params}', params=params)
        W_AB=jnp.exp(E_A-B_AB+0.5*F_AB + 0.5*F_AB_in)
        W_BA=jnp.exp(E_B-B_BA+0.5*F_BA + 0.5*F_BA_in)
        W_AC=jnp.exp(E_A-B_AC+0.5*F_AC + 0.5*F_AC_in)
        W_CA=jnp.exp(E_C-B_CA+0.5*F_CA + 0.5*F_CA_in)
        W_BC=jnp.exp(E_B-B_BC+0.5*F_BC + 0.5*F_BC_in)
        W_CB=jnp.exp(E_C-B_CB+0.5*F_CB + 0.5*F_CB_in)

        #mass action kinetics (for log(A), log(B), log(C))
        dAdt=W_AB*B + W_AC*C - W_CA*A - W_BA*A*C
        dBdt=2*W_BA*A*C+W_BC*C - W_AB*B - W_CB*B
        dCdt=W_CA*A + W_CB*B - W_AC*C - W_BC*C-W_BA*A*C

        return jnp.array([dAdt/A, dBdt/B, dCdt/C])

    def integrate(self, solver, stepsize_controller, t_points, dt0, initial_conditions, args, max_steps):
        if self.net_type == 'goldbeter_koshland':

            def wrapped_dynamics(t, y, args):
                return self.gbk_dynamics(t, y, args)
            
            term=ODETerm(wrapped_dynamics)

        elif self.net_type == 'triangle_a':
            def wrapped_dynamics(t, y, args):
                return self.triangle_topology_a(t, y, args)

            term=ODETerm(wrapped_dynamics)

        elif self.net_type == 'triangle_b':
            def wrapped_dynamics(t, y, args):
                return self.triangle_topology_b(t, y, args)
            
            term=ODETerm(wrapped_dynamics)

        elif self.net_type == 'triangle_c':
            def wrapped_dynamics(t, y, args):
                return self.triangle_topology_c(t, y, args)
            
            term=ODETerm(wrapped_dynamics)
        elif self.net_type == 'random':
            def wrapped_dynamics(t, y, args):
                return self.random_net(t, y, args)
            term=ODETerm(wrapped_dynamics)
            
        #right now doing no step size controller
        #solution = diffeqsolve(ODETerm(wrapped_dynamics), solver=solver, t0=t_points[0], t1=t_points[-1], dt0=dt0, y0=initial_conditions, args=args, saveat=SaveAt(ts=t_points), max_steps=max_steps)
        solution = diffeqsolve(ODETerm(wrapped_dynamics), solver=solver, stepsize_controller=stepsize_controller, t0=t_points[0], t1=t_points[-1], dt0=dt0, y0=initial_conditions, args=args, saveat=SaveAt(ts=t_points), max_steps=max_steps)

        return solution