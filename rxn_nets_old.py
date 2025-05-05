#code to define reaction networks
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt
import optax  
import networkx as nx

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