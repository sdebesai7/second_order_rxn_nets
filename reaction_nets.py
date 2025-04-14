#code to define reaction networks
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt
import optax  

#a general reaction net object. 
#different methods corresponding to differen types of reaction nets
#integrate function numerically solves given parameters
class rxn_net:
    def __init__(self, net_type):
        self.net_type=net_type

    def gbk_dynamics(self, t, y, params):

        Ea1, Ba1, Fa1, Ea2, Ba2, Fa2, Ed1, Bd1, Fd1, Ed2, Bd2, Fd2, Fd2_in, Ek1, Bk1, Fk1, Ek2, Bk2, Fk2, Fa1_in, Fa2_in=params
    
        Fd1_in=0
        Fd2_in=0
        Fk1_in=0
        Fk2_in=0

        #rates
        
        a1=np.exp(Ea1-Ba1+0.5*Fa1+Fa1_in)
        a2=np.exp(Ea2-Ba2+0.5*Fa2+Fa2_in)
        d1=np.exp(Ed1-Bd1+0.5*Fd1+Fd1_in)
        d2=np.exp(Ed2-Bd2+0.5*Fd2+Fd2_in)
        k1=np.exp(Ek1-Bk1+0.5*Fk1+Fk1_in)
        k2=np.exp(Ek2-Bk2+0.5*Fk2+Fk2_in)
        
        #testing 
        '''
        a1=Ea1 + Fa1_in
        a2=Ea2 + Fa2_in
        d1=Ed1
        d2=Ed2
        k1=Ek1
        k2=Ek2
        '''

        W, WE1, W_star, W_star_E2, E1, E2 = y

        dW_dt = -a1*W*E1 + d1*WE1 + k2*W_star_E2
        dWE1_dt = a1*W*E1 - (d1+k1)*WE1
        dW_star_dt = -a2*W_star*E2 + d2*W_star_E2 + k1*WE1
        dW_star_E2_dt = a2*W_star*E2 - (d2+k2)*W_star_E2
        dE1_dt = -dWE1_dt
        dE2_dt = -dW_star_E2_dt
   
        return jnp.array([dW_dt, dWE1_dt, dW_star_dt, dW_star_E2_dt, dE1_dt, dE2_dt])
    
    #from Figure 13 of Cal's SI revisions
    def triangle_topology_a(self, t, y, params):
        A, B, C=y
        E_k1, B_k1, F_k1, F_k1_in=params
        k1=np.exp(E_k1-B_k1+0.5*F_k1 + F_k1_in)
        dAdt=-k1*A
        dBdt=-dAdt
        dCdt=0

        E_k1, B_k1, F_k1, F_k1_in=params
        
        return jnp.array([dAdt, dBdt, dCdt])
    
    def triangle_topology_b(self, t, y, params):
        A, B, C=y

        E_k1, B_k1, F_k1, F_k1_in=params

        k1=np.exp(E_k1-B_k1+0.5*F_k1 + F_k1_in)

        dAdt=-k1*A*C
        dBdt=-dAdt
        dCdt=0

        return jnp.array([dAdt, dBdt, dCdt])
    
    def triangle_topology_c(self, t, y, params):
        A, B, C=y

        E_k1, B_k1, F_k1, F_k1_in=params

        k1=np.exp(E_k1-B_k1+0.5*F_k1 + F_k1_in)

        dAdt=-k1*A*C
        dBdt=-2*dAdt
        dCdt=dAdt

        return jnp.array([dAdt, dBdt, dCdt])

    
    def integrate(self, solver, t_points, dt0, initial_conditions, args, max_steps):
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

        solution = diffeqsolve(ODETerm(wrapped_dynamics), solver, t0=t_points[0], t1=t_points[-1], dt0=dt0, y0=initial_conditions, args=args, saveat=SaveAt(ts=t_points), max_steps=max_steps)

        return solution