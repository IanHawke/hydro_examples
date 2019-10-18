from functools import partial
import numpy
from matplotlib import pyplot
import advection
import weno
from scipy.integrate import ode, quad


class PathConsSimulation(advection.Simulation):

    def __init__(self, grid, u, C=0.8, weno_order=3):
        self.grid = grid
        self.t = 0.0  # simulation time
        self.u = u    # the constant advective velocity
        self.C = C    # CFL number
        self.weno_order = weno_order

    def B_matrix(self, q):
        return self.u

    def flux(self, q):
        return self.u * q

    def path_psi(self, q_a, q_b, s):
        return q_a + (q_b - q_a) * s

    def B_tilde(self, q_a, q_b):
        integrand = lambda s: self.B_matrix(self.path_psi(q_a, q_b, s))
        (result, _) = quad(integrand, 0, 1)
        return result

    def speeds(sim, q_L, q_R):
        """
        The minimum and maximum signal speeds in the problem.
        Ahead of going to relativity, do the very diffusive variant.
        """
        assert abs(sim.u) <= 1
        return (-1, 1)

    def HLL(sim, q_L, q_R):
        """
        A path consistent flux based on Dumbser & Balsara JCP 304 (275-319) 2016.
        """
        s_L, s_R = speeds(sim, q_L, q_R)
        f_L = flux(sim, q_L)
        f_R = flux(sim, q_R)
        # Equation (15)
        q_star_0 = 1 / (s_R - s_L) * \
                   ( (q_R * s_R - q_L * s_L) -
                     (f_R - f_L) -
                     B_tilde(sim, q_L, q_R) * (q_R - q_L) )
        # Not doing the iterative step for now
        q_star = q_star_0
        # Fluctuations from equations 23
        D_m = -s_L / (s_R - s_L) * ( (f_R - f_L) + 
                                     B_tilde(sim, q_L, q_star) * (q_star - q_L) +
                                     B_tilde(sim, q_star, q_R) * (q_R - q_star) )+\
               s_L * s_R / (s_R - s_L) * (q_R - q_L)
        D_p =  s_R / (s_R - s_L) * ( (f_R - f_L) +
                                     B_tilde(sim, q_L, q_star) * (q_star - q_L) +
                                     B_tilde(sim, q_star, q_R) * (q_R - q_star) )-\
               s_L * s_R / (s_R - s_L) * (q_R - q_L)
        # Done
        return D_m, D_p
    

    def init_cond(self, type="tophat"):
        """ initialize the data """
        if type == "sine_sine":
            self.grid.a[:] = numpy.sin(numpy.pi*self.grid.x - 
                       numpy.sin(numpy.pi*self.grid.x) / numpy.pi)
        else:
            super().init_cond(type)


    def rk_substep(self):
        
        g = self.grid
        g.fill_BCs()
        f = self.u * g.a
        alpha = abs(self.u)
        fp = (f + alpha * g.a) / 2
        fm = (f - alpha * g.a) / 2
        fpr = g.scratch_array()
        fml = g.scratch_array()
        flux = g.scratch_array()
        fpr[1:] = weno(self.weno_order, fp[:-1])
        fml[-1::-1] = weno(self.weno_order, fm[-1::-1])
        flux[1:-1] = fpr[1:-1] + fml[1:-1]
        rhs = g.scratch_array()
        rhs[1:-1] = 1/g.dx * (flux[1:-1] - flux[2:])
        return rhs


    def evolve(self, num_periods=1):
        """ evolve the linear advection equation using RK4 """
        self.t = 0.0
        g = self.grid

        tmax = num_periods*self.period()

        # main evolution loop
        while self.t < tmax:

            # fill the boundary conditions
            g.fill_BCs()

            # get the timestep
            dt = self.timestep()

            if self.t + dt > tmax:
                dt = tmax - self.t

            # RK4
            # Store the data at the start of the step
            a_start = g.a.copy()
            k1 = dt * self.rk_substep()
            g.a = a_start + k1 / 2
            k2 = dt * self.rk_substep()
            g.a = a_start + k2 / 2
            k3 = dt * self.rk_substep()
            g.a = a_start + k3
            k4 = dt * self.rk_substep()
            g.a = a_start + (k1 + 2 * (k2 + k3) + k4) / 6

            self.t += dt


g = advection.Grid1d(nx=100, ng=3)
s = advection.Simulation(g, 1, 0.5)
s.init_cond()
print(HLL(s, s.grid.a[20], s.grid.a[21]))
