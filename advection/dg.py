"""
Discontinuous Galerkin for the advection equation.
"""

import numpy
from matplotlib import pyplot
import matplotlib as mpl
import quadpy
from scipy.integrate import ode

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


class Grid1d(object):

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0, m=3):

        assert m > 0

        self.ng = ng
        self.nx = nx
        self.m = m

        self.xmin = xmin
        self.xmax = xmax

        # python is zero-based.  Make easy intergers to know where the
        # real data lives
        self.ilo = ng
        self.ihi = ng+nx-1

        # physical coords -- cell-centered, left and right edges
        self.dx = (xmax - xmin)/(nx)
        self.x = xmin + (numpy.arange(nx+2*ng)-ng+0.5)*self.dx
        self.xl = xmin + (numpy.arange(nx+2*ng)-ng)*self.dx
        self.xr = xmin + (numpy.arange(nx+2*ng)-ng+1.0)*self.dx

        # storage for the solution
        # These are the modes of the solution at each point, so the
        # first index is the mode
        # NO! These are actually the *nodal* values
        self.a = numpy.zeros((self.m+1, nx+2*ng), dtype=numpy.float64)

        # Need the Gauss-Lobatto nodes and weights in the reference element
        GL = quadpy.line_segment.GaussLobatto(m+1)
        self.nodes = GL.points
        self.weights = GL.weights
        # To go from modal to nodal we need the Vandermonde matrix
        self.V = numpy.polynomial.legendre.legvander(self.nodes, m)
        c = numpy.eye(m+1)
        # Orthonormalize
        for p in range(m+1):
            self.V[:, p] /= numpy.sqrt(2/(2*p+1))
            c[p, p] /= numpy.sqrt(2/(2*p+1))
        self.V_inv = numpy.linalg.inv(self.V)
        self.M = numpy.linalg.inv(self.V @ self.V.T)
        self.M_inv = self.V @ self.V.T
        # Derivatives of Legendre polynomials lead to derivatives of V
        dV = numpy.polynomial.legendre.legval(self.nodes,
                                              numpy.polynomial.legendre.legder(c)).T
        self.D = dV @ self.V_inv
        # Stiffness matrix for the interior flux
        self.S = self.M @ self.D

        # Nodes in the computational coordinates
        self.all_nodes = numpy.zeros((m+1)*(nx+2*ng), dtype=numpy.float64)
        self.all_nodes_per_node = numpy.zeros_like(self.a)
        for i in range(nx+2*ng):
            self.all_nodes[(m+1)*i:(m+1)*(i+1)] = (self.x[i] +
                                                     self.nodes * self.dx / 2)
            self.all_nodes_per_node[:, i] = (self.x[i] +
                                             self.nodes * self.dx / 2)

#    def modal_to_nodal(self):
#        nodal = numpy.zeros_like(self.a)
#        for i in range(self.nx+2*self.ng):
#            nodal[:, i] = self.V @ self.a[:, i]
#        return nodal

    def nodal_to_modal(self):
        modal = numpy.zeros_like(self.a)
        for i in range(self.nx+2*self.ng):
            modal[:, i] = self.V_inv @ self.a[:, i]
        return modal

    def plotting_data(self):
        return (self.all_nodes,
                self.a.ravel(order='F'))
        
    def plotting_data_high_order(self, npoints=50):
        assert npoints > 2
        p_nodes = numpy.zeros(npoints*(self.nx+2*self.ng), dtype=numpy.float64)
        p_data = numpy.zeros_like(p_nodes)
        for i in range(self.nx+2*self.ng):
            p_nodes[npoints*i:npoints*(i+1)] = numpy.linspace(self.xl[i],
                                                              self.xr[i],
                                                              npoints)
            modal = self.V_inv @ self.a[:, i]
            for p in range(self.m+1):
                modal[p] /= numpy.sqrt(2/(2*p+1))
            scaled_x = 2 * (p_nodes[npoints*i:npoints*(i+1)] - self.x[i]) / self.dx
            p_data[npoints*i:npoints*(i+1)] = numpy.polynomial.legendre.legval(scaled_x, modal)
        return p_nodes, p_data

    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return numpy.zeros((self.m+1, self.nx+2*self.ng), dtype=numpy.float64)

    def fill_BCs(self):
        """ fill all single ghostcell with periodic boundary conditions """

        for n in range(self.ng):
            # left boundary
            self.a[:, self.ilo-1-n] = self.a[:, self.ihi-n]
            # right boundary
            self.a[:, self.ihi+1+n] = self.a[:, self.ilo+n]

    def norm(self, e):
        """ 
        Return the norm of quantity e which lives on the grid.
        
        This is the 'broken norm': the quantity is integrated over each
        individual element using Gauss-Lobatto quadrature (as we have those
        nodes and weights), and the 2-norm of the result is then returned.
        """
        if not numpy.allclose(e.shape, self.all_nodes_per_node.shape):
            return None


        # This is L_inf norm...
#        return numpy.max(abs(e[:, self.ilo:self.ihi+1]))
        # This is actually a pointwise norm, not quadrature'd
        return numpy.sqrt(self.dx*numpy.sum(e[:, self.ilo:self.ihi+1]**2))

#        element_norm = self.weights @ e
#        return numpy.sqrt(self.dx*numpy.sum(element_norm[self.ilo:self.ihi+1]**2))


class Simulation(object):

    def __init__(self, grid, u, C=0.8):
        self.grid = grid
        self.t = 0.0  # simulation time
        self.u = u    # the constant advective velocity
        self.C = C    # CFL number

    def init_cond(self, type="sine"):
        """ initialize the data """
        if type == "tophat":
            def init_a(x):
                return numpy.where(numpy.logical_and(x >= 0.333,
                                                     x <= 0.666),
                                   numpy.ones_like(x),
                                   numpy.zeros_like(x))
        elif type == "sine":
            def init_a(x):
                return numpy.sin(2.0 * numpy.pi * x /
                                 (self.grid.xmax - self.grid.xmin))
        elif type == "gaussian":
            def init_a(x):
                local_xl = x - self.grid.dx/2
                local_xr = x + self.grid.dx/2
                al = 1.0 + numpy.exp(-60.0*(local_xl - 0.5)**2)
                ar = 1.0 + numpy.exp(-60.0*(local_xr - 0.5)**2)
                ac = 1.0 + numpy.exp(-60.0*(x - 0.5)**2)
                
                return (1./6.)*(al + 4*ac + ar)


        self.grid.a = init_a(self.grid.all_nodes_per_node)

    def timestep(self):
        """ return the advective timestep """
        return self.C*self.grid.dx/self.u

    def period(self):
        """ return the period for advection with velocity u """
        return (self.grid.xmax - self.grid.xmin)/self.u

    def states(self):
        """ compute the left and right interface states """

        # Evaluate the nodal values at the domain edges
        g = self.grid

        al = numpy.zeros(g.nx+2*g.ng)
        ar = numpy.zeros(g.nx+2*g.ng)

        # i is looping over interfaces, so al is the right edge of the left
        # element, etc.
        for i in range(g.ilo, g.ihi+2):
            al[i] = g.a[-1, i-1]
            ar[i] = g.a[ 0, i  ]

        # TODO: remove this return, fix the limiting.    
        return al, ar
    
        # Limiting, using minmod (Hesthaven p 443-4)
        theta = 2
        a_modal = g.nodal_to_modal()
        a_average = a_modal[0, :]
        a_m = numpy.zeros_like(a_average)
        a_m[1:] = a_average[:-1]
        a_el = g.a[0 , :]
        a_er = g.a[-1, :]
        
        check_left  = a_average - minmod([a_average - a_el,
                                          a_average - a_m,
                                          a_er - a_average])
        check_right = a_average + minmod([a_average - a_el,
                                          a_average - m,
                                          a_er - a_average])
        ids = numpy.logical_or(numpy.isclose(a_el, check_left),
                               numpy.isclose(a_er, check_right))
        

        return al, ar

    def riemann(self, al, ar):
        """
        Riemann problem for advection -- this is simply upwinding,
        but we return the flux
        """

        if self.u > 0.0:
            return self.u*al
        else:
            return self.u*ar

    def rk_substep(self):
        """
        Take a single RK substep
        """
        g = self.grid
        g.fill_BCs()
        rhs = g.scratch_array()

        # Integrate flux over element
        f = self.u * g.a
        interior_f = g.S.T @ f
        # Use Riemann solver to get fluxes between elements
        boundary_f = self.riemann(*self.states())
        rhs = interior_f
        rhs[ 0, 1:-1] += boundary_f[1:-1]
        rhs[-1, 1:-1] -= boundary_f[2:]

        # Multiply by mass matrix (inverse).
        rhs_i = 2 / g.dx * g.M_inv @ rhs

        return rhs_i

    def evolve(self, num_periods=1):
        """ evolve the linear advection equation using RK3 (SSP) """
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

            # RK3 - SSP
            # Store the data at the start of the step
            a_start = g.a.copy()
            k1 = dt * self.rk_substep()
            g.a = a_start + k1
            a1 = g.a.copy()
            k2 = dt * self.rk_substep()
            g.a = (3 * a_start + a1 + k2) / 4
            a2 = g.a.copy()
            k3 = dt * self.rk_substep()
            g.a = (a_start + 2 * a2 + 2 * k3) / 3

            self.t += dt

    def evolve_scipy(self, num_periods=1):
        """ evolve the linear advection equation using scipy """
        self.t = 0.0
        g = self.grid
        
        def rk_substep_scipy(t, y):
            local_a = numpy.reshape(y, g.a.shape)
            # Periodic BCs
            local_a[:, :g.ng] = local_a[:, -2*g.ng:-g.ng]
            local_a[:, -g.ng:] = local_a[:, g.ng:2*g.ng]
            # Integrate flux over element
            f = self.u * local_a
            interior_f = g.S.T @ f
            # Use Riemann solver to get fluxes between elements
            al = numpy.zeros(g.nx+2*g.ng)
            ar = numpy.zeros(g.nx+2*g.ng)
            # i is looping over interfaces, so al is the right edge of the left
            # element, etc.
            for i in range(g.ilo, g.ihi+2):
                al[i] = local_a[-1, i-1]
                ar[i] = local_a[ 0, i  ]
            boundary_f = self.riemann(al, ar)
            rhs = interior_f
            rhs[ 0, 1:-1] += boundary_f[1:-1]
            rhs[-1, 1:-1] -= boundary_f[2:]
    
            # Multiply by mass matrix (inverse).
            rhs_i = 2 / g.dx * g.M_inv @ rhs
            
            return numpy.ravel(rhs_i, order='C')

        tmax = num_periods*self.period()
        r = ode(rk_substep_scipy).set_integrator('dop853')
        r.set_initial_value(numpy.ravel(g.a), 0)
        dt = self.timestep()

        # main evolution loop
        while r.successful() and r.t < tmax:
            dt = min(dt, tmax - r.t)
            r.integrate(r.t+dt)
        g.a[:, :] = numpy.reshape(r.y, g.a.shape)
    

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Show the "grid" using a sine wave

    xmin = 0.0
    xmax = 1.0
    nx = 4
    ng = 1

    u = 1.0

    colors="kbr"
    symbols="sox"
    for m in range(1, 4):
        g = Grid1d(nx, ng, xmin=xmin, xmax=xmax, m=m)
        s = Simulation(g, u, C=0.5/(2*m+1))
        s.init_cond("sine")
        plot_x, plot_a = g.plotting_data()
        plot_x_hi, plot_a_hi = g.plotting_data_high_order()
        pyplot.plot(plot_x, plot_a, f'{colors[m-1]}{symbols[m-1]}',
                    label=fr"Nodes, $m={{{m}}}$")
        pyplot.plot(plot_x_hi, plot_a_hi, f'{colors[m-1]}:',
                    label=fr"Modes, $m={{{m}}}$")
    pyplot.xlim(0, 1)
    pyplot.vlines([0.25, 0.5, 0.75], -2, 2, linestyles='--')
    pyplot.ylim(-1, 1)
    pyplot.xlabel(r'$x$')
    pyplot.ylabel(r'$a$')
    pyplot.legend()
    pyplot.show()


    # Note that the highest m (4) doesn't converge at the expected rate - 
    # probably limited by the time integrator.
    colors = "brckgy"
    symbols = "xo^<sd"
    fig, axes = pyplot.subplots(1, 2)
    ms = numpy.array(range(1, 5))
    nxs = 2**numpy.array(range(3, 9))
    errs = numpy.zeros((len(ms), len(nxs)))
    for i, m in enumerate(ms):
        for j, nx in enumerate(nxs):
            g = Grid1d(nx, ng, xmin=xmin, xmax=xmax, m=m)
            s = Simulation(g, u, C=0.5/(2*m+1))
            s.init_cond("sine")
#            s.init_cond("gaussian")
            a_init = g.a.copy()
            s.evolve(num_periods=1)
            errs[i, j] = s.grid.norm(s.grid.a - a_init)
        axes[0].loglog(nxs, errs[i, :], f'{colors[i]}{symbols[i]}')
        if m < 4:
            axes[0].plot(nxs, errs[i,-2]*(nxs[-2]/nxs)**(m+1),
                        f'{colors[i]}--')
    axes[0].set_xlabel(r'$N$')
    axes[0].set_ylabel(r'$\|$Error$\|_2$')
    axes[0].set_title('RK4')
    
    
    # To check that it's a limitation of the time integrator, we can use
    # the scipy DOPRK8 integrator
    colors = "brckgy"
    symbols = "xo^<sd"
    ms = numpy.array(range(1, 6))
    nxs = 2**numpy.array(range(3, 9))
    errs = numpy.zeros((len(ms), len(nxs)))
    for i, m in enumerate(ms):
        for j, nx in enumerate(nxs):
            print(f"DOPRK8, m={m}, nx={nx}")
            g = Grid1d(nx, ng, xmin=xmin, xmax=xmax, m=m)
            s = Simulation(g, u, C=0.5/(2*m+1))
            s.init_cond("sine")
#            s.init_cond("gaussian")
            a_init = g.a.copy()
            s.evolve_scipy(num_periods=1)
            errs[i, j] = s.grid.norm(s.grid.a - a_init)
        axes[1].loglog(nxs, errs[i, :], f'{colors[i]}{symbols[i]}',
                       label=fr'$m={{{m}}}$')
        if m < 5:
            axes[1].plot(nxs, errs[i,-2]*(nxs[-2]/nxs)**(m+1),
                         f'{colors[i]}--',
                         label=fr'$\propto (\Delta x)^{{{m+1}}}$')
        else:
            axes[1].plot(nxs[:-1], errs[i,-2]*(nxs[-2]/nxs[:-1])**(m+1),
                         f'{colors[i]}--',
                         label=fr'$\propto (\Delta x)^{{{m+1}}}$')
    axes[1].set_xlabel(r'$N$')
    axes[1].set_ylabel(r'$\|$Error$\|_2$')
    axes[1].set_title('DOPRK8')
    fig.tight_layout()
    lgd = axes[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('dg_convergence_sine.pdf', 
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    pyplot.show()
    