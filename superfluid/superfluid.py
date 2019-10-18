import sys
import numpy
from scipy.optimize import root
from scipy.integrate import quad
import weno_coefficients

lam = 1  # Coupling coefficient?


def eos_abc(s2, t2, st):
    """
    Compute the \bar{A}, \bar{B}, \bar{C} coefficients.

    Parameters
    ----------

    s2 : float
        sigma^2, where sigma is conjugate to j (mass current)
    t2 : float
        Theta^2, where Theta is conjugate to s (entropy current)
    st : float
        sigma cdot Theta

    Returns
    -------

    \bar{A}, \bar{B}, \bar{C} : float
        Coefficients in the conversion from sigma, Theta, to j, s
    """

    coeff = numpy.pi**2 * numpy.sqrt(3) / 135

    Abar = 4 * coeff * st / s2 * (t2 + 2 * st**2 / s2)
    Bbar = s2 / lam - 4 * coeff * (st / s2)**2 * (t2 + 2 * st**2 / s2)
    Cbar = coeff * (2 * t2 + 4 * st**2 / s2)

    return Abar, Bbar, Cbar


def c2p(q, guess):
    """
    Go from evolved variables q = (j^t, s^t, sigma_i, Theta_i) to all
    variables w = (j^a, a^a, sigma_b, Theta_b)
    """

    j_t_up, s_t_up = q[0:2]
    Theta_x, Theta_y = q[2:4]
    sigma_x, sigma_y = q[4:6]

    def root_fn(v):
        """
        From guesses for sigma_t, Theta_t, get j^t, s^t, compare.
        """
        sigma_t, Theta_t = v
        s2 = -sigma_t**2 + sigma_x**2 + sigma_y**2
        t2 = -Theta_t**2 + Theta_x**2 + Theta_y**2
        st = -sigma_t*Theta_t + sigma_x*Theta_x + sigma_y*Theta_y
        Abar, Bbar, Cbar = eos_abc(s2, t2, st)
        j_t_hat_down = Bbar * sigma_t + Abar * Theta_t
        s_t_hat_down = Abar * sigma_t + Cbar * Theta_t
        j_t_hat = -j_t_hat_down
        s_t_hat = -s_t_hat_down
        return [j_t_up - j_t_hat, s_t_up - s_t_hat]

    sol = root(root_fn, guess)
    sigma_t, Theta_t = sol.x
    s2 = -sigma_t**2 + sigma_x**2 + sigma_y**2
    t2 = -Theta_t**2 + Theta_x**2 + Theta_y**2
    st = -sigma_t*Theta_t + sigma_x*Theta_x + sigma_y*Theta_y
    Abar, Bbar, Cbar = eos_abc(s2, t2, st)
    j_x_down = Bbar * sigma_x + Abar * Theta_x
    s_x_down = Abar * sigma_x + Cbar * Theta_x
    j_y_down = Bbar * sigma_y + Abar * Theta_y
    s_y_down = Abar * sigma_y + Cbar * Theta_y
    j_x_up = j_x_down
    s_x_up = s_x_down
    j_y_up = j_y_down
    s_y_up = s_y_down

    w = numpy.array([j_t_up, s_t_up, j_x_up, s_x_up, j_y_up, s_y_up,
                     sigma_t, Theta_t, sigma_x, Theta_x, sigma_y, Theta_y])
    return w


class Grid1d(object):

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0, bc="outflow"):

        self.ng = ng
        self.nx = nx

        self.xmin = xmin
        self.xmax = xmax

        self.bc = bc

        # python is zero-based.  Make easy intergers to know where the
        # real data lives
        self.ilo = ng
        self.ihi = ng+nx-1

        # physical coords -- cell-centered, left and right edges
        self.dx = (xmax - xmin)/(nx)
        self.x = xmin + (numpy.arange(nx+2*ng)-ng+0.5)*self.dx

        # storage for the solution
        self.q = numpy.zeros((6, (nx+2*ng)), dtype=numpy.float64)
        # storage for all variables
        self.w = numpy.zeros((12, (nx+2*ng)), dtype=numpy.float64)

    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return numpy.zeros((6, (self.nx+2*self.ng)), dtype=numpy.float64)

    def fill_BCs(self):
        """ fill all ghostcells as periodic """

        if self.bc == "periodic":

            # left boundary
            self.q[:, 0:self.ilo] = self.q[:, self.ihi-self.ng+1:self.ihi+1]

            # right boundary
            self.q[:, self.ihi+1:] = self.q[:, self.ilo:self.ilo+self.ng]

        elif self.bc == "outflow":

            for n in range(self.ng):
                # left boundary
                self.q[:, n] = self.q[:, self.ilo]

                # right boundary
                self.q[:, self.ihi+1+n] = self.q[:, self.ihi]

        else:
            sys.exit("invalid BC")


def weno(order, q):
    """
    Do WENO reconstruction

    Parameters
    ----------

    order : int
        The stencil width
    q : numpy array
        Scalar data to reconstruct

    Returns
    -------

    qL : numpy array
        Reconstructed data - boundary points are zero
    """
    C = weno_coefficients.C_all[order]
    a = weno_coefficients.a_all[order]
    sigma = weno_coefficients.sigma_all[order]

    qL = numpy.zeros_like(q)
    beta = numpy.zeros((order, q.shape[1]))
    w = numpy.zeros_like(beta)
    np = q.shape[1] - 2 * order
    epsilon = 1e-16
    for nv in range(3):
        for i in range(order, np+order):
            q_stencils = numpy.zeros(order)
            alpha = numpy.zeros(order)
            for k in range(order):
                for l in range(order):
                    for m in range(l+1):
                        beta[k, i] += (sigma[k, l, m] * q[nv, i+k-l] *
                                       q[nv, i+k-m])
                alpha[k] = C[k] / (epsilon + abs(beta[k, i])**order)
                for l in range(order):
                    q_stencils[k] += a[k, l] * q[nv, i+k-l]
            w[:, i] = alpha / numpy.sum(alpha)
            qL[nv, i] = numpy.dot(w[:, i], q_stencils)

    return qL


class WENOSimulation(object):

    def __init__(self, grid, C=0.5, weno_order=3):
        self.grid = grid
        self.t = 0.0  # simulation time
        self.C = C    # CFL number
        self.weno_order = weno_order
        self.guess = (1, 1)  # This really needs thought.

    def init_cond(self, type="const"):
        if type == "const":
            self.grid.q = 0
            self.grid.q[0] = numpy.ones_like(self.grid.x)
            self.grid.q[1] = numpy.ones_like(self.grid.x)

    def max_lambda(self):
        return 1

    def timestep(self):
        return self.C * self.grid.dx / self.max_lambda()

    def superfluid_flux(self, q):
        flux = numpy.zeros_like(q)
        w = c2p(q, self.guess)
#        jt = w[0, :]
#        st = w[1, :]
        jx = w[2, :]
        sx = w[3, :]
#        jy = w[4, :]
#        sy = w[5, :]
        sigmat = w[6, :]
        Thetat = w[7, :]
#        sigmax = w[8, :]
#        Thetax = w[9, :]
#        sigmay = w[10, :]
#        Thetay = w[11, :]

        flux[0, :] = jx
        flux[1, :] = sx
        flux[2, :] = -Thetat
        flux[4, :] = -sigmat
        return flux

    def B_matrix(self, q):
        w = c2p(q, self.guess)
        st = w[1]
        sx = w[3]
        sy = w[5]
        n = len(q)
        B = numpy.zeros((n, n), dtype=q.dtype)
        B[2, 3] = - sy / st
        B[3, 3] = sx / st
        return B

    def path_psi(self, q_a, q_b, s):
        return q_a + (q_b - q_a) * s

    def B_tilde(self, q_a, q_b):
        (result, _) = quad(lambda s: self.B_matrix(self.path_psi(q_a, q_b, s)),
                           0, 1)
        return result

    def speeds(self, q_L, q_R):
        """
        The minimum and maximum signal speeds in the problem.
        Relativity, max is c=1, do the very diffusive variant.
        """
        return (-1, 1)

    def HLL(self, q_L, q_R):
        """
        Path consistent flux based on Dumbser & Balsara JCP 304 (275-319) 2016.
        """
        s_L, s_R = self.speeds(q_L, q_R)
#        f_L = flux(sim, q_L)
#        f_R = flux(sim, q_R)
        # Equation (15)
        q_star_0 = 1 / (s_R - s_L) * \
                   ( (q_R * s_R - q_L * s_L) -
#                     (f_R - f_L) -
                     self.B_tilde(q_L, q_R) @ (q_R - q_L) )
        # Not doing the iterative step for now
        q_star = q_star_0
        # Fluctuations from equations 23
        D_m = -s_L / (s_R - s_L) * ( #(f_R - f_L) + 
                                     self.B_tilde(q_L, q_star) @ (q_star - q_L) +
                                     self.B_tilde(q_star, q_R) @ (q_R - q_star) )+\
               s_L * s_R / (s_R - s_L) * (q_R - q_L)
        D_p =  s_R / (s_R - s_L) * ( #(f_R - f_L) +
                                     self.B_tilde(q_L, q_star) @ (q_star - q_L) +
                                     self.B_tilde(q_star, q_R) @ (q_R - q_star) )-\
               s_L * s_R / (s_R - s_L) * (q_R - q_L)
        # Done
        # HACK to remove pure flux terms
        D_m[0:2] = 0
        D_m[4:] = 0
        D_p[0:2] = 0
        D_p[4:] = 0
        # END HACK
        return D_m, D_p

    def rk_substep(self):
        """RHS terms"""
        g = self.grid
        g.fill_BCs()
        f = self.superfluid_flux(g.q)
        alpha = self.max_lambda()
        fp = (f + alpha * g.q) / 2
        fm = (f - alpha * g.q) / 2
        fpr = g.scratch_array()
        fml = g.scratch_array()
        flux = g.scratch_array()
        fpr[:, 1:] = weno(self.weno_order, fp[:, :-1])
        fml[:, -1::-1] = weno(self.weno_order, fm[:, -1::-1])
        flux[:, 1:-1] = fpr[:, 1:-1] + fml[:, 1:-1]
        # --- WARNING!!! ---
        # Terms with no fluxes (Theta_y, sigma_y) will pick up diffusive
        # pieces from the L-F term, so set to zero by hand
        flux[3, :] = 0  # Theta_y
        flux[5, :] = 0    # sigma_y
        # --- END OF HACK ---
        rhs = g.scratch_array()
        rhs[:, 1:-1] = 1/g.dx * (flux[:, 1:-1] - flux[:, 2:])
        # Now reconstruct q raw
        qpr = g.scratch_array()
        qml = g.scratch_array()
        qpr[:, 1:] = weno(self.weno_order, self.q[:, :-1])
        qml[:, -1::-1] = weno(self.weno_order, self.q[:, -1::-1])
        D_m = g.scratch_array()
        D_p = g.scratch_array()
        for i in range(g.ilo-1, g.ihi+1):
            D_m[:, i], D_p[:, i] = self.HLL(qpr[:, i-1], qml[:, i])
            # CHECK:
            # Should there be a B delta Q term here?
        rhs[:, 1:-1] -= 1/g.dx * (D_m[2:] + D_p[1:-1])
        return rhs

    def evolve(self, tmax, reconstruction='componentwise'):
        """ evolve the Euler equation using RK3 """
        self.t = 0.0
        g = self.grid

        stepper = self.rk_substep
        if reconstruction == 'characteristic':
            raise NotImplementedError
#            stepper = self.rk_substep_characteristic

        # main evolution loop
        while self.t < tmax:

            # fill the boundary conditions
            g.fill_BCs()

            # get the timestep
            dt = self.timestep()

            if self.t + dt > tmax:
                dt = tmax - self.t

            # RK3: this is SSP
            # Store the data at the start of the step
            q_start = g.q.copy()
            q1 = q_start + dt * stepper()
            g.q[:, :] = q1[:, :]
            q2 = (3 * q_start + q1 + dt * stepper()) / 4
            g.q[:, :] = q2[:, :]
            g.q = (q_start + 2 * q2 + 2 * dt * stepper()) / 3

            self.t += dt
