import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit





# Useful constants
r_scale = 1.477 # in km
K_NS = 100





def TOV_eqs(r, mvpb_arr):
    # TOVEQS Function for solving the Tolman-Oppenheimer-Volkoff (TOV) ODEs.
    # This function sets the ewquations up like an ivp to be solved by SciPy's
    # solve_ivp(). It takes care of the edge cases where r = 0 or where p goes
    # below 0 during the integration.

    # v: time dialation factor (nu)
    # b: baryonic mass

    # Initial conditions
    m = mvpb_arr[0]
    v = mvpb_arr[1]
    p = mvpb_arr[2]
    b = mvpb_arr[3]

    # Case for if r = 0
    if r == 0:
        dvdr = 0
        dpdr = 0
        dmdr = 0
        dbdr = 0


    else:

        # Case for if p goes below 0
        if p < 0:
            p = 0

        # Compute ρ
        ρ = np.sqrt(p / K_NS)

        # Define ODEs
        dmdr = 4*pi*ρ * r**2
        dvdr = 2 * (m + 4*pi*p*r**3) / (r * (r - 2*m))
        dpdr = (-(m + 4*pi*p*r**3) / (r * (r - 2*m))) * (ρ + p)
        dbdr = dmdr / np.sqrt(1 - (2*m / r))

    return np.array([dmdr, dvdr, dpdr, dbdr])



def p_reaches_0(r, mvpb_arr):
    # PREACHES0 This function and the following lines of code act like a
    # signal to solve_ivp() to stop integration when p reaches 0. At that point,
    # we'd have reached the maximal radius of the NS.

    p = mvpb_arr[2]
    return p

p_reaches_0.terminal = True
p_reaches_0.direction = 0





def my_test_einstein():
    """
    Main function for Einstein.

    PARTS A and B: Solves the TOV equations for various ρ_c and plots the
    resultant M(ρ_c) vs R(ρ_c) and Δ(ρ_c) vs R(ρ_c) curves.

    """
    #=========================================================================
    # PARTS A and B
    #=========================================================================

    # NSs have radii 10-20kms. 15 * r_scale means we can see NSs with radii
    # from 0 ~ 30 kms.
    r_span = (0, 15)
    r_eval = np.linspace(r_span[0], r_span[1], 500)

    # In rescaled units, ρ ~ 10^-3, so we check values between 0.9*10^-3 to
    # 9*10^-3
    ρ_c_vals = np.linspace(9e-4, 9e-3, 101)

    # Initialize end of integration value arrays.
    R_pc_vals = np.zeros(len(ρ_c_vals))
    M_pc_vals = np.zeros(len(ρ_c_vals))
    Δ_pc_vals = np.zeros(len(ρ_c_vals))


    for i in range(len(ρ_c_vals)):

        # Record ρ_c and compute initial pressure
        ρ_c = ρ_c_vals[i]
        p_0 = K_NS * (ρ_c**2)

        # Full ODEs initial conditions
        mvp_0 = [0, 0, p_0, 0]

        # Solve ODE system, stopping when p reaches 0.
        sol = solve_ivp(TOV_eqs, r_span, mvp_0, t_eval=r_eval, method='RK45',
                                                        events = p_reaches_0)

        # Record final radius, mass, and baryonic mass
        R_pc = sol.t[-1]
        M_pc = sol.y[0, -1]
        B_pc = sol.y[3, -1]

        # Updatea rrays
        R_pc_vals[i] = R_pc
        M_pc_vals[i] = M_pc
        Δ_pc_vals[i] = (B_pc - M_pc) / M_pc

    # Scale radii
    R_pc_vals *= r_scale


    # Make cubic polynomial fit of datapoints
    R_fit = np.linspace(min(R_pc_vals), max(R_pc_vals), 150)

    def quad_fit(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d

    # Plotting MR datapoints and fit curve
    coeffsM, cov = curve_fit(quad_fit, R_pc_vals, M_pc_vals)
    aM, bM, cM, dM = coeffsM

    M_fit = quad_fit(R_fit, aM, bM, cM, dM)

    plt.scatter(R_pc_vals, M_pc_vals, alpha=0.5, color = "orangered", label = "Datapoints")
    plt.plot(R_fit, M_fit, color = "orangered", label = "Fit Curve")
    plt.title(r'$M(\rho_c)$ vs $R(\rho_c)$ plot for NSs')
    plt.xlabel(r'$R(\rho_c)$ (km)')
    plt.ylabel(r'$M(\rho_c) \quad (M \odot)$')
    plt.text(14, 1.3, r'Low $\rho_c$', bbox=dict(facecolor="white"))
    plt.text(10, 1.8, r'High $\rho_c$', bbox=dict(facecolor="white"))
    plt.legend()
    plt.show()


    # Plotting ΔR datapoints and fit curve (B here for baryonic mass)
    coeffsB, cov = curve_fit(quad_fit, R_pc_vals, Δ_pc_vals)
    aB, bB, cB, dB = coeffsB

    B_fit = quad_fit(R_fit, aB, bB, cB, dB)

    plt.scatter(R_pc_vals, Δ_pc_vals, alpha=0.5, color = "dodgerblue", label = "Datapoints")
    plt.plot(R_fit, B_fit, color = "dodgerblue", label = "Fit Curve")
    plt.title(r'$\Delta(\rho_c)$ vs $R(\rho_c)$ plot for NSs')
    plt.xlabel(r'$R(\rho_c)$ (km)')
    plt.ylabel(r'$\Delta(\rho_c)$')
    plt.legend()
    plt.show()



    #=========================================================================
    # PART C
    #=========================================================================
    return

my_test_einstein()