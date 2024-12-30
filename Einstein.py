import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit





# Useful constants
r_scale = 1.477 # in km
K_NS = 100





def TOV_eqs(r, mνpb_arr):
    m = mνpb_arr[0]
    ν = mνpb_arr[1]
    p = mνpb_arr[2]
    b = mνpb_arr[3]


    if r == 0:
        dνdr = 0
        dpdr = 0
        dmdr = 0
        dbdr = 0


    else:

        if p < 0:
            p = 0


        ρ = np.sqrt(p / K_NS)

        # Define ODEs
        dmdr = 4*pi*ρ * r**2
        dνdr = 2 * (m + 4*pi*p*r**3) / (r * (r - 2*m))
        dpdr = (-(m + 4*pi*p*r**3) / (r * (r - 2*m))) * (ρ + p)
        dbdr = dmdr / np.sqrt(1 - (2*m / r))

    return np.array([dmdr, dνdr, dpdr, dbdr])



def p_reaches_0(r, mνpb_arr):
    p = mνpb_arr[2]
    return p

p_reaches_0.terminal = True
p_reaches_0.direction = 0





def my_test_einstein():
    """
    Main function for Einstein.

    PART B: Reads the WD data .csv file, and records the loggs and masses of
    WDs. Performs a unit conversion of loggs to earth radii and plots these
    M(R) data points.

    PART C: Filters out WD datapoints with low mass. Performs a linear fit to
    determine q and n*. Solves the Lane-Emden equation with newly found n* to
    find ξn and θ'(ξn). Uses all these parameters to find K*. Finally computes
    ρ_c and plots it against low mass WD masses.

    """
    #=========================================================================
    # PART A and B
    #=========================================================================


    r_span = (0, 15)
    r_eval = np.linspace(r_span[0], r_span[1], 500)

    ρ_c_vals = np.linspace(9e-4, 9e-3, 101)

    R_pc_vals = np.zeros(len(ρ_c_vals))
    M_pc_vals = np.zeros(len(ρ_c_vals))
    Δ_pc_vals = np.zeros(len(ρ_c_vals))


    for i in range(len(ρ_c_vals)):

        ρ_c = ρ_c_vals[i]
        p_0 = K_NS * (ρ_c**2)

        mνp_0 = [0, 0, p_0, 0]


        sol = solve_ivp(TOV_eqs, r_span, mνp_0, t_eval=r_eval, method='RK45',
                                                        events = p_reaches_0)


        R_pc = sol.t[-1]
        M_pc = sol.y[0, -1]
        B_pc = sol.y[3, -1]

        R_pc_vals[i] = R_pc
        M_pc_vals[i] = M_pc
        Δ_pc_vals[i] = (B_pc - M_pc) / M_pc


    R_pc_vals *= r_scale


    def quad_fit(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d

    coeffs, cov = curve_fit(quad_fit, R_pc_vals, M_pc_vals)
    a, b, c, d = coeffs

    R_fit = np.linspace(min(R_pc_vals), max(R_pc_vals), 150)
    M_fit = quad_fit(R_fit, a, b, c, d)

    plt.scatter(R_pc_vals, M_pc_vals)
    plt.plot(R_fit, M_fit)
    plt.show()




    plt.scatter(R_pc_vals, Δ_pc_vals)
    plt.show()
    return

my_test_einstein()