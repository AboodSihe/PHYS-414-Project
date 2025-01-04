import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit





# Useful constants
r_scale_km = 1.477 # in km
r_scale_m = 1477 # in m
K_NS = 100
solar_mass = 1.988416e30





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
    resultant M(ρc) vs R(ρc) and Δ(ρc) vs R(ρc) curves.

    """



    #=========================================================================
    # PARTS A and B
    #=========================================================================
    print('Einstein--Parts A and B Start: \n')

    # NSs have radii 10-20kms. 15 * r_scale means we can see NSs with radii
    # from 0 ~ 30 kms.
    r_span = (0, 15)
    r_eval = np.linspace(r_span[0], r_span[1], 500)

    # In rescaled units, ρ ~ 10^-3, so we check values between 0.9*10^-3 to
    # 9*10^-3
    num_points = 101
    ρ_c_vals = np.linspace(9e-4, 9e-3, num_points)

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
    R_pc_vals *= r_scale_km


    # Make cubic polynomial fit of datapoints
    R_fit = np.linspace(min(R_pc_vals), max(R_pc_vals), 150)

    def cubic_fit(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d

    # Plotting MR datapoints and fit curve
    coeffsM, cov = curve_fit(cubic_fit, R_pc_vals, M_pc_vals)
    aM, bM, cM, dM = coeffsM

    M_fit = cubic_fit(R_fit, aM, bM, cM, dM)

    plt.scatter(R_pc_vals, M_pc_vals, alpha=0.4, color = "orangered", label = "Datapoints")
    plt.plot(R_fit, M_fit, color = "orangered", label = "Fit Curve")
    plt.title(rf'$M(\rho_c)$ vs $R(\rho_c)$ plot for NSs (N = {num_points})')
    plt.xlabel(r'$R(\rho_c)$ (km)')
    plt.ylabel(r'$M(\rho_c) \quad (M \odot)$')
    plt.text(14, 1.3, r'Low $\rho_c$', bbox=dict(facecolor="white"))
    plt.text(10, 1.8, r'High $\rho_c$', bbox=dict(facecolor="white"))
    plt.legend()
    plt.show()


    # Plotting ΔR datapoints and fit curve (B here for baryonic mass)
    coeffsB, cov = curve_fit(cubic_fit, R_pc_vals, Δ_pc_vals)
    aB, bB, cB, dB = coeffsB

    B_fit = cubic_fit(R_fit, aB, bB, cB, dB)

    plt.scatter(R_pc_vals, Δ_pc_vals, alpha=0.4, color = "dodgerblue", label = "Datapoints")
    plt.plot(R_fit, B_fit, color = "dodgerblue", label = "Fit Curve")
    plt.title(rf'$\Delta(\rho_c)$ vs $R(\rho_c)$ plot for NSs (N = {num_points})')
    plt.xlabel(r'$R(\rho_c)$ (km)')
    plt.ylabel(r'$\Delta(\rho_c)$')
    plt.legend()
    plt.show()
    print()

    print('Einstein--Parts A and B Done. \n\n\n')



    #=========================================================================
    # PART C
    #=========================================================================
    print('Einstein--Part C Start: \n')

    #----------------------------------
    # Computing M(ρ_c) vs ρ_c curve
    #----------------------------------

    # Scale ρ_c values by 1000 just to avoid errors with curve fitting
    ρ_c_vals_scaled = ρ_c_vals * 1e3

    # Perform a quintic polynomial curve fit
    ρ_c_fit = np.linspace(min(ρ_c_vals_scaled), max(ρ_c_vals_scaled), 300)

    def quintic_fit(x, a, b, c, d, e, f):
        return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

    coeffs, cov = curve_fit(quintic_fit, ρ_c_vals_scaled, M_pc_vals)
    a, b, c, d, e, f = coeffs

    M_fit = quintic_fit(ρ_c_fit, a, b, c, d, e, f)

    # Rescale ρ_c values and fit to be in units kg/m^3 as instructed
    ρ_c_vals_SI = ρ_c_vals_scaled * (1e-3) * solar_mass / ((r_scale_m)**3)
    ρ_c_fit_SI = ρ_c_fit * (1e-3) * solar_mass / ((r_scale_m)**3)


    #-----------------------------------------
    # Analyzing Stability through dM_dρ_c
    #-----------------------------------------

    # Derivative of quintic polynomial fit
    def derv_quintic_fit(x, a, b, c, d, e):
        return 5*a*x**4 + 4*b*x**3 + 3*c*x**2 + 2*d*x + e

    # Compute dM/dρc
    dM_dρ_c_fit = derv_quintic_fit(ρ_c_fit, a, b, c, d, e)

    # Locate point where derivative is 0, i.e maximal M
    zero_crossing_idx = np.where((dM_dρ_c_fit[:-1] > 0) & (dM_dρ_c_fit[1:] < 0))[0]
    max_M = M_fit[zero_crossing_idx]
    max_M_ρ_c_SI = ρ_c_fit_SI[zero_crossing_idx]

    # Seperate stable and unstable regimes
    M_fit_stable = M_fit[:int(zero_crossing_idx)]
    ρ_c_fit_SI_stable = ρ_c_fit_SI[:int(zero_crossing_idx)]
    M_fit_unstable = M_fit[int(zero_crossing_idx):]
    ρ_c_fit_SI_unstable = ρ_c_fit_SI[int(zero_crossing_idx):]


    # Plot M vs ρc datapoints and stable and unstable fit curves, report maximal Mass
    plt.scatter(ρ_c_vals_SI, M_pc_vals, alpha = 0.3, color = "darkorchid", label = "Datapoints")
    plt.scatter(max_M_ρ_c_SI, max_M, marker= "x", lw=2, label = "Maximal M", color = "gold", zorder = 3)
    plt.plot(ρ_c_fit_SI_stable, M_fit_stable, color = "darkorchid", label = "Stable region")
    plt.plot(ρ_c_fit_SI_unstable, M_fit_unstable, color = "deeppink", label = "Unstable region")
    plt.title(rf'$M(\rho_c)$ vs $\rho_c$ plot for NSs (N = {num_points})')
    plt.xlabel(r'$\rho_c$ $\left(\frac{kg}{m^3}\right)$')
    plt.ylabel(r'$M(\rho_c) \quad (M \odot)$')
    plt.text(max_M_ρ_c_SI-5e17, max_M - 0.13,
    rf'$(\rho_c, M_{{max}}):$ ({float(max_M_ρ_c_SI):.2e}, {float(max_M):.3f})',
    bbox=dict(facecolor="white"), fontsize = 10)
    plt.legend()
    plt.show()

    # Plot dM/dρc
    plt.plot(ρ_c_fit_SI, dM_dρ_c_fit, color = "chocolate")
    plt.scatter(ρ_c_fit_SI[zero_crossing_idx], 0,marker= "x", color = "gold", zorder=3)
    plt.title(rf'$\frac{{dM}}{{d\rho_c}}$ vs $\rho_c$ plot for NSs (N = {num_points})')
    plt.xlabel(r'$\rho_c$ $\left(\frac{kg}{m^3}\right)$')
    plt.ylabel(r'$dM/d\rho_c$')
    plt.grid(True)
    plt.show()
    print()

    print('Einstein--Part C. \n\n\n')



    #=========================================================================
    # PART D
    #=========================================================================




    return

my_test_einstein()