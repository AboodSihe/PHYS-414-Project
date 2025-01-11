import numpy as np
from numpy import pi, log, sqrt, arcsinh, exp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import csv





# Useful constants.
G_SI = 6.67430e-11
avg_earth_rad = 6.37814e6
solar_mass = 1.988416e30

# _SI will denote that a quantity is in SI units, _MSRE will denote if a
# quantity is in Solar Mass-Earth Radius units.





def read_WD_data(filename): # (Part B)
    #READWDDATA This function reads WD data from a csv file, namely the base-10
    # logarithm of a WD's gravitional acceleration and its mass, then returns
    # those values in an array.

    # Initialize lists
    loggs = []
    masses = []

    # Read data
    with open(filename, mode ="r") as file:
        data = csv.DictReader(file)
        for lines in data:
            loggs.append(float(lines["logg"]))
            masses.append(float(lines["mass"]))

    # Convert to NumPy arrays
    loggs = np.array(loggs)
    masses = np.array(masses)

    return loggs, masses



def general_EOS(ρ, D, q, K): # (Part D)
    #GENERAlEOS Takes in a values for ρ, D, q, and K and computes the Pressure
    # using the general EOS and the relation between C, D, and K.
    x = (ρ / D)**(1 / q)
    C = (5/8)* K * D**(5 / q)

    P = C*(x * (2*x**2 - 3) * (sqrt(x**2 + 1)) + 3*arcsinh(x))
    return P



def solve_ivp_for_D(r, mpρ_arr, D, K, G): # (Parts D and E)
    # SOLVEIVPFORD This function receives relevant constants and initial
    # conditions and solves system of DEs relating the mass, pressure, density,
    # and radius of WDs outside the low-mass region where the density and
    # pressure are related using the general EOS. The integration is
    # made to stop when the pressure becomes 0.

    # Acquire initial conditions
    m, p, ρ  = mpρ_arr

    # Small epsilon to deal with division by 0 errors
    ε = 1e-9

    # Case for if r = 0
    if r == 0:
        dpdr = 0
        dmdr = 0
        dρdr = 0


    else:

        # Case for if p jumps below 0 (Caused by solve_ivp()'s stepping)
        if p < 0:
            p = 0

        # Define system of DEs
        dmdr = 4*pi*ρ*r**2
        dpdr = -(G*m*ρ) / r**2
        dρdr = ((-3*G*m*ρ**(1/3)) / (5*K*r**2 + ε)) * sqrt((ρ/D)**(2/3) + 1)

    return np.array([dmdr, dpdr, dρdr])



def p_reaches_0(r, mpρ_arr, D, K, G): # (Parts D and E)
    # PREACHES0 This function and the following lines of code act like a
    # signal to solve_ivp() to stop integration when p reaches 0. At that point,
    # we'd have reached the maximal radius of the WD.

    p = mpρ_arr[1]
    return p

p_reaches_0.terminal = True
p_reaches_0.direction = 0





def my_test_newton():
    """
    Main function for Newton.

    PART B: Reads the WD data .csv file, and records the loggs and masses of
    WDs. Performs a unit conversion of loggs to earth radii and plots these
    M(R) data points.

    PART C: Filters out WD datapoints with low mass. Performs a linear fit to
    determine q and n*. Solves the Lane-Emden equation with newly found n* to
    find ξn and θ'(ξn). Uses all these parameters to find K*. Finally computes
    ρ_c and plots it against low mass WD masses.

    PART D: For several values of D and several values of ρc, computes M-R
    values of WDs by solving the IVP in solve_ivp_for_D(). These points are
    then interpolated with a cubic spline and compared with a spline of the
    raw data to find the value D that gives the best fit. This value D is used
    to compute C and the results are presented.

    PART E: Using the optimal value of D found, the IVP is solved again for
    many values of ρc ranging from small to very large density values. This
    data is again interpolated with a cubic spline and the Chandrasekhar mass
    is found and presented.
    """



    #=========================================================================
    # PART B
    #=========================================================================
    print("Newton--Part B Start: \n")

    # Get WD Data
    loggs, masses = read_WD_data("white_dwarf_data.csv")


    # Convert to loggs to radii in units of R🜨.
    gs = np.power(10, loggs) / 100
    stellar_radii = np.sqrt(G_SI * masses * solar_mass / gs) / avg_earth_rad


    # Plotting
    plt.scatter(stellar_radii, masses, lw=0.1, alpha=0.4, color = "teal")
    plt.title(r"$M-R$ Data of WDs in Units $(M_\odot)$ & $(R_\oplus)$")
    plt.xlabel(r"Radius $(R_\oplus)$")
    plt.ylabel(r"Mass $(M_\odot)$")
    plt.grid(True)
    plt.show()
    print()

    print("Newton--Part B Done. \n\n\n")



    #=========================================================================
    # PART C
    #=========================================================================
    print("Newton--Part C Start: \n")

    # Plotting R(M) data points to determine low mass cutoff
    plt.scatter(masses, stellar_radii, lw=0.1, alpha=0.4, color = "teal")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(r"$R-M$ Data of WDs in $loglog$ Scale")
    plt.xlabel(r"Mass $(M_\odot)$")
    plt.ylabel(r"Radius $(R_\oplus)$")
    plt.grid(True)
    plt.show()

    # Low mass cutoff = 0.4 M☉, adjust mass and radius arrays.
    low_mass_idx = masses < 0.35
    low_masses = masses[low_mass_idx]
    low_mass_radii = stellar_radii[low_mass_idx]


    #--------------------------
    # Determining q and n*
    #--------------------------

    # Logarithm fit of M(R) equation to find q
    def M_R_eq_in_log(R, q, a):
        return log(a) + ((15 - 4*q)/ (5 - 2*q))*R

    # Find q by linear fitting, then find n*
    param_bounds = ([2.5, -np.inf], [3.75, np.inf])
    q_arr, pcov = curve_fit(M_R_eq_in_log, log(low_mass_radii), log(low_masses), bounds = param_bounds)

    q = int(round(q_arr[0]))

    n = q / (5 - q)


    #-------------------------------------------------------
    # Solving Lane-Emden Eq. to determine ξn and θ'(ξn)
    #-------------------------------------------------------

    # Define equation by splitting SODE to two FODEs
    def lane_emden_eq(ξ, θ):
        v1, v2 = θ

        if ξ == 0:
            dθdξ = [v2, 1]

        elif v1 < 0:
            dθdξ = [v2, 0]

        else:
            dθdξ = [v2, -(v1**n + (2/ξ) * v2)]

        return dθdξ

    # Initial parameters
    ξ_span = (0, 4)
    θ_0_conds = [1, 0]
    ξ_eval = np.linspace(*ξ_span, 2000)

    sol = solve_ivp(lane_emden_eq, ξ_span, θ_0_conds, "RK45", t_eval = ξ_eval,
                    atol = 1e-15, rtol = 1e-13)

    # Solutions, ξn defined such that θ(ξn) = 0
    ξn_idx = np.argmax(np.diff(np.sign(sol.y[0])) != 0)
    ξn = sol.t[ξn_idx]
    dθ_at_ξn = sol.y[1, ξn_idx]

    plt.plot(sol.t, sol.y[0], color = "forestgreen")
    plt.title(r"Lane-Emden $\theta (\xi)$ Solution for $n = $" + str(n))
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\theta (\xi)$")
    plt.scatter(ξn, 0, color="darkorange", label=r"$\xi_n$", zorder = 3)
    plt.text(ξn-0.3, 0.1, "({:.2f}, {:.2f})".format(ξn, 0))
    plt.legend()
    plt.grid(True)
    plt.show()


    #--------------------
    # Determining K*
    #--------------------

    # Convert G to M☉ and R🜨 units
    G_MSRE = G_SI * (solar_mass) / (avg_earth_rad**3)

    # Full M(R) equation
    def M_R_eq_full(R, K):
        return -4*pi * ((4*pi*G_MSRE) / ((n+1)*K))**(n / (1-n)) * ξn**((n+1) / (n-1)) * dθ_at_ξn * R**((3-n) / (1-n))

    # Solution of K* from fit
    K_arr, pcov = curve_fit(M_R_eq_full, low_mass_radii, low_masses)
    K_MSRE = K_arr[0]

    # Convert K* to SI units
    K_SI = K_MSRE * (avg_earth_rad**4) / (solar_mass**(2/3))


    #------------------------
    # Evaluating Results
    #------------------------

    # Plotting full fit-curve using n*, K*, ξn, and θ'(ξn)
    R_plt_points = np.linspace(1, 2.7, 100)
    M_plt_points = M_R_eq_full(R_plt_points, K_MSRE)

    plt.plot( R_plt_points, M_plt_points, lw = 2, color = "lightseagreen", label = "Fit")
    plt.scatter(stellar_radii, masses, lw=0.1, alpha=0.4, color = "teal", label = "True Data")
    plt.title(r"$M-R$ Data of WDs With Low-Mass Data Fitting Line")
    plt.xlabel(r"Radius $(R_\oplus)$")
    plt.ylabel(r"Mass $(M_\odot)$")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Present Results
    fit_params_names = ["n*", "K* (SI)","K* (MSRE)", "ξ_n", "θ'(ξ_n)"]
    fit_params_vals = np.round(np.array([n, K_SI, K_MSRE, ξn, dθ_at_ξn]),3)

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, (text, val) in enumerate(zip(fit_params_names, fit_params_vals)):
        rect = Rectangle((0.1, 0.8 - i * 0.17), 0.8, 0.12, edgecolor="black", facecolor="lightgrey", lw=2)
        ax.add_patch(rect)
        ax.text(0.5, 0.85 - i * 0.17, f"$\mathbf{{{text}}}$: {val}", fontsize=11, ha="center", va="center")
    plt.title(r"Fitting Parameters of WD $M(R)$ Equation")
    ax.axis("off")
    plt.show()


    #----------------------------------
    # Determining Central Density
    #----------------------------------

    # Calculating ρ_c
    ρ_c_MSRE = -(low_masses * ξn) / (4*pi*(low_mass_radii**3)*dθ_at_ξn)

    # Different formulations for ρ_c
    #ρ_c = ((low_mass_radii / ξn)**2 * ((4*pi*Gc) / ((n+1)*K))) ** ((1-n) / n)
    #ρ_c = ((-low_masses/(4*pi*dθ_at_ξn**2)) * ((4*pi*Gc) / (n+1)*K)**1.5) **(2*n / (3-n))

    ρ_c_SI = ρ_c_MSRE * (solar_mass / avg_earth_rad**3)

    # Plotting ρ_c vs m
    plt.scatter(low_masses, ρ_c_MSRE, lw=0.1, alpha=0.7, color = "palevioletred")
    plt.title(r"$\rho_c$ vs $M$ Data Points for Low-Mass WDs")
    plt.xlabel(r"Mass $(M_\odot)$")
    plt.ylabel(r"$\rho_c \quad \frac{(M_\odot)}{(R_\oplus)^3}$")
    plt.grid(True)
    plt.show()
    print()

    print("Newton--Part C Done. \n\n\n")


    #=========================================================================
    # PART D
    #=========================================================================
    print("Newton--Part D Start: \n")

    # Sort true radii and masses
    sorted_idx = np.argsort(stellar_radii)
    stellar_radii = stellar_radii[sorted_idx]
    masses = masses[sorted_idx]
    cont_r = np.arange(min(stellar_radii), max(stellar_radii), 0.01)

    # Construct "true" spline
    MR_true_spline = CubicSpline(stellar_radii[::4], masses[::4])

    # List of D values
    D_list_MSRE = [0.2, 0.8, 0.3775]
    colors = ["maroon", "lightcoral", "red"]

    # IVP Solving
    for n in  range(len(D_list_MSRE)):
        D_MSRE = D_list_MSRE[n]

        r_span = (0, 3) # Highest radius in data is ~2.5 R🜨

        # List of ρc that cover range well. (Number = 11)
        ρ_c_vals = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 10, 30])

        # Initialize end of integration value arrays.
        R_pc_vals = np.zeros(len(ρ_c_vals))
        M_pc_vals = np.zeros(len(ρ_c_vals))

        for i in range(len(ρ_c_vals)):

            # Record ρc and compute initial pressure
            ρ_c = ρ_c_vals[i]
            p_c = general_EOS(ρ_c, D_MSRE, q, K_MSRE)

            # Initial conditions array
            mpρ0 = [0, p_c, ρ_c]

            # Solve DE system, stopping when p reaches 0.
            sol = solve_ivp(solve_ivp_for_D, r_span, mpρ0,
            args = (D_MSRE, K_MSRE, G_MSRE), method="RK45", events = p_reaches_0, atol = 1e-10, rtol = 1e-8)

            # Record final radius and mass.
            R_pc = sol.t[-1]
            M_pc = sol.y[0, -1]

            # Update arrays
            R_pc_vals[i] = R_pc
            M_pc_vals[i] = M_pc

        # Sort "numerical" radii and masses and interpolate
        sorted_idx = np.argsort(R_pc_vals)
        R_pc_vals = R_pc_vals[sorted_idx]
        M_pc_vals = M_pc_vals[sorted_idx]
        MR_fit_spline = CubicSpline(R_pc_vals, M_pc_vals)

        # Add to plot
        plt.scatter(R_pc_vals, M_pc_vals, s = 10, color = colors[n], zorder = n+2)
        plt.plot(cont_r, MR_fit_spline(cont_r), zorder = n+2, color = colors[n], label = rf"$D_{{MSRE}}$ = {D_MSRE} Spline")


    # Add "true" spline and Plot
    plt.plot(cont_r, MR_true_spline(cont_r), color = "teal", lw= 4, zorder = 1, label = "True Data Spline")
    plt.title("True $M-R$ Curve Against Splines of IVP Solutions\nfor Different Values $D$")
    plt.xlabel(r"Radius $(R_\oplus)$")
    plt.ylabel(r"Mass $(M_\odot)$")
    plt.legend()
    plt.show()


    # Define 'optimal' D and use it to compute C
    D_MSRE = D_list_MSRE[-1]
    D_SI = D_MSRE * (solar_mass / avg_earth_rad**3)

    C_MSRE = (5/8) * K_MSRE * D_MSRE**(5/q)
    C_SI = (5/8) * K_SI * D_SI**(5/q)

    fit_params_names = ["D (MSRE)", "D (SI)","C (MSRE)", "C (SI)"]
    fit_params_vals = np.round(np.array([D_MSRE, D_SI, C_MSRE, C_SI]),5)

    # Present results
    fig, ax = plt.subplots(figsize=(6, 4))

    for i, (text, val) in enumerate(zip(fit_params_names, fit_params_vals)):
        rect = Rectangle((0.1, 0.75 - i * 0.2), 0.8, 0.14, edgecolor="black", facecolor="lightgrey", lw=2)
        ax.add_patch(rect)
        ax.text(0.5, 0.825 - i * 0.2, f"$\mathbf{{{text}}}$: {val}", fontsize=12, ha="center", va="center")
    plt.title(r"Numerical Values for $D$ and $C$")
    ax.axis("off")
    plt.show()
    print()

    print("Newton--Part D Done. \n\n\n")



    #=========================================================================
    # PART E
    #=========================================================================
    print("Newton--Part E Start: \n")

    # List of ρc that ranges over small to very large values
    log_vals = np.linspace(log(0.01), log(5000), 50)
    ρ_c_vals = exp(log_vals)

    # Same procedure as Part D
    D_MSRE = 0.3775
    r_span = (0, 3)

    R_pc_vals = np.zeros(len(ρ_c_vals))
    M_pc_vals = np.zeros(len(ρ_c_vals))

    for i in range(len(ρ_c_vals)):

        ρ_c = ρ_c_vals[i]
        p_c = general_EOS(ρ_c, D_MSRE, q, K_MSRE)

        mpρ0 = [0, p_c, ρ_c]

        sol = solve_ivp(solve_ivp_for_D, r_span, mpρ0,
        args = (D_MSRE, K_MSRE, G_MSRE), method="RK45", events = p_reaches_0, atol = 1e-10, rtol = 1e-8)

        R_pc = sol.t[-1]
        M_pc = sol.y[0, -1]

        R_pc_vals[i] = R_pc
        M_pc_vals[i] = M_pc


    sorted_idx = np.argsort(R_pc_vals)
    R_pc_vals = R_pc_vals[sorted_idx]
    M_pc_vals = M_pc_vals[sorted_idx]
    MR_fit_spline = CubicSpline(R_pc_vals, M_pc_vals)

    # Chandrasekhar Mass at r → 0, ρc → ∞
    chandrasekhar_mass = round(float(MR_fit_spline(0)), 4)

    # Plot and present result
    plt.scatter(R_pc_vals, M_pc_vals, s = 3, color = colors[n], zorder = n+2, alpha = 0.5)
    plt.plot(cont_r, MR_fit_spline(cont_r), zorder = n+2, color = "red")
    plt.hlines(chandrasekhar_mass, 0, cont_r[-1], color="teal", linestyle ="--", lw = 1.2)
    plt.text(1.7, 1.35, fr"$M_{{Ch}} =$ {chandrasekhar_mass} $M_\odot$", bbox=dict(facecolor="white"))
    plt.title("Numerical $M-R$ Curve")
    plt.xlabel(r"Radius $(R_\oplus)$")
    plt.ylabel(r"Mass $(M_\odot)$")
    plt.grid(True)
    plt.show()
    print()

    print("Newton--Part E Done. \n\n\n")



    #=========================================================================
    # END
    #=========================================================================
    return


my_test_newton()