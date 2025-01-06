import numpy as np
from numpy import pi, log
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import csv




# Useful constants.
G_SI = 6.67430e-11
avg_earth_rad = 6.37814e6
solar_mass = 1.988416e30

# _SI will denote that a quantity is in SI units, _MSRE will denote if a
# quantity is in Solar Mass-Earth Radius units.





def read_WD_data(filename):
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

    """



    #=========================================================================
    # PART B
    #=========================================================================
    print("Newton--Part B Start: \n")

    # Get WD Data
    loggs, masses = read_WD_data("white_dwarf_data.csv")


    # Convert to loggs to radii in units of average earth radius.
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
    low_mass_idx = masses < 0.4
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
    plt.scatter(ξn, 0, color="darkorange", label=r"$\xi_n$")
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

    plt.plot( R_plt_points, M_plt_points, lw = 2, color = "darkred", label = "Fit")
    plt.scatter(stellar_radii, masses, lw=0.1, alpha=0.4, color = "teal", label = "True Data")
    plt.title(r"$M-R$ Data of WDs With Low-Mass Data Fitting Line")
    plt.xlabel(r"Radius $(R_\oplus)$")
    plt.ylabel(r"Mass $(M_\odot)$")
    plt.legend()
    plt.grid(True)
    plt.show()


    fit_params_names = ["n*", "K* (SI)","K* (MSRE)", "ξ_n", "θ'(ξ_n)"]
    fit_params_vals = np.round(np.array([n, K_SI, K_MSRE, ξn, dθ_at_ξn]),3)

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, (text, val) in enumerate(zip(fit_params_names, fit_params_vals)):
        # Draw a rectangle
        rect = Rectangle((0.1, 0.8 - i * 0.17), 0.8, 0.12, edgecolor="black", facecolor="lightgrey", lw=2)
        ax.add_patch(rect)
        # Add text inside the rectangle
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








    return


my_test_newton()