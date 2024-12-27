import numpy as np
from numpy import pi, log
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import csv




# Useful constants.
G = 6.67430e-11
avg_earth_rad = 6371e3
solar_mass = 1.988416e30
M_sun = "(M\u2609)"
R_earth = "(R\u2295)"





def read_WD_data(filename):
    #READWDDATA This function reads WD data from a csv file, namely the base-10
    # logarithm of a WD's gravitional acceleration and its mass, then returns
    # those values in an array.
    
    # Initialize lists
    loggs = []
    masses = []
    
    # Read data
    with open(filename, mode ='r') as file:    
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
    
    PART C: Starts with a guess for the parameters dH and oM (omega_M) and
    performs a nonlinear fit by solving a nonlinear root finding problem as 
    defined in Eq. (3.1.22). This allows us to find the best fit values for dH
    and oM. Additionally, a plot of this nonlinear best fit line with the found
    parameters is presented.
    
    """    
    
    
    
    
    
    
    #=========================================================================
    # PART B
    #=========================================================================    
    print('Newton--Part B Start: \n')
    
    # Get WD Data
    loggs, masses = read_WD_data("white_dwarf_data.csv")
    
    
    # Convert to loggs to radii in units of average earth radius.
    gs = np.power(10, loggs) / 100
    stellar_radii = np.sqrt(G * masses * solar_mass / gs) / avg_earth_rad
    
    
    # Plotting
    plt.scatter(stellar_radii, masses, lw=0.1, alpha=0.4, color = "teal")
    plt.title("M~R Data of WDs in units " + M_sun + " & " + R_earth)
    plt.xlabel("Radius " + R_earth)
    plt.ylabel("Mass " + M_sun)
    plt.grid(True)
    plt.show()
    print()    

    print('Newton--Part B Done. \n\n\n')    
    
        
        
    #=========================================================================
    # PART C
    #=========================================================================       
    print('Newton--Part C Start: \n')
    
    # Plotting R(M) data points to determine low mass cutoff
    plt.scatter(masses, stellar_radii, lw=0.1, alpha=0.4, color = "teal")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("R~M Data of WDs in log-log scale")
    plt.xlabel("Mass " + M_sun)
    plt.ylabel("Radius " + R_earth)
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
    
    sol = solve_ivp(lane_emden_eq, ξ_span, θ_0_conds, 'RK45', t_eval = ξ_eval,
                    atol = 1e-15, rtol = 1e-13)
    
    # Solutions, ξn defined such that θ(ξn) = 0
    ξn_idx = np.argmax(np.diff(np.sign(sol.y[0])) != 0)
    ξn = sol.t[ξn_idx]
    dθ_at_ξn = sol.y[1, ξn_idx]
    
    
    #--------------------
    # Determining K*
    #--------------------   
    
    # Full M(R) equation
    def M_R_eq_full(R, K):
        return -4*pi * ((4*pi*G) / (n+1)*K)**(n / (1-n)) * ξn**((n+1) / (n-1)) * dθ_at_ξn * R**((3-n) / (1-n))
    
    # Solution of K* from fit
    K_arr, pcov = curve_fit(M_R_eq_full, low_mass_radii, low_masses)
    K = K_arr[0]
    
    
    #------------------------
    # Evaluating Results
    #------------------------   
    
    # Plotting full fit-curve using n*, K*, ξn, and θ'(ξn)
    R_plt_points = np.linspace(1, 2.7, 100)
    M_plt_points = M_R_eq_full(R_plt_points, K)
    
    plt.plot( R_plt_points, M_plt_points, lw = 2, color = "darkred", label = "Fit")
    plt.scatter(stellar_radii, masses, lw=0.1, alpha=0.4, color = "teal", label = "True Data")
    plt.title("M~R Data of WDs with low-mass data fitting line")
    plt.xlabel("Radius " + R_earth)
    plt.ylabel("Mass " + M_sun)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    fit_params_names = ["n*", "K*", "ξ_n", "θ'(ξ_n)"]
    fit_params_vals = np.round(np.array([n, K, ξn, dθ_at_ξn]), 3)
    
    fig, ax = plt.subplots(figsize=(6, 4))

    for i, (text, val) in enumerate(zip(fit_params_names, fit_params_vals)):
        # Draw a rectangle
        rect = Rectangle((0.1, 0.7 - i * 0.2), 0.8, 0.15, edgecolor='black', facecolor='lightgrey', lw=2)
        ax.add_patch(rect)
        # Add text inside the rectangle
        ax.text(0.5, 0.775 - i * 0.2, f'$\mathbf{{{text}}}$: {val}', fontsize=12, ha='center', va='center')
    plt.title("Fitting parameters of WD M(R) equation")
    ax.axis('off')
    plt.show()
    
    
    #----------------------------------
    # Determining Central Density (Not done)
    #----------------------------------    
    
    rho_c = ((low_mass_radii / ξn)**2 * ((4*pi*G) / (n+1)*K)) ** ((1-n) / n)
    
    #rho_c = ((-low_masses/(4*pi*dθ_at_ξn**2)) * ((4*pi*G) / (n+1)*K)**1.5) **(2*n / (3-n))
    
    rho_c *= (solar_mass / avg_earth_rad**3)
    plt.scatter(rho_c, low_masses)
    plt.show()
    print()    

    print('Newton--Part C Done. \n\n\n')    
    
    
    
    #=========================================================================
    # PART D
    #=========================================================================    
    
    
    
    
    
    
    
    
    return


my_test_newton()