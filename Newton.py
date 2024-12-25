import numpy as np
from numpy import pi, log
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import csv





G = 6.67430e-11
avg_earth_rad = 6371e3
solar_mass = 1.988416e30
M_sun = "(M\u2609)"
R_earth = "(R\u2295)"





def read_WD_data(filename):
    
    loggs = []
    masses = []
    
    
    with open(filename, mode ='r') as file:    
        data = csv.DictReader(file)
        for lines in data:
            loggs.append(float(lines["logg"]))
            masses.append(float(lines["mass"]))
            
    loggs = np.array(loggs)
    masses = np.array(masses)        
    return loggs, masses





def my_test_newton():
    
    
    
    
    
    
    
    #=========================================================================
    # PART B
    #=========================================================================    
    
    
    loggs, masses = read_WD_data("white_dwarf_data.csv")
    
    gs = np.power(10, loggs) / 100
    
    stellar_radii = np.sqrt(G * masses * solar_mass / gs) / avg_earth_rad
    
    
    plt.scatter(stellar_radii, masses, lw=0.1, alpha=0.4, color = "teal")
    plt.title("M~R Data of Low Mass WDs in " + M_sun + " & " + R_earth + " units")
    plt.xlabel("Radius " + R_earth)
    plt.ylabel("Mass " + M_sun)
    plt.grid(True)
    plt.show()
    
    
    #=========================================================================
    # PART B
    #=========================================================================       

    plt.scatter(masses, stellar_radii, lw=0.1, alpha=0.4, color = "teal")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("M~R Data of Low Mass WDs in " + M_sun + " & " + R_earth + " units")
    plt.xlabel("Mass " + M_sun)
    plt.ylabel("Radius " + R_earth)
    plt.grid(True)
    plt.show()
    
    
    low_mass_idx = masses < 0.4
    low_masses = masses[low_mass_idx]
    low_mass_radii = stellar_radii[low_mass_idx]
    
    
    
    def M_R_eq_in_log(R, q, a):
        return log(a) + ((15 - 4*q)/ (5 - 2*q))*R
    
    
    """
    def mass_radius_eq(R, n, K):
        return R**((3-n) / (1-n)) * (pi**(2/(n-1))) * 4*pi * ((4*pi*G) / (n+1)*K )**(n/(1-n))
    """
    
    param_bounds = ([2.5, -np.inf], [3.75, np.inf])
    
    popt, pcov = curve_fit(M_R_eq_in_log, log(low_mass_radii), log(low_masses), bounds = param_bounds)

    q = int(round(popt[0]))
    
    n = q / (5 - q)
    
    def lane_emden_eq(xi, theta):
        v1, v2 = theta
        
        if abs(xi) < 1e-15:
            dtheta_dxi = [v2, 1]
            
        elif v1 < 1e-15:
            dtheta_dxi = [v2, 0]
            
        else:    
            dtheta_dxi = [v2, -(v1**n + (2/xi) * v2)]
            
        return dtheta_dxi

    
    xi_span = (0, 4)
    xi_0 = [1, 0]
    xi_eval = np.linspace(*xi_span, 500)
    
    sol = solve_ivp(lane_emden_eq, xi_span, xi_0, 'RK45', t_eval = xi_eval)
    
    #print(sol.t)
    idx = abs(sol.y[0]) < 1e-3
    print(sol.t[idx])
    print(sol.y[1, idx])
    plt.plot(sol.t, sol.y[0])
    plt.grid(True)


    
    
    
    return


my_test_newton()