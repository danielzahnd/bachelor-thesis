"""
Author: Daniel Zahnd
Created: 06.07.2022
Last modified: 26.07.2022
"""


"INITIALIZATION"


# Import python libraries
import math as m
from scipy import optimize
import numpy as np
import time as t
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Start program runtime measurement
start_time = t.time()

# Initialize global variables
global x_p_true
global y_p_true # True position coordinates of object
global z_p_true

global x_v_true
global y_v_true # True velocity coordinates of object
global z_v_true

global x_p_min, x_p_max, x_v_min, x_v_max
global y_p_min, y_p_max, y_v_min, y_v_max # Minimum and maximum values for coordinates
global z_p_min, z_p_max, z_v_min, z_v_max

global x_1, y_1, z_1
global x_2, y_2, z_2
global x_3, y_3, z_3 # Receiver coordinates
global x_4, y_4, z_4 # Emitter is assumed to be at orignin of coordinate system
global x_5, y_5, z_5
global x_6, y_6, z_6

global del_f1
global del_f2
global del_f3 # Frequency shifts at receiver station
global del_f4 # Emitter is assumed to be at orignin of coordinate system
global del_f5
global del_f6

global a # Baseline lenght of measurement facility geometry triangle


"FUNCTION DEFINITIONS"


# Nonlinear equation system
def eqsyst(f):
	x_p = f[0] # x-position of object
	y_p = f[1] # y-position of object
	z_p = f[2] # z-position of object
	x_v = f[3] # x-velocity of object
	y_v = f[4] # y-velocity of object
	z_v = f[5] # z-velocity of object
	F = np.empty(6)
	F[0] = (x_v*x_p+y_v*y_p+z_v*z_p)/(m.sqrt(x_p**2+y_p**2+z_p**2)) + (x_v*(x_p-x_1)+y_v*(y_p-y_1)+z_v*(z_p-z_1))*((x_p-x_1)**2+(y_p-y_1)**2+(z_p-z_1)**2)**(-1/2)-del_f1
	F[1] = (x_v*x_p+y_v*y_p+z_v*z_p)/(m.sqrt(x_p**2+y_p**2+z_p**2)) + (x_v*(x_p-x_2)+y_v*(y_p-y_2)+z_v*(z_p-z_2))*((x_p-x_2)**2+(y_p-y_2)**2+(z_p-z_2)**2)**(-1/2)-del_f2
	F[2] = (x_v*x_p+y_v*y_p+z_v*z_p)/(m.sqrt(x_p**2+y_p**2+z_p**2)) + (x_v*(x_p-x_3)+y_v*(y_p-y_3)+z_v*(z_p-z_3))*((x_p-x_3)**2+(y_p-y_3)**2+(z_p-z_3)**2)**(-1/2)-del_f3
	F[3] = (x_v*x_p+y_v*y_p+z_v*z_p)/(m.sqrt(x_p**2+y_p**2+z_p**2)) + (x_v*(x_p-x_4)+y_v*(y_p-y_4)+z_v*(z_p-z_4))*((x_p-x_4)**2+(y_p-y_4)**2+(z_p-z_4)**2)**(-1/2)-del_f4
	F[4] = (x_v*x_p+y_v*y_p+z_v*z_p)/(m.sqrt(x_p**2+y_p**2+z_p**2)) + (x_v*(x_p-x_5)+y_v*(y_p-y_5)+z_v*(z_p-z_5))*((x_p-x_5)**2+(y_p-y_5)**2+(z_p-z_5)**2)**(-1/2)-del_f5
	F[5] = (x_v*x_p+y_v*y_p+z_v*z_p)/(m.sqrt(x_p**2+y_p**2+z_p**2)) + (x_v*(x_p-x_6)+y_v*(y_p-y_6)+z_v*(z_p-z_6))*((x_p-x_6)**2+(y_p-y_6)**2+(z_p-z_6)**2)**(-1/2)-del_f6
	return F

# Solution seeker function
def solseeker(z_vel, div):
	half = int(div/2)
	x_p = np.linspace(x_p_min, x_p_max, num=div, endpoint=True)
	y_p = np.linspace(y_p_min, y_p_max, num=div, endpoint=True)
	z_p = np.linspace(z_p_min, z_p_max, num=div, endpoint=True)
	x_v_neg = np.linspace(-x_v_max,-x_v_min, num=half, endpoint=True)
	x_v_pos = np.linspace(x_v_min,x_v_max, num=half, endpoint=True)
	x_v = np.hstack((x_v_neg,x_v_pos))
	y_v_neg = np.linspace(-y_v_max,-y_v_min, num=half, endpoint=True)
	y_v_pos = np.linspace(y_v_min,y_v_max, num=half, endpoint=True)
	y_v = np.hstack((y_v_neg,y_v_pos))
	z_v =  z_vel# Should be assumed to be near zero if object orbit is nearly circular
	solutions = np.empty((0,12)) # Solutions array [solution/initial guess]
	it_number = int(0)
	perc_number = int(0)
	print(0, '% progress (initial solving)')
	for i in range(0,len(x_p)):
		for j in range(0,len(y_p)):
			for k in range(0,len(z_p)):
				for l in range(0,len(x_v)):
					for n in range(0,len(y_v)):
						guess = np.array([x_p[i],y_p[j],z_p[k],x_v[l],x_v[n],z_v])
						sol = optimize.root(eqsyst, guess, method='hybr') #, options={'maxiter':10000, 'fatol':1e-9})
						if sol.success == True:
							# print("Solution found")
							sol_entry = np.concatenate((sol.x,guess))
							solutions = np.vstack((solutions,sol_entry))
						it_number = it_number + 1
						if it_number % (int((div**5)/10)) == 0:
							perc_number = perc_number + 1
							print(perc_number*10, '% progress (initial solving)')
	if len(solutions) == 0:
		print("No solutions were found at initial solving.")
		return 
	else:
		ind_bestsol = bestsol(solutions)
		bestsolution = solutions[ind_bestsol,:]
		return bestsolution

# Search for the index of the best solutions in terms of residues
def bestsol(sols):
	(solutions,guesses) = np.hsplit(sols,2)
	res = np.array([])
	rows, columns = solutions.shape
	for i in range(0,rows):
		val = np.linalg.norm(eqsyst(solutions[i,:]))
		res = np.append(res,val)
	index = np.argmin(res)
	return index

# Seek for more exact convergences in environment of best first solution
def refiner1(z_vel, sol, diff, div):
	if sol is None:
		return
	x_p = np.linspace(sol[0]-diff, sol[0]+diff, num=div, endpoint=True)
	y_p = np.linspace(sol[1]-diff, sol[1]+diff, num=div, endpoint=True)
	z_p = np.linspace(sol[2]-diff, sol[2]+diff, num=div, endpoint=True)
	x_v = np.linspace(sol[3]-diff, sol[3]+diff, num=div, endpoint=True)
	y_v = np.linspace(sol[4]-diff, sol[4]+diff, num=div, endpoint=True)
	z_v = z_vel # Should be assumed to be near zero if object orbit is nearly circular
	solutions = np.empty((0,12)) # Solutions array [solution/initial guess]
	it_number = int(0)
	perc_number = int(0)
	print(0, '% progress (solution refining)')
	for i in range(0,len(x_p)):
		for j in range(0,len(y_p)):
			for k in range(0,len(z_p)):
				for l in range(0,len(x_v)):
					for n in range(0,len(y_v)):
						guess = np.array([x_p[i],y_p[j],z_p[k],x_v[l],y_v[n],z_v])
						solut = optimize.root(eqsyst, guess, options={'xtol':1e-10})
						if solut.success == True:
							# print("Solution found")
							sol_entry = np.concatenate((solut.x,guess))
							solutions = np.vstack((solutions,sol_entry))
						it_number = it_number + 1
						if it_number % (int((div**5)/10)) == 0:
							perc_number = perc_number + 1
							print(perc_number*10, '% progress (solution refining)')
	if len(solutions) == 0:
		print("No solutions were found at solution refining.")
		return 
	else:
		ind_bestsol = bestsol(solutions)
		solu = solutions[ind_bestsol,:]
		ressolu = np.linalg.norm(eqsyst(solu))
		ressol = np.linalg.norm(eqsyst(sol))
		if ressolu < ressol:
			return solu
		else:
			return sol
		
# Seek for more exact convergences in environment of best refined solution
def refiner2(sol, diff, div):
	if sol is None:
		return
	x_p = np.linspace(sol[0]-diff, sol[0]+diff, num=div, endpoint=True)
	y_p = np.linspace(sol[1]-diff, sol[1]+diff, num=div, endpoint=True)
	z_p = np.linspace(sol[2]-diff, sol[2]+diff, num=div, endpoint=True)
	x_v = sol[3]
	y_v = sol[4]
	z_v = sol[5]
	solutions = np.empty((0,12)) # Solutions array [solution/initial guess]
	it_number = int(0)
	perc_number = int(0)
	print(0, '% progress (solution refining)')
	for i in range(0,len(x_p)):
		for j in range(0,len(y_p)):
			for k in range(0,len(z_p)):
				guess = np.array([x_p[i],y_p[j],z_p[k],x_v,y_v,z_v])
				solut = optimize.root(eqsyst, guess, options={'xtol':1e-10})
				if solut.success == True:
					# print("Solution found")
					sol_entry = np.concatenate((solut.x,guess))
					solutions = np.vstack((solutions,sol_entry))
				it_number = it_number + 1
				if it_number % (int((div**3)/10)) == 0:
					perc_number = perc_number + 1
					print(perc_number*10, '% progress (solution refining)')
	if len(solutions) == 0:
		print("No solutions were found at solution refining.")
		return 
	else:
		ind_bestsol = bestsol(solutions)
		solu = solutions[ind_bestsol,:]
		ressolu = np.linalg.norm(eqsyst(solu))
		ressol = np.linalg.norm(eqsyst(sol))
		if ressolu < ressol:
			return solu
		else:
			return sol
		
# Printer, prints results of iteration
def printer(sol,descr):
	if sol is None:
		return
	else:
		(prin,noprin) = np.hsplit(sol,2)
		totvel = m.sqrt(x_v_true**2 + y_v_true**2 + z_v_true**2)
		print('\n\n')
		print('**************', descr, '**************')
		print('x_{S,c} = ', round(prin[0],3), 'm')
		print('y_{S,c} = ', round(prin[1],3), 'm')
		print('z_{S,c} = ', round(prin[2],3), 'm')
		print('\dot{x}_{S,c} = ', round(prin[3],3), 'm/s')
		print('\dot{y}_{S,c} = ', round(prin[4],3), 'm/s')
		print('\dot{z}_{S,c} = ', round(prin[5],3), 'm/s')
		print('\Delta x_{S,c} = ', round(abs(prin[0]-x_p_true),3), 'm   or  ', round((abs(sol[0]-x_p_true)/a)*100,3), '% of baseline')
		print('\Delta y_{S,c} = ', round(abs(prin[1]-y_p_true),3), 'm   or  ', round((abs(sol[1]-y_p_true)/a)*100,3), '% of baseline')
		print('\Delta z_{S,c} = ', round(abs(prin[2]-z_p_true),3), 'm   or  ', round((abs(sol[2]-z_p_true)/a)*100,3), '% of baseline')
		print('\Delta \dot{x}_{S,c} = ', round(abs(prin[3]-x_v_true),3), 'm/s   or  ', round((abs(sol[3]-x_v_true)/totvel)*100,3), '% of total velocity')
		print('\Delta \dot{y}_{S,c} = ', round(abs(prin[4]-y_v_true),3), 'm/s   or  ', round((abs(sol[4]-y_v_true)/totvel)*100,3), '% of total velocity')
		print('\Delta \dot{z}_{S,c} = ', round(abs(prin[5]-z_v_true),3), 'm/s   or  ', round((abs(sol[5]-z_v_true)/totvel)*100,3), '% of total velocity')
		print('****************************************************************\n')
		return
	
# Callback function to write out iteration information
def examiner(sol,residue):
	global iterstep
	global iterations
	iterst = np.array([iterstep])
	appnd = np.concatenate((iterst,sol))
	iterations = np.vstack((iterations,appnd))
	iterstep = iterstep + 1

		
"MAIN PROGRAM"


# Define Constants
M_E = 5.972e24 # Mass of earth
G = 6.67430e-11 # Gravitational constant
R_E = 6.371e6 # Radius of spherical earth

# Define geometry of measurement device
a = 1*1.0e5 # Edge lenght of triangle (baseline)
h = (m.sqrt(3)/2)*a # Height of triangle

# Position of receiver R_1
x_1 = -a
y_1 = 0.0
z_1 = 0.0
# Position of receiver R_2
x_2 = -a/2
y_2 = -h 
z_2 = 0.0
# Position of receiver R_3
x_3 = a/2
y_3 = -h 
z_3 = 0.0
# Position of receiver R_4
x_4 = a
y_4 = 0.0 
z_4 = 0.0
# Position of receiver R_5
x_5 = a/2
y_5 = h 
z_5 = 0.0
# Position of receiver R_6
x_6 = -a/2
y_6 = h 
z_6 = 0.0

"""*********************  SIMULATED VALUES  **********************"""
"""***************************************************************"""
# True position coordinates of object
true_vector = np.array([8.3e4, -1.4e3, 1.9e5, 7.8e3, -6.9e3, -1.1e2])
x_p_true = true_vector[0] # True x-coordinate of object
y_p_true = true_vector[1] # True y-coordinate of object
z_p_true = true_vector[2] # True z-coordinate of object 
	
# True velocity coordinates of object 
x_v_true = true_vector[3] # True x-velocity of object
y_v_true = true_vector[4] # True y-velocity of object
z_v_true = true_vector[5] # True z-velocity of object

"""***************************************************************"""
"""***************************************************************"""

# Calculate expected frequency shifts at receiver stations
del_f1 = (x_v_true*x_p_true+y_v_true*y_p_true+z_v_true*z_p_true)/(m.sqrt(x_p_true**2+y_p_true**2+z_p_true**2)) + (x_v_true*(x_p_true-x_1)+y_v_true*(y_p_true-y_1)+z_v_true*(z_p_true-z_1))/(m.sqrt((x_p_true-x_1)**2+(y_p_true-y_1)**2+(z_p_true-z_1)**2))
del_f2 = (x_v_true*x_p_true+y_v_true*y_p_true+z_v_true*z_p_true)/(m.sqrt(x_p_true**2+y_p_true**2+z_p_true**2)) + (x_v_true*(x_p_true-x_2)+y_v_true*(y_p_true-y_2)+z_v_true*(z_p_true-z_2))/(m.sqrt((x_p_true-x_2)**2+(y_p_true-y_2)**2+(z_p_true-z_2)**2))
del_f3 = (x_v_true*x_p_true+y_v_true*y_p_true+z_v_true*z_p_true)/(m.sqrt(x_p_true**2+y_p_true**2+z_p_true**2)) + (x_v_true*(x_p_true-x_3)+y_v_true*(y_p_true-y_3)+z_v_true*(z_p_true-z_3))/(m.sqrt((x_p_true-x_3)**2+(y_p_true-y_3)**2+(z_p_true-z_3)**2))
del_f4 = (x_v_true*x_p_true+y_v_true*y_p_true+z_v_true*z_p_true)/(m.sqrt(x_p_true**2+y_p_true**2+z_p_true**2)) + (x_v_true*(x_p_true-x_4)+y_v_true*(y_p_true-y_4)+z_v_true*(z_p_true-z_4))/(m.sqrt((x_p_true-x_4)**2+(y_p_true-y_4)**2+(z_p_true-z_4)**2))
del_f5 = (x_v_true*x_p_true+y_v_true*y_p_true+z_v_true*z_p_true)/(m.sqrt(x_p_true**2+y_p_true**2+z_p_true**2)) + (x_v_true*(x_p_true-x_5)+y_v_true*(y_p_true-y_5)+z_v_true*(z_p_true-z_5))/(m.sqrt((x_p_true-x_5)**2+(y_p_true-y_5)**2+(z_p_true-z_5)**2))
del_f6 = (x_v_true*x_p_true+y_v_true*y_p_true+z_v_true*z_p_true)/(m.sqrt(x_p_true**2+y_p_true**2+z_p_true**2)) + (x_v_true*(x_p_true-x_6)+y_v_true*(y_p_true-y_6)+z_v_true*(z_p_true-z_6))/(m.sqrt((x_p_true-x_6)**2+(y_p_true-y_6)**2+(z_p_true-z_6)**2))

# Define boundaries of possible values for a solution (dependant upon geometry of measurement facility)
x_p_min = -a
x_p_max = a
y_p_min = -a
y_p_max = a
z_p_min = 1.6e5 # Minimum for LEO (accessible via doppler radar)
z_p_max = 2.0e6 # Maximum for LEO (accessible via doppler radar)
x_v_min = m.sqrt((G*M_E)/(R_E+z_p_max)) # Of order 10^3, approx. 6.9e3
x_v_max = m.sqrt((G*M_E)/(R_E+z_p_min)) # Of order 10^3, approx. 7.8e3
y_v_min = m.sqrt((G*M_E)/(R_E+z_p_max)) # Of order 10^3, approx. 6.9e3
y_v_max = m.sqrt((G*M_E)/(R_E+z_p_min)) # Of order 10^3, approx. 7.8e3
z_v_min = -2.0e2 # Reasonable assumption for almost circular orbit
z_v_max = 2.0e2 # Reasonable assumption for almost circular orbit

"""**********************  FUNCTION CALLS  ***********************"""
"""***************************************************************"""
# Run solution seeker, returns best initial solution
solution = solseeker(0.0, 4)
sol_plot = solution
printer(solution,'Initial solution')

# Run solution refiner, returns refined solution if better than initial solution
solution = refiner1(0.0, solution, 1e3, 4)
printer(solution,'1st ref. solution')

# Run solution refiner, returns refined solution if better than refined solution
solution = refiner2(solution, 5e2, 8)
printer(solution,'2nd ref. solution')

# Run solution refiner, returns refined solution if better than refined solution
solution = refiner2(solution, 1e2, 10)
printer(solution,'3rd ref. solution')

# Run solution refiner, returns refined solution if better than refined solution
solution = refiner1(0.0, solution, 5, 4)
printer(solution,'4th ref. solution')

"""***************************************************************"""
"""***************************************************************"""

# # Plotting of convergence behaviours for initial solving
# iterstep = 1
# iterations = np.empty((0,7))
# (sol_plot_sol, sol_plot_guess) = np.hsplit(sol_plot,2)
# printsol = optimize.root(eqsyst, sol_plot_guess, method='broyden1', callback=examiner)
# firstline = np.concatenate(([0],sol_plot_guess))
# iterations = np.vstack((firstline,iterations))

# itdata = iterations[:,0]
# xpdata = iterations[:,1]
# truexpdata = np.full(len(itdata), x_p_true)
# ypdata = iterations[:,2]
# trueypdata = np.full(len(itdata), y_p_true)
# zpdata = iterations[:,3]
# truezpdata = np.full(len(itdata), z_p_true)
# xvdata = iterations[:,4]
# truexvdata = np.full(len(itdata), x_v_true)
# yvdata = iterations[:,5]
# trueyvdata = np.full(len(itdata), y_v_true)
# zvdata = iterations[:,6]
# truezvdata = np.full(len(itdata), z_v_true)

# fig1 = plt.figure(figsize=(10,7))
# plt.title("\nKonvergenzverhalten von $x_{S,c}$\n", fontsize=25)
# plt.plot(itdata, xpdata, 'b', label='Iterationsfortschritt', linewidth=1.5)
# plt.plot(itdata, truexpdata, 'r', label='Wahrer Wert', linewidth=0.75)
# plt.legend(loc = 'lower right', fontsize=18)
# plt.xlabel('Anzahl Iterationen', fontsize=20)
# plt.ylabel('$x_{S,c}$ [m]', fontsize=20)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.grid(True)
# plt.savefig('conv_xp.png', dpi=600, bbox_inches='tight')
# plt.show()
# plt.close(fig1)
# 		  
# fig2 = plt.figure(figsize=(10,7))
# plt.title("\nKonvergenzverhalten von $y_{S,c}$\n", fontsize=25)
# plt.plot(itdata, ypdata, 'b', label='Iterationsfortschritt', linewidth=1.5)
# plt.plot(itdata, trueypdata, 'r', label='Wahrer Wert', linewidth=0.75)
# plt.legend(loc = 'lower right', fontsize=18)
# plt.xlabel('Anzahl Iterationen', fontsize=20)
# plt.ylabel('$y_{S,c}$ [m]', fontsize=20)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.grid(True)
# plt.savefig('conv_yp.png', dpi=600, bbox_inches='tight')
# plt.show()
# plt.close(fig2)

# fig3 = plt.figure(figsize=(10,7))
# plt.title("\nKonvergenzverhalten von $z_{S,c}$\n", fontsize=25)
# plt.plot(itdata, zpdata, 'b', label='Iterationsfortschritt', linewidth=1.5)
# plt.plot(itdata, truezpdata, 'r', label='Wahrer Wert', linewidth=0.75)
# plt.legend(loc = 'lower right', fontsize=18)
# plt.xlabel('Anzahl Iterationen', fontsize=20)
# plt.ylabel('$z_{S,c}$ [m]', fontsize=20)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.grid(True)
# plt.savefig('conv_zp.png', dpi=600, bbox_inches='tight')
# plt.show()
# plt.close(fig3)

# fig4 = plt.figure(figsize=(10,7))
# plt.title("\nKonvergenzverhalten von $\dot{x}_{S,c}$\n", fontsize=25)
# plt.plot(itdata, xvdata, 'b', label='Iterationsfortschritt', linewidth=1.5)
# plt.plot(itdata, truexvdata, 'r', label='Wahrer Wert', linewidth=0.75)
# plt.legend(loc = 'lower right', fontsize=18)
# plt.xlabel('Anzahl Iterationen', fontsize=20)
# plt.ylabel('$\dot{x}_{S,c}$ [m/s]', fontsize=20)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.grid(True)
# plt.savefig('conv_xv.png', dpi=600, bbox_inches='tight')
# plt.show()
# plt.close(fig4)

# fig5 = plt.figure(figsize=(10,7))
# plt.title("\nKonvergenzverhalten von $\dot{y}_{S,c}$\n", fontsize=25)
# plt.plot(itdata, yvdata, 'b', label='Iterationsfortschritt', linewidth=1.5)
# plt.plot(itdata, trueyvdata, 'r', label='Wahrer Wert', linewidth=0.75)
# plt.legend(loc = 'lower right', fontsize=18)
# plt.xlabel('Anzahl Iterationen', fontsize=20)
# plt.ylabel('$\dot{y}_{S,c}$ [m/s]', fontsize=20)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.grid(True)
# plt.savefig('conv_yv.png', dpi=600, bbox_inches='tight')
# plt.show()
# plt.close(fig5)

# fig6 = plt.figure(figsize=(10,7))
# plt.title("\nKonvergenzverhalten von $\dot{z}_{S,c}$\n", fontsize=25)
# plt.plot(itdata, zvdata, 'b', label='Iterationsfortschritt', linewidth=1.5)
# plt.plot(itdata, truezvdata, 'r', label='Wahrer Wert', linewidth=0.75)
# plt.legend(loc = 'lower right', fontsize=18)
# plt.xlabel('Anzahl Iterationen', fontsize=20)
# plt.ylabel('$\dot{z}_{S,c}$ [m/s]', fontsize=20)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.grid(True)
# plt.savefig('conv_zv.png', dpi=600, bbox_inches='tight')
# plt.show()
# plt.close(fig6)

# Evaluate and print runtime of program
print("\nRuntime: %s seconds" % (t.time() - start_time))