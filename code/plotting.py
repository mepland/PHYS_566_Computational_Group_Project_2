import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
import math

########################################################
# Define a function to create the output dir
# If it already exists don't crash, otherwise raise an exception
# Adapted from A-B-B's response to http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
# Note in python 3.4+ 'os.makedirs(output_path, exist_ok=True)' would handle all of this...
def make_path(path):
	try: 
		os.makedirs(path)
	except OSError:
		if not os.path.isdir(path):
			raise Exception('Problem creating output dir %s !!!\nA file with the same name probably already exists, please fix the conflict and run again.' % output_path)
# end def for make_path

# Define a function to do part a plotting and fitting
def part_a(m_path):
	if(debugging): print 'Beginning part_a()'

	# setup data
	arr_N = [5, 10, 15, 20, 30, 50, 80]
	ave_pc = np.load('../output/part a/pc.npy')  # load data

	InverseN_list = [1.0 / float(k) for k in arr_N]
	InverseN = np.array(InverseN_list)
	
	# setup data to fit
	ave_pc_fit = np.delete(ave_pc, [0, 1])
	InverseN_fit = np.delete(InverseN, [0, 1])

	# Set up the figure and axes
	fig = plt.figure('fig')
	ax = fig.add_subplot(111)
	ax.set_xlabel(r'$N^{-1}$')
	ax.set_ylabel(r'$\langle p_{c} \rangle$')

        # Create the plot
        ax.scatter(InverseN, ave_pc, marker='o', edgecolor='blue', label=r'$\langle p_{c} \rangle$', c='blue')

        # adjust axis range
	zoom = 0.01
        x1,x2,y1,y2 = ax.axis()
        ax.set_xlim((0.0, (1+zoom)*x2))
        ax.set_ylim(((1-zoom)*y1, (1+zoom)*y2))

        # Fitting 
        ########################################################

        ########################################################
        # Define the linear fit function
        def linear_fit_function(n_data, offset_fit, slope_fit):
                return offset_fit + slope_fit*n_data
        # end def linear_fit_function

        # actually perform the fits
        # op_par = optimal parameters, covar_matrix has covariance but no errors on plot so it's incorrect...

        linear_p0 = [0.593, 0.0]
        linear_fit_status = True

        maxfev=m_maxfev = 2000

        fit_text = ''

	try:
		linear_op_par, linear_covar_matrix = curve_fit(linear_fit_function, InverseN_fit, ave_pc_fit, p0=linear_p0, maxfev=m_maxfev)
	except RuntimeError:
		print sys.exc_info()[1]
		print 'linear curve_fit failed, continuing...'
		linear_fit_status = False

	# plot the fit
        x1,x2,y1,y2 = ax.axis()
	fit_x = np.linspace(x1, x2, 1000)

	if(linear_fit_status):
		linear_fit_line, = ax.plot(fit_x, linear_fit_function(fit_x, *linear_op_par), ls='solid', label='Linear Fit', c="black")
		fit_boundary_line = ax.axvline(x=InverseN_fit[0], ls = 'dashed', label='Fit Boundary', c='grey')

	# Write out the fit parameters
	fit_text = r'Linear Fit Function: $\langle p_{c}\rangle (N^{-1}) = p_{c\,0} + b N^{-1}$'
	if(linear_fit_status):
		fit_text += '\n$p_{c\,0\,\mathrm{Expected}} =$ %2.2f\n$p_{c\,0\,\mathrm{Fit}} =$ %2.5f' % (linear_p0[0], linear_op_par[0])
#		fit_text += '\n$b_{\mathrm{Expected}} =$ %2.2f, $b_{\mathrm{Fit}} =$ %2.5f' % (linear_p0[1], linear_op_par[1])
		fit_text += '\n$b_{\mathrm{Fit}} =$ %2.5f' % (linear_op_par[1])
		fit_text += '\nFit Range: $N^{-1} <$ %2.2f' % (InverseN_fit[0])
	else:
		fit_text += '\nLinear Fit Failed'


        # Draw the legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), borderaxespad=0, fontsize='x-small')

	# Print the fit parameters
	ax.text(0.025, 1-0.03, fit_text, bbox=dict(edgecolor='black', facecolor='white', fill=True), size='x-small', transform=ax.transAxes, va='top')

        # Print it out
        make_path(m_path)
        fig.savefig(m_path+'/pc_ave_vs_InverseN.pdf')

        fig.clf() # Clear fig for reuse

	if(debugging): print 'part_a() completed!!!'
# end def for part_a



# Define a function to do part b plotting and fitting
def part_b(m_path):
	if(debugging): print 'Beginning part_b()'

	# setup data
	F_ave = np.load('../output/part b/F.npy')
	p = np.array([float(i+1)/10000.0 for i in range(len(F_ave))])
		
	# setup data to fit
	fit_range_min = 6*10**-3
	fit_range_max = 6*10**-2

	F_ave_fit_list = []
	p_fit_list = []

	for i in range(p.size):
		if fit_range_min <= p[i] and p[i] <= fit_range_max:
			F_ave_fit_list.append(F_ave[i])
			p_fit_list.append(p[i])

	F_ave_fit = np.array(F_ave_fit_list)
	p_fit = np.array(p_fit_list)



	# Set up the figure and axes
	fig = plt.figure('fig')
	ax = fig.add_subplot(111)
	ax.set_xlabel('$p$')
	ax.set_ylabel(r'$\langle F(p>p_{c}) \rangle$')

        # Create the plot
	# ax.scatter(p, F_ave, marker='o', label='$<F(p>p_{c})>$', c='blue')
        ax.plot(p, F_ave, marker=None, ls='solid', label=r'$\langle F(p>p_{c}) \rangle$', c='blue')


        # adjust axis range
        ax.set_xlim((5.0*10**-5, 1.0))
        ax.set_ylim((3*10**-1, 2*10**0))

        # Fitting 
        ########################################################

        ########################################################
        # Define the power law fit function
        def power_law_fit_function(n_data, pow_fit, slope_fit):
                return slope_fit*pow(n_data, pow_fit)
        # end def power_law_fit_function

        # actually perform the fits
        # op_par = optimal parameters, covar_matrix has covariance but no errors on plot so it's incorrect...

        power_p0 = [5.0/36.0, 1.0]
        power_fit_status = True

        maxfev=m_maxfev = 2000

        fit_text = ''

	try:
		power_op_par, power_covar_matrix = curve_fit(power_law_fit_function, p_fit, F_ave_fit, p0=power_p0, maxfev=m_maxfev)
	except RuntimeError:
		print sys.exc_info()[1]
		print 'power curve_fit failed, continuing...'
		power_fit_status = False

	# plot the fit
	x1,x2,y1,y2 = ax.axis()
	fit_x = np.linspace(x1, x2, 1000)

	if(power_fit_status):
		power_fit_line, = ax.plot(fit_x, power_law_fit_function(fit_x, *power_op_par), ls='solid', label='Power Law Fit', c="black")
		fit_boundary_line = ax.axvline(x=fit_range_min, ls = 'dashed', label='Fit Boundary', c='grey')
		fit_boundary_line = ax.axvline(x=fit_range_max, ls = 'dashed', label=None, c='grey')

	# Write out the fit parameters
	fit_text = r'Power Law Fit Function: $\langle F(p>p_{c}) \rangle (p) = F_{0}(p-p_{c})^{\beta}$'
	if(power_fit_status):
		fit_text += '\n$\\beta_{\mathrm{Expected}} =$ %2.2f\n$\\beta_{\mathrm{Fit}} =$ %2.5f' % (power_p0[0], power_op_par[0])
		fit_text += '\n$F_{0\,\mathrm{Fit}} =$ %2.5f' % (power_op_par[1])
		fit_text += '\nFit Range: %2.2f $< p <$ %2.2f' % (fit_range_min, fit_range_max)
	else:
		fit_text += '\nPower Law Fit Failed'


        # Draw the legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), borderaxespad=0, fontsize='x-small')

	# Print the fit parameters
	ax.text(0.025, 1-0.03, fit_text, bbox=dict(edgecolor='black', facecolor='white', fill=True), size='x-small', transform=ax.transAxes, va='top')

        # Print it out
        make_path(m_path)

	fig.savefig(m_path+'/F_ave_vs_p_linearScale.pdf')

	# Make the axis log log
	ax.set_xscale('log')
	ax.set_yscale('log')
       
	fig.savefig(m_path+'/F_ave_vs_p.pdf')

        fig.clf() # Clear fig for reuse

	if(debugging): print 'part_b() completed!!!'
# end def for part_b


########################################################
########################################################
########################################################
# Finally, actually run things!

########################################################
########################################################
# Development Runs 

if(False):
	output_path = '../output/dev'
	debugging = True

	# part_a(output_path)
	# part_b(output_path)

########################################################
########################################################
# Production Runs for paper 

if(True):
	output_path = '../output/plots_for_paper'

	debugging = True

	# Part a
	part_a(output_path)

	# Part b
	part_b(output_path)


########################################################
print '\n\nDone!\n'


