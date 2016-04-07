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

	# Set up the figure and axes
	fig = plt.figure('fig')
	ax = fig.add_subplot(111)
	ax.set_xlabel('$N^{-1}$')
	ax.set_ylabel('$<p_{c}>$')

        # Create the plot
        ax.scatter(InverseN, ave_pc, marker='o', label='$<p_{c}>$', c='blue')

        # Fitting 
        ########################################################

        ########################################################
        # Define the linear fit function
        def linear_fit_function(n_data, offset_fit, slope_fit):
                return offset_fit + slope_fit*n_data
        # end def linear_fit_function

        # actually perform the fits
        # op_par = optimal parameters, covar_matrix has covariance but no errors on plot so it's incorrect...

        linear_p0 = [1.0, 0.0]
        linear_fit_status = True

        maxfev=m_maxfev = 2000

        fit_text = ''

	try:
		linear_op_par, linear_covar_matrix = curve_fit(linear_fit_function, InverseN, ave_pc, p0=linear_p0, maxfev=m_maxfev)
	except RuntimeError:
		print sys.exc_info()[1]
		print 'linear curve_fit failed, continuing...'
		linear_fit_status = False

	# plot the fit
	if(linear_fit_status):
		linear_fit_line, = ax.plot(InverseN, linear_fit_function(InverseN, *linear_op_par), ls='dashed', label='Linear Fit', c="black")

	# Write out the fit parameters
	fit_text = 'Linear Fit Function: $<p_{c}>(N^{-1}) = a + b N^{-1}$'
	if(linear_fit_status):
		fit_text += '\n$a_{\mathrm{Expected}} =$ %2.2f, $a_{\mathrm{Fit}} =$ %2.5f' % (linear_p0[0], linear_op_par[0])
		fit_text += '\n$b_{\mathrm{Expected}} =$ %2.2f, $b_{\mathrm{Fit}} =$ %2.5f' % (linear_p0[1], linear_op_par[1])
	else:
		fit_text += '\nLinear Fit Failed'

        # adjust axis range
	zoom = 0.04
        x1,x2,y1,y2 = ax.axis()
        ax.set_xlim((0.0, (1+zoom)*x2))
        ax.set_ylim(((1-zoom)*y1, (1+zoom)*y2))

        # Draw the legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), borderaxespad=0, fontsize='x-small')

	# Print the fit parameters
	ax.text(0.025, 1-0.03, fit_text, bbox=dict(edgecolor='black', facecolor='white', fill=False), size='x-small', transform=ax.transAxes, va='top')

        # Print it out
        make_path(m_path)
        fig.savefig(m_path+'/pc_ave_vs_InverseN.pdf')

        fig.clf() # Clear fig for reuse

	if(debugging): print 'part_a() completed!!!'


########################################################
########################################################
########################################################
# Finally, actually run things!

########################################################
########################################################
# Development Runs 

if(True):
	output_path = '../output/dev'
	debugging = True
	debugging2 = False
	debugging3 = False

	part_a(output_path)

########################################################
########################################################
# Production Runs for paper 

if(False):
	top_output_path = '../output/plots_for_paper'

	debugging = False
	debugging2 = False
	debugging3 = False


	# Part a
	output_path = top_output_path+'/part_a'

	# TODO Run part a code/top level functions here

	# Part b
	output_path = top_output_path+'/part_b'

	# TODO Run part b code/top level functions here


########################################################
print '\n\nDone!\n'


