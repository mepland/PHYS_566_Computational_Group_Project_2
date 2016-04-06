import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
import math
import random


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


########################################################
# Set fixed/global parameters

neighborhood = 'Von Neumann'
#neighborhood = 'Moore'

n_sim = 50

########################################################
# Print out fixed values
print '\nBeginning cluster_objects.py'
print '\nFixed Parameters are:'
print '---------------------------------------------'

print '\nneighborhood type = '+neighborhood
print '\nnumber of simulations to average = %d' % n_sim

print '\n---------------------------------------------'
print '---------------------------------------------\n'

########################################################
########################################################

########################################################
# Define cluster class to hold all the relevant parameters of a cluster
# Also has helpful member functions
class cluster:

    # constructor
    def __init__(self, cluster_index, N):

	self.index = cluster_index
	self.member_points = []

	# generate random starting position on (0 to N) x (0 to N) grid
	self.member_points.append([random.randrange(0, N+1, 1), random.randrange(0, N+1, 1)])

    # end def for constructor

    # Merge ctm cluster into this cluster, throwing out overlapping points
    # ctm = cluster to merge
    def merge(self, ctm):

	for i in range(len(ctm.member_points)):
		j = 0
		overlap = False
		while j < len(self.member_points) and not overlap:
			if self.member_points[j][0] == ctm.member_points[i][0] and self.member_points[j][1] == ctm.member_points[i][1]:
				overlap = True
			j += 1

		if not overlap:
			self.member_points.append(ctm.member_points[i])

    # end def for merge()


    # See if self and cluster2 touch/overlap
    def touching(self, cluster2):

	# Set up the neighborhood of points to check
	pos_to_check = []
	if neighborhood == 'Moore':
		pos_to_check = [[-1,1], [0,1], [1,1], [-1,0], [0,0], [1,0], [-1,-1], [0,-1], [1,-1]] # Moore neighborhood
	elif neighborhood == 'Von Neumann':
		pos_to_check = [[0,1], [-1,0], [0,0], [1,0], [0,-1]] # Von Neumann neighborhood
	else:
                print 'ERROR!! Unknown neighborhood, exiting!!'
                sys.exit()


	# Check the points, halting as soon as one touches, otherwise go through them all and return False
	touching = False
	i = 0
	while i < len(cluster2.member_points) and not touching:
		j = 0
		while j < len(self.member_points) and not touching:
			k = 0
			while k < len(pos_to_check) and not touching:
				if self.member_points[j][0] == cluster2.member_points[i][0] + pos_to_check[k][0] and self.member_points[j][1] == cluster2.member_points[i][1] + pos_to_check[k][1]:
					touching = True
				k += 1
			j += 1
		i += 1

	return touching
    # end def for touching()


    # see if this cluster spans the N x N space
    # ie one point has y index 0, and another N
    def spanning(self, N):

	touch_left = False
	touch_right = False

	touch_floor = False
	touch_ceiling = False

	span = False

	i = 0

	while i < len(self.member_points) and not span:

		if self.member_points[i][0] == 0: touch_left = True
		if self.member_points[i][0] == N: touch_right = True

		if self.member_points[i][1] == 0: touch_floor = True
		if self.member_points[i][1] == N: touch_ceiling = True


		if touch_left and touch_right and touch_floor and touch_ceiling:
			span = True
			if debugging3: print 'SPANS!!!!'
		i += 1

	return span
    # end def for spanning()


# end class for cluster_point

# Define a function to compute p
def find_p(N, clusters):

	# Find number of occupied sites
	occupied = 0.0

	for i in range(len(clusters)):
		occupied += len(clusters[i].member_points)
	
	return occupied/(N*N)
# end def for find_p()

# Define a function to generate a new cluster, merge clusters in when one touches, deleting the unnecessary one
# keep looping until fully_merged comes back True as more than one cluster may merge per step/new point
def create_next_cluster(index, N, clusters = []):
	
	clusters.append(cluster(index, N))

	fully_merged = False 

	while not fully_merged:
		fully_merged = True
		current_len_clusters = len(clusters)
		i = 0
		while i < current_len_clusters: 
			j = i + 1
			while j < current_len_clusters:
				if debugging3: print 'create_next_cluster(): i = %d, j = %d, len(clusters) = %d' % (i, j, current_len_clusters)
				if clusters[i].touching(clusters[j]):
					clusters[i].merge(clusters[j])
					del clusters[j]
					fully_merged = False
					current_len_clusters += -1
				j += 1
			i += 1

	return clusters
# end def for create_next_cluster()

# Define a function to generate a clusters until one spans
# also save an animation of a list of clusters lists
def create_spanning_cluster(N, m_seed, optional_title, m_path, fname, create_ani):

	# Setup animation, some objects will always be needed and are outside the if statement
	gif_fname = ''
	if create_ani:
		make_path(m_path)
		if make_slides: make_path(m_path+'/'+fname+'_slides')
		gif_fname = m_path+'/'+fname+'.gif'
	else:
		gif_fname = m_path+'/.IGNORE.gif'
		# There doesn't seem to be a way not to make this file
		# Setup so it will be hidden on unix file systems
		# and overwritten repeatedly 

	gifWriter = animation.writers['imagemagick']	
	metadata = dict(title=optional_title, artist='Matplotlib', comment='')
	writer = gifWriter(metadata=metadata)
	
	fig = plt.figure()
	

	# now generate the cluster

	# set the random seed
	random.seed(m_seed)

	# start with one point
	clusters = []
	clusters.append(cluster(0, N))

	span = False

	n_try = 1
	max_try = 20*N
	# max_try = 70 # set artificially low for debugging purposes...

	with writer.saving(fig, gif_fname, 100):

		# keep adding clusters till we span or time out
		while not span and n_try < max_try:
			n_try += 1
			clusters = create_next_cluster(n_try, N, clusters)

			# grab an animation frame for each step
			if create_ani:
				fig = draw_grid_figure(optional_title, fig, [N, m_seed, clusters])
				writer.grab_frame()

				# Save out each frame of the animation for debugging/paper
				if make_slides:
					m_name = m_path+'/'+fname+'_slides/'+fname+'_n_try_%d.pdf' % n_try
					fig.savefig(m_name)
	
			fig.clf() # Clear fig for reuse


			if debugging3: print 'checking span for %d clusters' % len(clusters)

			# See if a cluster spans the world
			i = 0
			while i < len(clusters) and not span:
				span = clusters[i].spanning(N)
				i += 1



		# grab the last frame a few times so it lingers in the gif
		if create_ani:
			fig = draw_grid_figure(optional_title, fig, [N, m_seed, clusters])
			for i in range(4):
				writer.grab_frame()
			fig.clf() # Clear fig for reuse

	# Now that all the plots are done, see if we timed out, warn if so
	# and close the figure to save memory/prevent warnings
	plt.close(fig)

	if n_try >= max_try:
		print 'create_spanning_cluster timed out!'

	return [N, m_seed, clusters]

# end def for create_spanning_cluster()


# Define a function to draw the grid
def draw_grid_figure(optional_title, fig, run = []):
	if(debugging): print 'Beginning draw_grid_figure()'

	N = run[0]
	seed = run[1]
	clusters = run[2]

	# Set up the figure and axes
	fig.clf() # Clear fig for reuse

	ax = fig.add_subplot(111)
	ax.set_title(optional_title)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')

	# adjust axis range
	ax.axis('scaled')
	axis_offset = 0.1*(N+1)
	ax.set_xlim((-axis_offset, N+axis_offset))
	ax.set_ylim((-axis_offset, N+axis_offset))

	# start list for legend entries/handles
 	legend_handles = []

	Dx = 1.0 # grid spacing of the world

	colors = ['#44AA99', '#332288', '#88CCEE', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']
	spanned = False

	# plot the clusters
	first_nonspanning_ij = True
	for i in range(len(clusters)):
		span = clusters[i].spanning(N)
		for j in range(len(clusters[i].member_points)):
			if not span:
				cp = plt.Rectangle((clusters[i].member_points[j][0]-Dx/2, clusters[i].member_points[j][1]-Dx/2), Dx, Dx, color=colors[i%len(colors)], alpha=0.4, fill=True, label='Non-Spanning\nClusters')
				ax.add_artist(cp)

				if first_nonspanning_ij:
					first_nonspanning_ij = False
					legend_handles.append(cp)
			else:
				spanned = True
				span_cp = plt.Rectangle((clusters[i].member_points[j][0]-Dx/2, clusters[i].member_points[j][1]-Dx/2), Dx, Dx, color='blue', alpha=0.95, fill=True, label='Spanning\nCluster')
				ax.add_artist(span_cp)

	if spanned: legend_handles.append(span_cp)

	# make a square on the world border
	world_border = plt.Rectangle((0-Dx/2,0-Dx/2), (N+1)*Dx, (N+1)*Dx, color='black', alpha=1, ls='dashed', fill=False, label='World Border')
	ax.add_artist(world_border)
	legend_handles.append(world_border)

	# draw legend
 	ax.legend(handles=legend_handles, bbox_to_anchor=(1.03, 1), borderaxespad=0, loc='upper left', fontsize='x-small')

	# Annotate
	ann_text = ''
	if spanned:
		ann_text = '$p_{c} =$ %2.3f' % (find_p(N, clusters))
	else:
		ann_text = '$p =$ %2.3f' % (find_p(N, clusters))
	ann_text += '\n$N =$ %d' % (N)
	ann_text += '\nRNG Seed = %d' % (seed)
	ann_text += '\n\nNeighborhood:\n'+neighborhood

	ax.text(1.0415, 0.65, ann_text, bbox=dict(edgecolor='black', facecolor='white', fill=False), size='x-small', transform=ax.transAxes)

	return fig

	if(debugging): print 'draw_grid_figure() completed!!!'
# end def for draw_grid_figure()

# Define a function to find the mean pc for a given N, averaging over n_sim = 50 simulations
# Create plot and animation for the first of the n_sim simulations
# TODO Be careful to not use the same initial_seed/seed twice; do initial_seed += n_sim in larger loop
def find_pc(N, initial_seed, m_path):
	if(debugging): print 'Beginning find_pc()'

	N_name = 'N_%d' % N

	ps = []

	for i in range(0, n_sim):
		if(debugging2): print 'simulation number %d' % i
		first = False
		if i == 0: first = True
		run = create_spanning_cluster(N, initial_seed+i, '', m_path, 'animation_'+N_name, first)
		ps.append( find_p(N, run[2]) )
		if first: plot_grid('', m_path, 'spanned_'+N_name, run)

	if(debugging): print 'find_pc() completed!!!'

	pc = np.mean(ps)

	return pc
 # end def for find_pc()


# Define a function to plot the grid
# needs to be separated to run the animation code nicely
def plot_grid(optional_title, m_path, fname, run = []):
	fig2 = plt.figure()
	fig2 = draw_grid_figure(optional_title, fig2, run)

	# Print it out
	make_path(m_path)
	fig2.savefig(m_path+'/'+fname+'.pdf')
	fig2.clf() # Clear fig2 for reuse
	plt.close(fig2) # Close fig2 to save memory

	if(debugging): print 'plot_grid() completed!!!'
# end def for plot_grid()



########################################################
########################################################
########################################################
# Finally, actually run things!

########################################################
########################################################
# Development Runs 

if(True):
	output_path = '../output/dev'
	debugging = False
	debugging2 = True
	debugging3 = False
	make_slides = False

	'''
 	run = create_spanning_cluster(10, 7, '', output_path, 'test_gif', True)
	plot_grid('Spanned', output_path, 'spanned', run)
	'''

	# find_pc(N, initial_seed, m_path)
	test_pc = find_pc(10, 0, output_path)
	print 'test with N = 10, pc = %2.3f' % test_pc

########################################################
########################################################
# Production Runs for paper 

if(False):
	top_output_path = '../output/plots_for_paper'

	debugging = False
	debugging2 = False
	debugging3 = False
	make_slides = False

	# Part a
	output_path = top_output_path+'/part_a'

	# TODO Run part a code/top level functions here

	# Part b
	output_path = top_output_path+'/part_b'

	# TODO Run part b code/top level functions here


########################################################
print '\n\nDone!\n'


