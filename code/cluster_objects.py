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

# N size of lattice, N x N


########################################################
# Print out fixed values
print '\nBeginning cluster_objects.py'
#print '\nFixed Parameters are:'
#print '---------------------------------------------'

#print '\n---------------------------------------------'
#print '---------------------------------------------\n'

########################################################
########################################################

########################################################
# Define cluster class to hold all the relevant parameters of a cluster
# Also has helpful member functions
class cluster:

    # constructor
    def __init__(self, N):

	self.member_points = []

	# generate random starting position on (0 to N) x (0 to N) grid
	self.member_points.append([random.randrange(0, N+1, 1), random.randrange(0, N+1, 1)])

	'''
	print self.member_points[0]
	print 'x = %d' % self.member_points[0][0]
	print 'y = %d' % self.member_points[0][1]
	'''

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

	pos_to_check = [[-1,1], [0,1], [1,1], [-1,0], [0,0], [1,0], [-1,-1], [0,-1], [1,-1]]

	touching = False
	i = 0
	while i < len(cluster2.member_points) and not touching:
		j = 0
		while j < len(self.member_points) and not touching:
			k = 0
			while k < len(pos_to_check) and not touching:
				# print 'touch i = %d, j = %d, k = %d' % (i, j, k)
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

	touch_floor = False
	touch_ceiling = False

	i = 0

	while i < len(self.member_points) and not (touch_floor and touch_ceiling):
		if self.member_points[i][1] == 0: touch_floor = True
		if self.member_points[i][1] == N: touch_ceiling = True
		i += 1

	return (touch_floor and touch_ceiling)
    # end def for spanning()


# end class for cluster_point

# Define a function to generate a new cluster, merge when necessary
def create_next_cluster(N, clusters = []):
	
	clusters.append(cluster(N))

	current_len_clusters = len(clusters)
	i = 0

	while i < current_len_clusters: 
		j = i+1
		while j < current_len_clusters:
			# print 'i = %d, j = %d' % (i, j)

			if clusters[i].touching(clusters[j]):
				clusters[i].merge(clusters[j])
				del clusters[j]
				current_len_clusters += -1
			j += 1
		i += 1

	return clusters
# end def for create_next_cluster()

# Define a function to generate a clusters until one spans
def create_spanning_cluster(N, m_seed):

	random.seed(m_seed)

	clusters = []
	clusters.append(cluster(N))

	span = False

	while not span:
		clusters = create_next_cluster(N, clusters)


		print 'checking span for %d clusters' % len(clusters)

		for i in range(len(clusters)):
			span = clusters[i].spanning(N)

	return [N, m_seed, clusters]

# end def for create_spanning_cluster()




# Define a function to plot the grid
def plot_grid(optional_title, m_path, fname, run = []):
	if(debugging): print 'Beginning plot_grid()'

	N = run[0]
	seed = run[1]
	clusters = run[2]

	# Set up the figure and axes
 	fig = plt.figure('fig')
	ax = fig.add_subplot(111)
	ax.set_title(optional_title)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')

	# adjust axis range
	ax.axis('scaled')
	axis_offset = 0.1*N
	ax.set_xlim((-axis_offset, N+axis_offset))
	ax.set_ylim((-axis_offset, N+axis_offset))

	# start list for legend entries/handles
 	legend_handles = []

	Dx = 1.0 # grid size of our world

	# plot the clusters
	for i in range(len(clusters)):
		for j in range(len(clusters[i].member_points)):
			cp = plt.Rectangle((clusters[i].member_points[j][0]-Dx/2, clusters[i].member_points[j][1]-Dx/2), Dx, Dx, color='blue', alpha=1, fill=True, label='Non Spanning\nClusters')
			ax.add_artist(cp)

	legend_handles.append(cp)

	# make a square on the world border
	world_border = plt.Rectangle((0,0), N*Dx, N*Dx, color='black', alpha=1, fill=False, label='World Border')
	ax.add_artist(world_border)
	legend_handles.append(world_border)

	# draw legend
 	ax.legend(handles=legend_handles, bbox_to_anchor=(1.03, 1), borderaxespad=0, loc='upper left', fontsize='x-small')

	# Annotate
	ann_text = 'RNG Seed = %d\n$N =$ %d' % (seed, N)

	ax.text(1.0415, 0.65, ann_text, bbox=dict(edgecolor='black', facecolor='white', fill=False), size='x-small', transform=ax.transAxes)

	# Print it out
	make_path(m_path)
	# fig.savefig(m_path+'/'+fname+'.png', dpi=900)
	# if len(cluster) < 10**3: fig.savefig(m_path+'/'+fname+'.pdf')
	fig.savefig(m_path+'/'+fname+'.pdf')

	fig.clf() # Clear fig for reuse

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
	debugging = True
	debugging2 = True	

# 	spanned_space = create_spanning_cluster(20, 7)

	m_seed = 7
	N = 10

	random.seed(m_seed)

	clusters = []
	clusters.append(cluster(N))

	for i in range(10):
		clusters = create_next_cluster(N, clusters)

	# plot_grid(optional_title, m_path, fname, run = [])
	# N = run[0]
	# seed = run[1]
	# clusters = run[2]

	plot_grid('test', output_path, 'test', [N, m_seed, clusters])


########################################################
########################################################
# Production Runs for paper 

if(False):
	output_path = '../output/plots_for_paper/problem_3'
	debugging = False
	debugging2 = False


########################################################
print '\n\nDone!\n'


