from percolation import percolate
import matplotlib.pyplot as plt
import numpy as np

######################## Test ##########################

p, clusters, grid = percolate(N=20, show=True)  # pc, # of cluster, final grid = percolate(size, show=True/False)
print p, clusters
plt.imshow(grid, origin='lower', interpolation='nearest')
plt.savefig('figure.pdf')


pc = []
for i in range(10):
    pc.append(percolate(N=20, show='False'))

print np.sum(pc) / 10.0