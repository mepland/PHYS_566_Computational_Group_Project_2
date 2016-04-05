import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import time

global N, grid, labels

def percolate(**kwargs):
    global N, grid, labels

    N = kwargs.get('N', 20)
    show = kwargs.get('show', False)

    grid = np.zeros([N, N])     # initialize
    labels = set([0])              # tree set

    # generate random sequence
    sq = np.arange(N * N)
    np.random.shuffle(sq)

    if show:
        plt.imshow(grid, origin='lower', interpolation='nearest')
        plt.ion()
        plt.show()
    count = 0
    while not percolation():
        x, y = position(sq[count])    # new site
        update(x, y)              # update the grid
        if show:
            plt.imshow(grid, origin='lower', interpolation='nearest')
            plt.draw()
            time.sleep(0.05)
        count += 1

    p = float(count) / float(N * N)
    clusters = len(labels) - 1
    return p, clusters, grid




def percolation():
    """
    Test whether the system percolates
    :return: boolean value
    """
    global N, grid, labels
    side1 = set()
    side2 = set()
    side3 = set()
    side4 = set()
    for i in range(N):
        side1.add(grid[0][i])
        side2.add(grid[i][0])
        side3.add(grid[N-1][i])
        side4.add(grid[i][N-1])
    int_set = side1.intersection(side2, side3, side4)
    if len(int_set) > 1:
        return True
    if len(int_set) == 0:
        return False
    elif int_set.pop() != 0:
        return True


def position(n):
    """
    return the x-, y-coodinate of the nth grid
    """
    global N, grid, labels
    x = n / N
    y = n - x * N
    return x, y


def label(i, j):
    """
    return the label of site(i,j)
    """
    global N, grid, labels
    if -1 < i < N and -1 < j < N:
        return grid[i][j]
    else:
        return 0


def update(i, j):
    global N, grid, labels
    neighbor_labels = set([label(i-1, j), label(i+1, j), label(i, j-1), label(i, j+1)])
    if neighbor_labels.__contains__(0):
        neighbor_labels.remove(0)
    length = len(neighbor_labels)
    if length > 0:
        grid[i][j] = neighbor_labels.pop()
        # merge
        if length > 1:
            for m in range(N):
                for n in range(N):
                    if neighbor_labels.__contains__(grid[m][n]):
                        grid[m][n] = grid[i][j]
        labels -= neighbor_labels        # remove labels
    else:
        grid[i][j] = list(labels)[len(labels)-1] + 1     # new label
        labels.add(grid[i][j])