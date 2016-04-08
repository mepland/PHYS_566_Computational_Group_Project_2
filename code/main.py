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
    fraction = kwargs.get('fraction', False)

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

    pc = float(count) / float(N * N)
    # part a
    if not fraction:
        return pc, grid
    # part b
    frac = []
    pc_count = count
    while count < N*N:
        x, y = position(sq[count])    # new site
        update(x, y)              # update the grid
        count += 1
        p_label = same_label()
        f = float(len(np.where(grid == p_label)[0])) / float(count)
        frac.append(f)
    frac = np.array(frac)
    return pc, pc_count, frac, grid


def same_label():
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
    int_set -= set([0])
    return list(int_set)[0]

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
    int_set -= set([0])
    if len(int_set) > 0:
        return True
    else:
        return False


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

######################## Documentation ###############################################
# from percolation import percolate                                                  #
# percolate(**kwargs)                                                                #
# @param: N         grid size                                                        #
# @param: show      True -- real-time plotting. Default value: False                 #
# @param: fraction  True -- return fraction F(p>pc) for part b. Default value: False #
######################################################################################

############################# Test ##############################
# real-time plotting
if True:
    pc, grid = percolate(N=20, show=False)
    plt.close()
    # plot the figure at p = pc
    plt.imshow(grid, origin='lower', interpolation='nearest')
    print pc
    plt.show()

############################ Part a #############################
if False:
    arr_N = [5, 10, 15, 20, 30, 50, 80]
    ave_pc = []
    if False:
        for i in range(len(arr_N)):
            print i
            sum = 0.0
            for j in range(50):
                p, grid = percolate(N=arr_N[i], show=False)
                sum += p
            ave_pc.append(sum / 50.0)
        np.save('../output/part a/pc', ave_pc)

############################ Part b #############################
def func(x, a, b):
    return a * x + b

if False:
    if False:
        F_ave = np.zeros(100*100)
        counter = np.zeros(100*100)
        max_count = 0
        pc_ave = 0.0
        for i in range(50):
            print i
            pc, pc_count, frac, grid = percolate(N=100, fraction='True')
            F_ave[0: len(frac)] += frac
            counter[0: len(frac)] += 1.0
            pc_ave += pc
            if len(frac) > max_count:
                max_count = len(frac)
        pc_ave /= 50.0
        F_ave = F_ave[0: max_count]
        for i in range(0, max_count):
            F_ave[i] /= counter[i]
        np.save('../output/part b/max_count', max_count)
        np.save('../output/part b/pc', pc_ave)
        np.save('../output/part b/F', F_ave)

