from percolation import percolate
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

######################## Documentation ###############################################
# from percolation import percolate                                                  #
# percolate(**kwargs)                                                                #
# @param: N         grid size                                                        #
# @param: show      True -- real-time plotting. Default value: False                 #
# @param: fraction  True -- return fraction F(p>pc) for part b. Default value: False #
######################################################################################

############################# Test ##############################
# real-time plotting
if False:
    pc, grid = percolate(N=100, show=False)
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

    ave_pc = np.load('../output/part a/pc.npy')  # load data

    x = [1.0 / float(k) for k in arr_N]
    plt.plot(x, ave_pc, marker='o')
    plt.show()
    # TODO curve-fitting

############################ Part b #############################
def func(x, a, b):
    return a * x + b

if True:
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
        for i in range(0, max_count):
            F_ave[i] /= counter[i]
        np.save('../output/part b/max_count', max_count)
        np.save('../output/part b/pc', pc_ave)
        np.save('../output/part b/F', F_ave)

    max_count = np.load('../output/part b/max_count.npy')
    pc_ave = np.load('../output/part b/pc.npy')
    F_ave = np.load('../output/part b/F.npy')
    p = np.array([float(i+1)/10000.0 for i in range(len(F_ave))])
    plt.scatter(np.log(p), np.log(F_ave))
    y = 5.0/36.0 * np.log(p)   # reference line
    plt.plot(np.log(p), y, color='r')
    plt.show()
    # TODO linear-fit
