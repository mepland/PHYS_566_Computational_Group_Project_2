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

