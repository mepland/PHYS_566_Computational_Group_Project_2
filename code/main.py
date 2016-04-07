from percolation import percolate
import matplotlib.pyplot as plt
import numpy as np

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

    ave_pc = np.load('../output/part a/pc.npy')  # load data

    x = [1.0 / float(k) for k in arr_N]
    plt.plot(x, ave_pc, marker='o')
    plt.show()
    # TODO curve-fitting

############################ Part b #############################
if True:
    if False:
        F_ave = np.zeros(100*100)
        for i in range(100):
            print i
            pc, p, frac, grid = percolate(N=100, fraction='True')
            p -= pc
            if len(F_ave) <= len(p):
                F_ave = F_ave + p[0: len(F_ave)]
            else:
                F_ave = F_ave[0: len(p)] + p
        F_ave /= 100.0
        np.save('../output/part b/F',F_ave)

    F_ave = np.load('../output/part b/F.npy')

    x = [float(i)/(100.0**2) for i in range(1, len(F_ave)+1)]
    plt.loglog(x, F_ave)
    plt.show()
