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
    p, grid = percolate(N=50, show=True)
    # plot the figure at p = pc
    plt.imshow(grid, origin='lower', interpolation='nearest')
    plt.show()

############################ Part a #############################
if True:
    arr_N = [5, 10, 15, 20, 30, 50, 80]
    ave_pc = []

    for i in range(len(arr_N)):
        print i
        sum = 0.0
        for j in range(50):
            p, grid = percolate(N=arr_N[i], show=False)
            sum += p
        ave_pc.append(sum / 50.0)

    np.save('../output/pc', ave_pc)

    ave_pc = np.load('../output/pc.npy')

    x = [1.0 / float(k) for k in arr_N]
    plt.plot(x, ave_pc, marker='o')
    plt.show()
    # TODO curve-fitting

##################### Part b #####################
if False:
    pc, p, frac, grid = percolate(N=100, fraction='True')
    p -= pc
    plt.plot(p, frac)
    plt.show()
    # TODO