
import math
import numpy as np
import matplotlib.pyplot as plt
from brian2 import siemens

'''
Used to count the number of spikes in the time interval st-->et and make a matrix of size rs, cs for plotting
Returns the matrix, the sum of all spikes, the average nbr of spikes / neuron /image
'''
def makeSpikeMatrix(mon, rs, cs, st, et, nbrIm):
    all_times_at_spikes = mon.values('t')
    count = 0
    mon_act = np.zeros([rs, cs])
    summ = 0
    for r in range(0, rs):
        for c in range(0, cs):
            #for each neuron in IM
            times = all_times_at_spikes[count]
            t_i = times[times > st]
            t_i = t_i[t_i < et]
            nbr = len(t_i)
            mon_act[r,c]= nbr/nbrIm
            summ = summ + nbr/nbrIm
            count = count + 1
    
    return mon_act, summ, summ/count


'''
Visualizes the spiking activity of the network in a given time period

    IM : SpikeMonitor for the input neurons
    OM : SpikeMonitor for the output neurons
    N_output : number of neurons in the output layer
    start_t : the time from which to count the spikes
    end_t : the time to which to count the spikes
'''
def plotActivity(IM, OM, N_output, start_t, end_t):
    #do it for IM
    IM_matrix = makeSpikeMatrix(IM, 28, 28, start_t, end_t)
    
    sqr = int(math.sqrt(N_output))
    OM_matrix = makeSpikeMatrix(OM, sqr, sqr, start_t, end_t)
    
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.imshow(IM_matrix,vmin=0, vmax=20)
    
    plt.subplot(1, 2, 2)
    plt.imshow(OM_matrix)
    
    plt.show()
 
    
'''
Visualizes the spiking activity of a layer in a given time period

    M : A SpikeMonitor for the given layer
    N_neurons : Number of neurons in the layer
    start_t: first time to consider
    end_t: last time to consider
    nbrOfImages: the number of images that end_t - start_t corresponds to
'''
def plotLayerActivity(M, N_neurons, start_t, end_t, nbrOfImages):
    
    sqr = int(math.sqrt(N_neurons))
    OM_matrix, summ, av = makeSpikeMatrix(M, sqr, sqr, start_t, end_t, nbrOfImages)
    
    plt.figure(figsize=(5,4))
    
    im = plt.imshow(OM_matrix,vmin=0, vmax=10)
    plt.title('Sum spikes / image %.2f <%.2f>' % (summ,av))
    plt.colorbar(im)
    plt.show()


def indexToRowCol(i, rows, cols):
    #from a given rectangular dimension return the coordinates based on a single pixel index
    # counts columns first, rows later. Fits with for r in row: for c in cols nested for loop
    r = int(math.floor(i/cols))
    c = i - r*cols
    return (r,c)

'''
plots the trained synapses to the output layer
assumes that S indices are ordered by all pre to 1st post, all pre to 2nd post, etc  
'''
def plotSynapseMap(S, N_pre, N_post, rows, cols, maxx):
    
    sqr = int(math.sqrt(N_post))
    post_rows = sqr
    post_cols = sqr
    mapp = np.zeros((post_rows*rows, post_cols*cols)) #size of the array to plot    
       
    
    for si in range(len(S)):
                
        npost_index = int(math.floor(si/N_pre)) #gives the post neuron index
        
        npre_index = si-npost_index*N_pre #gives the input neuron index
        
        #transform indices to row and cols
        rcpost = indexToRowCol(npost_index, post_rows, post_cols)
        
        rcpre = indexToRowCol(npre_index, rows, cols)        
                
        mapp[rcpost[0]*rows+rcpre[0]][rcpost[1]*cols+rcpre[1]] = S.g[si]/siemens #save w to map (unit 1)
        
    im = plt.imshow(mapp, vmin=0, vmax=maxx/siemens)

    plt.colorbar(im)
    plt.show()
    