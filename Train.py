from brian2 import *

import timeit
import math
import glob
import random
import pickle5 as pickle

#for loading training data
from datahandling import *
#for plotting stuff
from plottingtools import *

import os
import numpy as np
import matplotlib.pyplot as plt

debug = True #if set to false you will compile the code into c++ code which is slightly faster, but then you can't plot or save stuff during training
useFashion = False #if set to true you will use the MNIST Fashion data set instead of MNIST numbers. Fashion is a harder challenge to be tested at the end of the project

if debug:
    prefs.codegen.target = 'numpy'  # use the Python fallback
    set_device('runtime', build_on_run=False)
else:
    set_device('cpp_standalone', build_on_run=False)
    prefs.devices.cpp_standalone.openmp_threads = 4

if useFashion:
    MNIST_data_path = 'fashion-mnist-master/data/fashion'
else:
    MNIST_data_path = 'number-mnist-master'




# Load MNIST data set
print('LOADING TRAINING DATA - ',end='')
training_data = get_MNIST_data('training', MNIST_data_path)
print('Done.')
print('LOADING TESTING DATA - ',end='')
testing_data = get_MNIST_data('testing', MNIST_data_path, bTrain = False)
print('Done.')


start_scope() #brian2 code, never mind it

rows = training_data['rows']
cols = training_data['cols']
N_input = rows*cols;

#GLOBAL OPTIONS #########################
train = False #if true it trains the network
test = False #only important if train=False, then if test=False then determine neuron classes, if true then test network
restart_sim = False # if set to true the training will start over regardless of whether there are previous versions of the network

plotEvery = 50 #plot information every X images trained (only done in debug mode)
#saveEvery = 1000 #save the network state every X images trained (only possible in debug mode)
saveEvery = 1000
monitorSpikes = True #if true will create a SpikeMonitor on the network (makes training slower) and will plot the spike activity as a map at every <plotEvery> interval
N_out = 196;  #Number of neurons in output layer, should be N^2, where N is an integer (so that one can plot a nice square matrix of the synapse weights)

latInh = 2 # if =2 use winner takes all as lateral inhibition, if =1 uses soft lateral inhibition, if < 1 use no lateral inhibition at all!

rate_max = 50 #Hz (unit is added later) Maximum spiking rate of input neurons
presentation_time = 250*ms #the time that an image is presented to the network
rest_time = 150*ms #the time inbetween image presentation, to allow for dynamic variables to reach resting states

epochs = 1 #the number of epochs to train for (I have only tested the code for 1 epoch)


network_file = 'network_N=' + str(N_out) + '_latInh='+str(latInh)

#########################################

#STDP parameters

if train:
    learning_rate = 0.01
    T_plus = 10.*mV #adaptive threshold
else:
    learning_rate = 0. #don't change weights
    T_plus = 0.*mV #keep thresholds constant

gmax = 100*nS
Q_max = 1E-15*coulomb
C_post = 1E-12*farad
dV_max = Q_max/C_post #with the standard C_post and Q_max this gives dV_max = 2.5 mV

pot_win = 20*ms #time window for STDP potentiation
dgp = learning_rate*gmax # change of conductivity during potentiation
depression_domination = 0.5 #a factor to make depression dominate over potentation
dgd = learning_rate*gmax*depression_domination # change of conductivity during depression


# tpre is a timer that is reset at each pre spike. On post-spike it is used to check 
# how long since the last prespike and to determine whether to potentiate or depress

stdp_eq = '''
    g : siemens
    dtpre/dt = 1. : second (event-driven)
    '''
on_pre_eq = '''
    v_post += dV_max * g/gmax
    tpre = 0.*ms
    '''
on_pre_eq_inh = 'tpre = 0.*ms'
if latInh > 1:
    on_pre_eq_inh += '\n v_post = v_0'
elif latInh == 1:
    on_pre_eq_inh += '\n v_post -= dV_max'
#else have no lateral inhibition at all!

on_post_eq =  '''    
    g = clip(g + dgp*int(tpre <= pot_win)-dgd*int(tpre>pot_win),0*nS, gmax)
    '''

#Neuron differential equations
tau_v = 20*ms #time constant for neuron potential decay
v_0 = 0*mV #resting state potential
tau_T = 40*ms #time constant for the threshold


# neuron diff eq
eqs = '''
dv/dt = -(v-v_0)/tau_v : volt
dT/dt = -T/tau_T : volt
T_0 : volt
'''

#Neuron rules
thres = '(v > T_0 + T)'
res = '''
v=v_0
T+=T_plus
'''

####### BUILD NETWORK ########

#Create Explicit Network
N = Network()

# create input layer
I = PoissonGroup(N_input, rates=10*Hz, name='Input_Neurons')


# create output layer
O = NeuronGroup(N_out, eqs, threshold=thres, reset=res, method='exact', name='Output_Neurons')

O.T_0 = 'v_0 + (40.0+rand()*10.0)*mV'  #equilibrium threshold

N.add(I,O) # add neuron layers to network

if monitorSpikes:
    OM = SpikeMonitor(O,name="spikemonitor")
    N.add(OM)
   
# connect to input layer
S_i = Synapses(I, O, stdp_eq,on_pre=on_pre_eq, on_post=on_post_eq, name='Excitatory_Synapses')
S_inh = Synapses(O,O, stdp_eq,on_pre=on_pre_eq_inh, name='Inhibitory_Synapses') #inhibitory connections within output layer
N.add(S_i)
N.add(S_inh)

print('CONNECTING INPUT LAYER: ',end='')
allInputs = range(N_input)
for o in range(N_out):
    S_i.connect(i=allInputs, j = o)
S_i.g = 'rand()*gmax'

#connect inhibitory synapses
S_inh.connect(condition='i!=j')
S_inh.g = 'gmax'

print('Done')



if train:

    ###### CONTINUE WITH TRAINING ON A PREVIOUS VERSION OF THE NETWORK? ######
    
    latest_version, images_trained = findLatestVersion(network_file)
    
    if latest_version:
        #previous version of network exists, so reload the latest version  
        N.restore(filename=latest_version)
        print('Loaded network from file: ' + latest_version + ' (' + str(images_trained) + ' images trained)')
    else:
        print('No previous training found with this network name.')
    
    ####### TRAIN THE NETWORK ####
    
    print('TRAINING...')
    print('')

    for e in range(epochs):
        #train on each image
        training_counter = 0
        ratess = [0 for x in range(N_input)] #create rate array to fill with rates for each image
        rest_rates = [0 for x in range(N_input)]*Hz #a blank copy to use in the resting state
        
        start = timeit.default_timer() #for checking computing time
        t_s = N.t #simulation start time
        
        for image in training_data['x']:        
            
            training_counter = training_counter + 1 #counts how many images have been trained
            
            if training_counter <= images_trained: #in case we continue an old sim
                continue #skip until next untrained image
                
            #### SET RATES ON INPUT
            j = 0 #counter for transferring into rates
            for r in range(0, rows):
                for c in range(0, cols):                
                    ratess[j] = image[r, c]*rate_max/255.0*Hz #converts greyscale to frequency (rate)
                    
                    j = j + 1       
            I.rates = ratess
            
            
            ######
            
            N.run(presentation_time, profile=True) #present data to network for given time
            
            #turn off input       
            I.rates = rest_rates
            N.run(rest_time, profile=True)          
            
            #FOR PLOTTING AND SAVING DURING A RUN          
            if training_counter % 10 == 0:            
                endtime = timeit.default_timer()
                print('\r\tTrained on #%s images (%.2f s/image )' % (training_counter, (endtime - start)/10))

                if debug and training_counter % plotEvery == 0:
                    plotSynapseMap(S_i, N_input, N_out, rows, cols, gmax)                    
                    if monitorSpikes:
                        t_e = N.t #simulation end time
                        plotLayerActivity(OM, N_out, t_s, t_e, plotEvery)
                        t_s = t_e #new simulation start time
                        #reset spike monitor (for memory purposes)
                        N.remove(OM) #remove from network
                        OM = SpikeMonitor(O) #create a new spike monitor
                        N.add(OM)
                        
                #SAVE NETWORK
                if debug and training_counter % saveEvery == 0:
                    current_version = network_file+'_'+str(training_counter)
                    N.store(filename=current_version) #saves the state (not the objects) of the network
                    print('\r\t---- Saved network state to disk ----')
                    
                start = timeit.default_timer() #for checking computing time
        
else:
    #CLASSIFICATION ------
    # Pick a random set of images and labels and see which neurons spike most to which. in this way one can give each neuron a "class" for testing
    latest_version, images_trained = findLatestVersion(network_file)

    if latest_version and not restart_sim:
    #previous version of network exists, so reload the latest version
    
        N.restore(filename=latest_version)
        print('Loaded network from file: ' + latest_version + ' (' + str(images_trained) + ' images trained)')
        OM = SpikeMonitor(O)
        N.add(OM)
    learning_rate = 0. #don't change weights
    T_plus = 0.*mV #keep thresholds constant  
    result = np.zeros((10,N_out))
    images = training_data['x']
    labels = training_data['y']
    neuronClass = np.ones(N_out)*(-1)
    ratess = [0 for x in range(N_input)]
    rest_rates = [0 for x in range(N_input)]*Hz
    valArr = np.zeros(N_out)
    for index in range(1000):
        r = randint(0,60000)
        image = images[r]        
        label = labels[r]
        if index % 50 == 0:
            print("Image: ",index)
        #### SET RATES ON INPUT
        j = 0 #counter for transferring into rates
        for r in range(0, rows):
            for c in range(0, cols):                
                ratess[j] = image[r, c]*rate_max/255.0*Hz #converts greyscale to frequency (rate)
                j = j + 1       
        I.rates = ratess
        
        
        ######
        
        N.run(presentation_time, profile=True) #present data to network for given time
        values = OM.values('t')
        prevValArr=valArr
        valArr = np.array([len(values[item]) for item in range(N_out)]) 
        
        result[int(label)] = result[int(label)] + valArr - prevValArr
        #turn off input       
        I.rates = rest_rates
        N.run(rest_time, profile=True)   
    for index, neuron in enumerate(result.transpose()):
        neuronClass[index] = np.argmax(neuron)
    print(neuronClass)
    
    #TESTING -----
    # Go through the test data and determine whether the network responds in the way that it should given the input data
    #  i.e if the input is a "2" do the neurons classified as "2" neurons spike most?
    

  
    
  
    images = testing_data['x']
    labels = testing_data['y']
    AccVec=[]
    toRun=[1,2,3,5,10,15,20,40,60]
    for file in toRun:
        filename="network_N=196_latInh=2_" + str(file) + "000"
        N.restore(filename=filename)
        
        correct = 0
        ratess = [0 for x in range(N_input)]
        rest_rates = [0 for x in range(N_input)]*Hz
        
        
        #OM = SpikeMonitor(O,name='spikemonitor')
        #N.add(OM)
        for index in range(len(images)):
            image = images[index]
            label = labels[index]
                
            #### SET RATES ON INPUT
            j = 0 #counter for transferring into rates
            for r in range(0, rows):
                for c in range(0, cols):                
                    ratess[j] = image[r, c]*rate_max/255.0*Hz #converts greyscale to frequency (rate)
                    j = j + 1       
            I.rates = ratess
            if index % 50 == 0:
                print("test: ",index)
            
            ######
            
            N.run(presentation_time, profile=True) #present data to network for given time
            values = OM.values('t')
            prevValArr=valArr
            valArr = np.array([len(values[item]) for item in range(N_out)]) 
            
            neuron = np.argmax(valArr-prevValArr)
            pred = neuronClass[neuron]
            if pred == label:
                correct +=1
        accuracy = correct/len(images)
        AccVec.append(accuracy)
        print("accuracy: %f", (accuracy))
        
#Fpr cpp implementation, now compile the code and run it
device.build(directory='output', compile=True, run=True, debug=False)

toRun=[1,2,3,5,10,15,20,40,60]
off=[0.4246,0.4154,0.4928,0.5147,0.56,0.6319, 0.6639, 0.778, 0.7971]
on=[0.463, 0.4808, 0.5744, 0.6067, 0.6682, 0.7057, 0.7294, 0.7659, 0.7663]
dan= [0.3,0.9,3,15,40,60]
LI2= [ 0.32,0.43,0.58,0.71,0.73,0.76]
  
plot(dan,LI2,label="reference")    
plot(toRun,on,label="on")
plot(toRun,off,label="off")


plt.xlabel("Training Images (Thousands)")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Adaptive threshold effect with device variations")
plt.ylim([0, 1])


plt.savefig('adapt.png', dpi=300)


    


