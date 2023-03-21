
import os.path
import sys
import numpy as np
from pathlib import Path
from re import findall
import glob

try:
   import cPickle as pickle
except:
   import pickle
from struct import unpack

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

def get_MNIST_data(picklename, MNIST_data_path, bTrain = True):
    """Efficient data retrieval
       First time it resaves the data in pickle-format which is much faster to retrieve the next time
       
       Returns data dictionary
       data['x'] = image data
       data['y'] = image labels
       data['cols'] = number of columns
       data['rows'] = number of rows
       
    """
  
    picklename = '%s.pickle' % picklename
        
    path = Path(os.path.dirname(__file__) )
    path = path / MNIST_data_path    
    
    pn = path / picklename # full path for pickle file
    
    if pn.exists():
        data = pickle.load(open(pn,'rb'))
    else:
        # Open the images with gzip in read binary mode    
        if bTrain:
            images = open(path / 'train-images-idx3-ubyte','rb')
            labels = open(path / 'train-labels-idx1-ubyte','rb')
        else:
            images = open(path / 't10k-images-idx3-ubyte','rb')
            labels = open(path / 't10k-labels-idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]
    
        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]
            
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open(pn, "wb"))
    return data

def reduce_data(factor, data):
    x = data['x'] #image data
    rows = data['rows']
    cols = data['cols']
    
    if rows % factor or cols % factor:
        print('Data reduction failed to due that data size not divisible by factor')
        return data
    
    x_new = []
    count = 0;
    for image in x:        
        count = count + 1;
        if count % 100:
            print('Reducing image #%i' % count)
        #if count == 5:
        #    break
        i_new = zeros((int(rows/factor), int(cols/factor)))
        for r in range(0, rows, factor):
            for c in range(0,cols, factor):
                min_r = max(r-(factor-1),0)
                max_r = min(r+(factor-1),rows-1)
                min_c = max(c-(factor-1),0)
                max_c = min(c+(factor-1),cols-1)
                
                #sum over pixels that should be averaged
                sum_sub = 0;
                for i in image[min_r:max_r]:
                    sum_sub = sum_sub + sum(xx for xx in i[min_c:max_c])
                #normalize
                nbrElements = (max_c-min_c)*(max_r-min_r)
                sum_sub = sum_sub/(nbrElements)
                i_new[int(r/factor), int(c/factor)] = sum_sub #save in new array
    
        x_new.append(i_new)
    
    data['x'] = x_new
    data['rows'] = rows/factor
    data['cols'] = cols/factor
    return data

def savePickle(data, picklename):
    
    picklename = '%s.pickle' % picklename
    
    path = Path(os.path.dirname(__file__) )
    path = path / MNIST_data_path    
    
    pn = path / picklename # full path for pickle file
    pickle.dump(data, open(pn, "wb"))
    
    
#looks throught the current path for files starting with <network_file> and returns the most trained version:
# this is evaluated by the size of the last number in the file name. Thus file_196Neurons_10000 is considered newer than  file_196Neurons_5000
def findLatestVersion(network_file):
    nf_versions = [n for n in glob.glob(network_file+'*') if os.path.isfile(n)] #finds all files starting with <network_file>
    if not nf_versions:
        return [], 0
    nbrs = [int(findall('[0-9]+',v)[-1]) for v in nf_versions]
    return nf_versions[nbrs.index(max(nbrs))], max(nbrs) #returns the filename corresponding to the latest version and the images trained
    
    