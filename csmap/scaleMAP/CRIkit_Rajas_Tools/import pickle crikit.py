import pickle
import numpy as np
stdimg=None
stdreal=None
cm_start = 0
cm_end = 4000
cm_step = 2
cm_len = (cm_end + cm_step - cm_start) // cm_step
stdfreq = np.arange(cm_start,cm_end+cm_step,cm_step)

ylen, xlen, zlen = crikit_data.hsi.shape
stdimg = np.zeros([ylen,xlen,cm_len])
stdreal = np.zeros([ylen,xlen,cm_len])

for x in range(xlen):
    for y in range(ylen):
        stdimg[y,x] = np.interp(stdfreq,
                                crikit_data.hsi.freq.data,
                                crikit_data.hsi.data[y,x].imag)

for x in range(xlen):
    for y in range(ylen):
        stdreal[y,x] = np.interp(stdfreq,
                                crikit_data.hsi.freq.data,
                                crikit_data.hsi.data[y,x].real)                                

def export_stdimg_pickle():
	filename = input('Please give filename for standard image: ')
	with open(filename + '.pickle', 'wb') as file_handle:
		pickle.dump({'stdfreq' : stdfreq, 'stdimg' : stdimg}, file_handle, protocol=4) 

def export_stdreal_pickle():
	filename = input('Please give filename for standard image: ')
	with open(filename + '.pickle', 'wb') as file_handle:
		pickle.dump({'stdfreq' : stdfreq, 'stdreal' : stdreal}, file_handle, protocol=4) 