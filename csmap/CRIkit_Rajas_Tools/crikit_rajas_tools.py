import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pickle
import flowio
import pandas as pd

import time

import socket
import threading
import socketserver

r'''
Doc: Intended pathway is to go to CRIkit Jupyter/IPython console and do:

Import and initialize:

import sys
sys.path.append(r'C:\Users\rajas\OneDrive - Georgia Institute of Technology\Raman\Curci\SPADE\spade\CRIkit_Rajas_Tools')
import crikit_rajas_tools as ct
ct.init(crikit_data)

-----------------

First calibrate wavenumbers before exporting:

ct.calibrate()

There should be a peak near 1004 cm-1, maybe 997 - 999 cm-1.
Find the peak and set the wavenumber in "Measured" section in 
the dialog box to that wavenumber. So 1004 -> 997. Hit OK in the top.

-----------------

Now standardize the CRIkit image wavenumbers using:

ct.standardize()

Alternatively, import a standardized image using:

ct.import_stdimg_pickle()

You can export the standardized image using:

ct.export_stdimg_pickle()

-----------------

Import the main mask that tells where the worm is using:

ct.import_main_mask()

Import smaller masks covering regions of interest (ROIs) using:

ct.import_submasks()

Submasks are required to all be in one folder, with no other files.
All files must be the submasks.

-----------------

Perform the PCA using:

ct.perform_PCA(mask,threshold)

mask can be:
* totalmask : Combination of all submasks
* mainmask  : The main mask that covers the whole worm
* masks[n]  : Some particular mask 

threshold is noise floor, specified as the number of pixels that the least important component must explain.
threshold should usually be 1 or 0.1: 1 or 0.1 pixels must be explained by the least important component.

-----------------

Export the PCA results to CSV via:

ct.export_pca_components_csv()
ct.export_pca_mean_csv()

This will allow MATLAB to invert the PCAs back into spectra later on, in SPADE

-----------------

Export the masked regions into FCS files for SPADE to process:

ct.export_all_FCS()

Remember while importing in SPADE that the first column is Y and the second is X.
Set SPADE to ignore these.

-----------------

All done for now! Move into SPADE now. 
We may be back when we want to get the SPADE nodes to show spectra.

'''
global crikit_data
global stdimg, stdfreq, nrmimg

stdimg=None
cm_start = 0
cm_end = 4000
cm_step = 2
cm_len = (cm_end + cm_step - cm_start) // cm_step
stdfreq = np.arange(cm_start,cm_end+cm_step,cm_step)

def init(crikit_data_in):
	global crikit_data
	crikit_data = crikit_data_in
	global yi, xi, xlen, ylen, zlen
	xlen, ylen, zlen = crikit_data.hsi.shape
	xi, yi = np.meshgrid(np.arange(xlen),np.arange(ylen))
	global stdimg, stdfreq
	stdimg = np.zeros([ylen,xlen,cm_len])
	stdfreq = np.arange(cm_start,cm_end+cm_step,cm_step)
	
def calibrate():
	crikit_data.calibrate()

def standardize_wavenumber_x():
	for x in range(xlen):
		for y in range(ylen):
			stdimg[y,x] = np.interp(stdfreq,
								 crikit_data.hsi.freq.data,
								 crikit_data.hsi.data[x,y].imag)
	
# def standardize_abs():
	# global stdimg, stdfreq
	# stdimg = np.zeros([ylen,xlen,cm_len])
	# stdfreq = np.arange(cm_start,cm_end+cm_step,cm_step)
	# for x in range(xlen):
		# for y in range(ylen):
			# stdimg[y,x] = np.interp(stdfreq,
								 # crikit_data.hsi.freq.data,
								 # np.abs(crikit_data.hsi.data[y,x]))

def standardize_normalize(img,mask): # expects wavenumber standardized image
	global scaler, nrmimg
	scaler = standardize_get_zscaler(img,mask)
	nrmimg = standardize_zscale_image(img,scaler)
	#scaler = preprocessing.StandardScaler().fit(img[mask])
	#nrmimg = scaler.transform(img.reshape((xlen*ylen,cm_len))).reshape((xlen,ylen,cm_len))
	return(nrmimg)

def standardize_get_zscaler(img,mask=None):
	if (mask == None):
		scaler = preprocessing.StandardScaler().fit(img.reshape((xlen*ylen,cm_len)))
	else:
		scaler = preprocessing.StandardScaler().fit(img[mask])
	return(scaler)	

def standardize_zscale_image(img,scaler):
	out_nrmimg = scaler.transform(img.reshape((xlen*ylen,cm_len))).reshape((xlen,ylen,cm_len))
	return(out_nrmimg)	
	
def silence_silent_region(img): # predefined wavenumbers in cm-1 for silent region
	return(silence_region(img,[1800,2700]))

def silence_ch_stretch(img): # predefined wavenumbers in cm-1 for ch_stretch
	return(silence_region(img,[2700,4000]))
	
def silence_pre_fingerprint(img): # predefined wavenumbers in cm-1 for pre-fingerprint region
	return(silence_region(img,[0,475]))
	
def silence_region(img,region=[1800,2700]): # predefined wavenumbers in cm-1 for silent region
	chosen_freqs = np.logical_and(stdfreq >= region[0], stdfreq <= region[1])
	temp_img = img.copy()
	temp_img[:,:,chosen_freqs] = 0
	return(temp_img)

def import_submasks():
	maskfolderpath = input("Drag and drop submask folder into this prompt: ")
	maskfolderpath = maskfolderpath.replace('"','')
	maskfolderpath = maskfolderpath.replace("'",'')
	os.chdir(maskfolderpath)
	maskfiles = os.listdir()
	global masks, masknames
	masknames = []
	masks = []
	for file in maskfiles:
		masknames.append(os.path.splitext(file)[0])
		img = plt.imread(file)[:,:]
		img = np.mean(img,axis=2)
		masks.append(img < np.mean(img)) # Select "black" pixels as "in" the mask
	# Create a mask that covers all the selected regions. Basically OR all the masks together.
	global totalmask
	totalmask = np.full_like(masks[0],False)
	for mask in masks:
		totalmask = (totalmask | mask)
	
def import_main_mask():
	mainmaskpath = input("Drag and drop main mask image into this prompt: ")
	mainmaskpath = mainmaskpath.replace('"','')
	mainmaskpath = mainmaskpath.replace("'",'')
	img = plt.imread(mainmaskpath)[:,:]
	img = np.mean(img,axis=2)
	global mainmask
	#mainmask = (img < np.mean(img)+1) # Select "black" pixels as "in" the mask
	mainmask = (img < 255) # Select non-white pixels as "in" the mask

def fill_main_mask():
	global mainmask
	mainmask = np.full([xlen,ylen],True)
	return
		
def perform_PCA(mask, img=stdimg, n_components = 'mle', threshold=0.1):
	global pca, nmaincomponents
	pca = PCA(n_components = n_components, svd_solver = 'full')
	#pca = PCA(n_components = 100, svd_solver = 'full')
	pca.fit(img[mask])
	# Get enough components to represent more than 0.1 pixel worth of information
	# Hence 0.1/(xlen*ylen). Okay, I modified it to be settable by the user.
	nmaincomponents = len(pca.explained_variance_ratio_[pca.explained_variance_ratio_ > threshold / (xlen*ylen)])
	
def get_PCs(mask,img=stdimg):
	return(pca.transform(img[mask])[:,0:nmaincomponents])
	
def get_YXs(mask):
	return(np.stack((yi[mask],xi[mask]),axis=1))

def get_YX_PCs(mask,img=stdimg):
	return(np.concatenate([get_YXs(mask),get_PCs(mask,img)],axis=1))
	
def export_stdimg_pickle():
	chworkingdir()
	filename = input('Please give filename for standard image: ')
	with open(filename + '.pickle', 'wb') as file_handle:
		pickle.dump({'stdfreq' : stdfreq, 'stdimg' : stdimg}, file_handle, protocol=4) 
		
def import_stdimg_pickle():
	path = input("Drag and drop standard image pickle into this prompt: ")
	path = path.replace('"','')
	path = path.replace("'",'')
	global stdimg, stdfreq
	with open(path, 'rb') as file_handle:
		tempdict = pickle.load(file_handle)
		stdfreq = tempdict['stdfreq']
		stdimg = tempdict['stdimg']
		del tempdict

def export_nrmimg_pickle():
	chworkingdir()
	filename = input('Please give filename for z-score image: ')
	with open(filename + '.pickle', 'wb') as file_handle:
		pickle.dump({'stdfreq' : stdfreq, 'nrmimg' : nrmimg}, file_handle, protocol=4) 
		
def import_nrmimg_pickle():
	path = input("Drag and drop z-score image pickle into this prompt: ")
	path = path.replace('"','')
	path = path.replace("'",'')
	global nrmimg, stdfreq
	with open(path, 'rb') as file_handle:
		tempdict = pickle.load(file_handle)
		stdfreq = tempdict['stdfreq']
		nrmimg = tempdict['nrmimg']
		del tempdict
		
def export_pca_pickle():
	chworkingdir()
	filename = input('Please give filename for PCA variable export: ')
	with open(filename + '.pickle', 'wb') as file_handle:
		pickle.dump(pca, file_handle, protocol=4) 
		
def import_pca_pickle():
	path = input("Drag and drop PCA pickle into this prompt: ")
	path = path.replace('"','')
	path = path.replace("'",'')
	global pca
	with open(path, 'rb') as file_handle:
		pca = pickle.load(file_handle)
		
def export_scaler_pickle():
	chworkingdir()
	filename = input('Please give filename for scaler variable export: ')
	with open(filename + '.pickle', 'wb') as file_handle:
		pickle.dump(scaler, file_handle, protocol=4) 
		
def import_scaler_pickle():
	path = input("Drag and drop scaler pickle into this prompt: ")
	path = path.replace('"','')
	path = path.replace("'",'')
	global scaler
	with open(path, 'rb') as file_handle:
		scaler = pickle.load(file_handle)
		
def export_pca_components_csv():
	chworkingdir()
	filename = input('Please give filename for PCA component CSV export: ')
	pca_components_df = pd.DataFrame(pca.components_[0:nmaincomponents])
	pca_components_df.columns = stdfreq
	pca_components_df.index = [ 'PC' + str(i) for i in range(1,nmaincomponents+1)]
	pca_components_df.to_csv(filename + '.csv')
	
def export_pca_mean_csv():
	chworkingdir()
	filename = input('Please give filename for PCA mean CSV export: ')
	pca_components_df = pd.DataFrame(pca.mean_)
	pca_components_df.index = stdfreq
	pca_components_df.columns = ['Intensity']
	pca_components_df.to_csv(filename + '.csv')	
	
def export_normalize_mean_csv():
	chworkingdir()
	filename = input('Please give filename for Normalize mean CSV export: ')
	nrm_components_df = pd.DataFrame(scaler.mean_)
	nrm_components_df.index = stdfreq
	nrm_components_df.columns = ['Intensity']
	nrm_components_df.to_csv(filename + '.csv')		

def export_normalize_scale_csv():
	chworkingdir()
	filename = input('Please give filename for Normalize scale CSV export: ')
	nrm_components_df = pd.DataFrame(scaler.scale_)
	nrm_components_df.index = stdfreq
	nrm_components_df.columns = ['Intensity']
	nrm_components_df.to_csv(filename + '.csv')		

def export_FCS(mask,filename,img=stdimg):
	channel_names = ['Y', 'X']
	for name in ['PC' + str(i) for i in range(1,nmaincomponents+1)]:
		channel_names.append(name)
	with open(filename + '.fcs', 'wb') as file_handle:
		flowio.create_fcs(	file_handle,
							get_YX_PCs(mask,img).flatten(),
							channel_names)
							
def export_all_FCS(img=stdimg):
	chworkingdir()
	for i in range(len(masknames)):
		export_FCS(masks[i],masknames[i],img)	
	
def chworkingdir():
	workingdir = input("Drag and drop working directory into this prompt: ")
	if workingdir != '':
		workingdir = workingdir.replace('"','')
		workingdir = workingdir.replace("'",'')
		os.chdir(workingdir)
		
def cmdgetfilepath(message):
	filepath = input(message)
	filepath = filepath.replace('"','')
	filepath = filepath.replace("'",'')
	return(filepath)
	

# class ThreadedTCPRequestHandler(socketserver.StreamRequestHandler):
	# def handle(self):
		# self.data = self.rfile.readline().strip()
		# print(self.data)
		# cur_thread = threading.current_thread()
		# response = bytes("{}: {}".format(cur_thread.name, self.data), 'ascii')
		# self.wfile.write(response)

# class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    # pass

# def connectToMatlab(PORT=9999):
	# HOST = "localhost"
	# server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
	# ip, port = server.server_address
	# server_thread = threading.Thread(target=server.serve_forever)
   # # Exit the server thread when the main thread terminates
	# server_thread.daemon = True
	# server_thread.start()
	# print("Server loop running in thread:", server_thread.name)
	# input("Press any key to kill server running on port " + str(port))
	# server.shutdown()
	
def updateSpadeMask(maskpath, color):
	try:
		masklist = np.genfromtxt(maskpath,delimiter=',',dtype=int)
		masklen = masklist.shape[1]
		mask = np.full_like(crikit_data.hsi.data[:,:,0],False,dtype=bool)
		for i in range(masklen):
			 mask[masklist[0,i],masklist[1,i]]= True
	except:
		return	
	spade_out = {'mask' : np.flipud(mask), 'color' : color}
	if (len(crikit_data.img_overlays) > 0) : 
		crikit_data.img_overlays.pop()
	crikit_data.img_overlays.append(spade_out)
	crikit_data.changeSlider()
	

def thread_LoopUpdateSpadeMask(maskpath, color):
	print("Updated")
	updateSpadeMask(maskpath,color)
	mask_lastmodified = os.path.getmtime(maskpath)
	while threadWillLive:
		mask_lastmodified_new = os.path.getmtime(maskpath)
		if (mask_lastmodified_new > mask_lastmodified):
			#time.sleep(0.3)
			updateSpadeMask(maskpath, color)
			print("Updated")
			mask_lastmodified = mask_lastmodified_new
		time.sleep(0.1)
		
def updateSpadeMaskContinuously(color=[0,255,0,255]):
	commsMaskPath = cmdgetfilepath("Please drag and drop the CSV comm file MATLAB is updating: ")
	global threadWillLive
	threadWillLive = True
	threadHandle = threading.Thread(target=thread_LoopUpdateSpadeMask, args=(commsMaskPath,color),daemon=True)
	threadHandle.start()
	input("Press any key to stop listening for updates to the CSV comm file")
	threadWillLive = False
	threadHandle.join()
	
	

	