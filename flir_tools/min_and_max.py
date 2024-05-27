# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:02:37 2020

@author: JGiaquin
FLIR Support

This sample will determine the location and value of the minimum and maximum temperatures in each image of the recording.
"""

# import the module
import fnv
import fnv.reduce
import fnv.file   

# modules for math and plotting
import numpy as np        
from matplotlib import pyplot as plt  

# modules for opening file
from tkinter import filedialog
import tkinter
import os

#%% browse to file

root = tkinter.Tk()
root.withdraw() 
root.call('wm', 'attributes', '.', '-topmost', True)

currdir = os.getcwd()
path = filedialog.askopenfilename(filetypes = (("Radiometric files", 
                                            # all radiometric extensions:
                                                ["*.seq", 
                                                 "*.jpg", 
                                                 "*.ats", 
                                                 "*.sfmov", 
                                                 "*.img"]), 
                                               ("All files", "*")))

im = fnv.file.ImagerFile(path)              # open the file
print(path)                         # print file name

#%% find min and max
# define function to find the values of min/max and then find the their positions in the image

def min_and_max(data):
    # initialize global values of interest
    global max_x, max_y, maxT, min_x, min_y, minT
    
    max_x, max_y = np.unravel_index(np.argmax(data), data.shape) # find location of max in the whole image
    maxT = data[(max_x, max_y)] # find value of the maximum
    
    # repeat for minimum temp
    min_x, min_y = np.unravel_index(np.argmin(data), data.shape)
    minT = data[(min_x, min_y)] 

#%% display image 
# choose desired units
# define function to show a color-scaled image with pointers that show the location of minimum and maximum temperatures

# set desired units
if im.has_unit(fnv.Unit.TEMPERATURE_FACTORY):
    # set units to temperature, if available
    im.unit = fnv.Unit.TEMPERATURE_FACTORY
    im.temp_type = fnv.TempType.KELVIN         # set temperature unit
else:
    # if file has no temperature calibration, use counts instead
    im.unit = fnv.Unit.COUNTS

# print out min and max
if im.temp_type == fnv.TempType.CELSIUS:
    unitstr = "°C"
elif im.temp_type == fnv.TempType.FAHRENHEIT:
    unitstr = "°F"
elif im.temp_type == fnv.TempType.KELVIN:
    unitstr = "K"
elif im.temp_type == fnv.TempType.COUNTS:
    unitstr = "counts"

def plot_image(data):
    # initialize global values of interest
    global max_x, max_y, maxT, min_x, min_y, minT
    
    # Clear current reference of a figure. This will improve display speed significantly
    plt.clf()
    
    # display image
    plt.imshow(data, 
                    cmap="afmhot",          # choose a color palette 
                    aspect="auto")          # set aspect ratio
    plt.colorbar(format='%.2f') # add color bar to image
    
    # show min and max on image
    plt.scatter(max_y, max_x, s=50, c='red', marker='+')
    plt.scatter(min_y, min_x, s=50, c='blue', marker='+')
    plt.axis('off') # remove axis labels for image
    plt.show()
    plt.pause(0.001)

#%% import radiometric data
# import the data from the file FOR EACH FRAME of the recording
# call the functions defined above to find and display the min and max

print("frame number; maximum ({}); minimum ({})".format(unitstr, unitstr))   # print heading for data

plt.figure("Temperatures of interest")       # open new figure

# iterate through the recording, if there are multiple frames
for i in range(im.num_frames):
    im.get_frame(i)                         # get the current frame
    
    # convert image to np array
    # this makes it easy to find min/max
    data = np.array(im.final, copy=False).reshape(
        (im.height, im.width))
     
    min_and_max(data)     # determine min and max
    plot_image(data)      # plot image
    
    # print data to console
    print("{:d}; {:2.2f}; {:2.2f}".format(i, maxT, minT))   

#%% finish and dispose image
# this will delete the im variable and free up space in memory
# data will still be available in the workspace

im = None                                   # done with the file

