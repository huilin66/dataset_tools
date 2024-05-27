# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:56:29 2020

@author: RDanjoux
FLIR Support

This sample will modify some important parameters before importing all of the data.
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

#%% select units

# set desired units
if im.has_unit(fnv.Unit.TEMPERATURE_FACTORY):
    # set units to temperature, if available
    im.unit = fnv.Unit.TEMPERATURE_FACTORY
    im.temp_type = fnv.TempType.CELSIUS         # set temperature unit
else:
    # if file has no temperature calibration, use counts instead
    im.unit = fnv.Unit.COUNTS
    
#%% modify object parameters
# What follows is just an example!
# It shows how to modify the emissivity.
# The same procecedure can be applied to any other Object Parameters. Find their names in the workspace.
# Setting new Object Parameters shall be done prior to calling for frames in final definition.

# get all object parameters
ObjParam=im.object_parameters

# change emissivity
ObjParam.emissivity = 0.9

# change reflected temperature
# value is in Kelvin
ObjParam.reflected_temp = 293.15

# set all object parameters in im
im.object_parameters = ObjParam

#%% import radiometric data
# import the data from the file FOR EACH FRAME of the recording

print("emissivity: {:2.2f} \nreflected temperature: {:2.2f}".format(im.object_parameters.emissivity,
             im.object_parameters.reflected_temp))   # print new object parameters

data = []

# iterate through the recording, if there are multiple frames
for i in range(im.num_frames):
    im.get_frame(i)                         # get the current frame
    
    # convert image to np array
    # this makes it easy to find min/max
    data.append(np.array(im.final, copy=False).reshape(
        (im.height, im.width)))

#%% finish and dispose image
# this will delete the im variable and free up space in memory
# data will still be available in the workspace

im = None                                   # done with the file