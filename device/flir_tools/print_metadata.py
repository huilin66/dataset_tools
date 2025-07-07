# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:14:42 2020

@author: RDanjoux
FLIR Support

This sample will find and print some important metadata from each frame in the file.
"""

# import the module
import fnv
import fnv.reduce
import fnv.file   

# modules for math and plotting
import numpy as np        
from matplotlib import pyplot as plt  

# import for timestamp handling
import datetime

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

#%% print all metadata 
# define function to print all metadata for each frame

def print_metadata(frame_info):
    for f in frame_info:
        # simply print the name and value of each metadatum
        print("{}: {}".format(f['name'], f['value']))
        
        # look for the timestamp in the frame. 
        # The time stamp is stored in a hard to read format and we can convert it to something more user friendly.
        if f['name'] == 'Time' :
            # get the timestamp string
            DateStr = f['value']
            
            # extract values from the timestamp string
            Microsecond = int(DateStr[-6])
            Second = int(DateStr[-9:-7])
            Minute = int(DateStr[7:9])
            Hour = int(DateStr[4:6])
            DayOfYear = int(DateStr[0:3])
            
            # no year info is saved in the timestamp, so we grab it from PC clock
            Year = datetime.datetime.now().year
            
            # the month and day can be determined from the day of the year
            Month = datetime.datetime.strptime(str(DayOfYear), '%j').month
            Day = datetime.datetime.strptime(str(DayOfYear), '%j').day
            
            # combine all date info into a datetime object
            TimeStamp = datetime.datetime(tzinfo=datetime.timezone.utc, # timezone is UTC
                                          year=Year,
                                          month=Month,
                                          day=Day,
                                          hour=Hour,
                                          minute=Minute,
                                          second=Second,
                                          microsecond=Microsecond)
            
            # print a pretty-looking timestamp
            print("{}: {}".format("Pretty Time", 
                                  TimeStamp.strftime("%B %d th %H:%M:%S") # you can easily format datetimes
                                  ))


#%% import radiometric data
# import the data from the file FOR EACH FRAME of the recording

print("Length of recoring is {} frames".format(im.num_frames))

# iterate through the recording, for each frame
for i in range(im.num_frames):
    im.get_frame(i)                         # get the current frame
    
    # print the frame number, relative to the start of recording
    # the frame number field is relative to the start of image acquisition
    print("\nMetadata for frame {}".format(i))
    
    # print all the metadata
    print_metadata(im.frame_info)
    

#%% finish and dispose image
# this will delete the im variable and free up space in memory
# data will still be available in the workspace

im = None                                   # done with the file