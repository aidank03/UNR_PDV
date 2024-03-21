#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 22:43:16 2023

@author: Aidanklemmer
"""


import matplotlib.pyplot as plt
from PIL import Image
import matplotlib as mpl
import numpy as np


file_path = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/PRL/Figures/10194_334.tiff'


#file_path = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_2/12F_94_10.tif'

#file_path = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_2/12F_94_6.tif'

#file_path = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_2/12F_94_3_#2.tif'



image_1 = Image.open(file_path)

#image_1 = image_1.rotate(2) 


# Create the figure and axes for the plot
fig1, (ax) = plt.subplots(figsize=(6, 4))
#fig2, (ax2) = plt.subplots(figsize=(6, 4))

#colormap_name = 'plasma'

width, height = image_1.size

#print(width)
#print(height)


#im1 = im.crop((left, top, right, bottom))
crop_image_cp = image_1.crop((30, 30, 1000, 892))


image = np.flip(crop_image_cp)



# Recolor the image with the desired colormap
#colormap = plt.cm.get_cmap(colormap_name)
image_cp = np.array(image)

    
ax.imshow(image_cp, cmap='plasma')

#ax2.imshow(recolored_image)



            #ax.set_title(titles[image_index])
ax.axis('off')
#ax2.axis('off')

print(np.min(image_cp))


# colorbar
norm = mpl.colors.Normalize(vmin=np.min(image), vmax=np.max(image))
#N = len()
cmap = plt.get_cmap('turbo')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
#sm.set_array([])


#plt.colorbar(sm, ticks=np.linspace(0, 2, N), format="%0.1f", location='top')
#cbar = fig1.colorbar(sm,format="%0.0f", location='right', pad=0.005)
#cbar.set_label('Scaled Intensity [arb. units]', fontsize = 8)
#cbar.tick_params(axis='both', which='major', pad=2,size=8)
#cbar.ax.tick_params(labelsize=7)


#plt.rcParams["font.family"] = "sanserif"
   

#plt.colorbar(plt.cm.ScalarMappable(cmap='turbo'));
    
fig1.set_dpi(1000)
#fig2.set_dpi(1400)
    # Show the plot
#plt.show()










