#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 14:19:12 2023

@author: Aidanklemmer
"""

import numpy as np
import matplotlib.pyplot as plt



#data = np.loadtxt('/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/PRL/Figures/WFI_Ra_10197.csv', delimiter=',')

data = np.loadtxt('/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/PRL/Figures/Default Dataset 1.csv', delimiter=',')


x = data[:,0]*1000
y = data[:,1]



fig1, (ax1) = plt.subplots(1,1,figsize=(4.5,1))



ax1.plot(x,y, '-', color=[0,0.5,1], lw=0.75)


ax1.spines['top'].set_visible(False)
#ax1.spines['right'].set_visible(False)
#ax1.spines['bottom'].set_visible(False)
#ax1.spines['left'].set_visible(False)

ax1.axhline(0, ls=':', color='black', lw=0.5)


ygrid = np.linspace(-0.05, 0.025, 4)
ax1.set_yticks(ygrid)
ax1.set_yticklabels(ygrid)

#xgrid = np.linspace(-5, 210, 6)
#ax1.set_xticks(xgrid)
#ax1.set_xticklabels(xgrid)

ax1.set_yticks(ax1.get_yticks())
ax1.set_yticklabels(['{:.3f}'.format(x) for x in ax1.get_yticks()])

#ax1.set_xticks(ax1.get_xticks())
#ax1.set_xticklabels(['{:.2f}'.format(x) for x in ax1.get_xticks()])




ax1.set_xlabel('Distance [$\mathrm{\mu}$m]')
ax1.set_ylabel('Height [$\mathrm{\mu}$m]')

ax1.set_xlim(-2,210)
ax1.set_ylim(-0.05,0.025)

plt.rcParams["font.family"] = "sans-serif"

fig1.set_dpi(600)



