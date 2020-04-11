#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:19:05 2020

@author: gnanfack
"""
import numpy as np

from matplotlib.patches import Rectangle

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from scipy.special import softmax



def plot_hyperrectangles(X, y, x_axis, y_axis, lower, upper, nb_hyperrectangles, file_name, color_map, mu):

	currentAxis = plt.gca()

	for i in range(nb_hyperrectangles):
		low = lower[i]

		upp = upper[i]

		currentAxis.add_patch(Rectangle((low[x_axis], low[y_axis]), upp[x_axis]-low[x_axis], upp[y_axis]-low[y_axis], fill=None,
                                    edgecolor=color_map[i], alpha=1))
		plt.scatter(*mu[i,[x_axis, y_axis]], marker='^')

	plt.scatter(X[:,x_axis], X[:,y_axis], color=[color_map[i] for i in y])


	plt.savefig(file_name, dpi = 150)
	plt.clf()

def plot_pdfR(X, Y, filename, model, color_map):

	x = X
	y = Y
	# Define the borders
	deltaX = (max(x) - min(x))/10
	deltaY = (max(y) - min(y))/10
	xmin = min(x) - deltaX
	xmax = max(x) + deltaX
	ymin = min(y) - deltaY
	ymax = max(y) + deltaY

	# Create meshgrid
	xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
	positions = np.vstack([xx.ravel(), yy.ravel()])

	values = np.vstack([x, y])
	predictions = model.compute_pdf(positions.T).numpy()
	pred1 = predictions[:,0]
	pred2 = predictions[:,1]
	f1 = np.reshape(pred1.T, xx.shape)
	f2 = np.reshape(pred2.T, xx.shape)


	alpha = softmax(model.logits_k.numpy())
	fig = plt.figure(figsize=(10,10))
	ax = fig.gca()
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	cfset = ax.contourf(xx, yy, f1, cmap='BuGn', alpha = alpha[0])
	# ax.imshow(np.rot90(f1), cmap='BuGn', extent=[xmin, xmax, ymin, ymax])
	cset1 = ax.contour(xx, yy, f1, colors='k')
	ax.clabel(cset1, inline=1, fontsize=10)

	cfset = ax.contourf(xx, yy, f2, cmap='BuGn', alpha = alpha[1])
	cset = ax.contour(xx, yy, f2, colors='k')
	ax.clabel(cset, inline=1, fontsize=10)
	currentAxis = plt.gca()
	currentAxis = plt.gca()
	x_axis = 0
	y_axis = 1
	for i in range(2):
	    low = model.lower.numpy()[i]
	    upp = model.upper.numpy()[i]
	    currentAxis.add_patch(Rectangle((low[x_axis], low[y_axis]), upp[x_axis]-low[x_axis], upp[y_axis]-low[y_axis], fill=None,
	                                    edgecolor=color_map[i], alpha=1))
	    plt.scatter(*model.mu.numpy()[i,[x_axis, y_axis]], marker='^')
	plt.savefig(filename)
	plt.close(fig)
	plt.clf()
# =============================================================================
# 	X, Y = np.meshgrid(X, Y)
#
# 	pos = np.empty(X.shape + (2,))
# 	pos[:, :, 0] = X[:,1]
# 	pos[:, :, 1] = X[:,1]
# 	Z = model.compute_pdf(pos).numpy()
# 	fig = plt.figure()
# 	ax1 = fig.add_subplot(2,1,1,projection='3d')
# 	ax1.plot_surface(X, Y, Z [:,:,0], rstride=3, cstride=3, linewidth=1, antialiased=True,
#                     cmap=cm.viridis)
# 	ax1.plot_surface(X, Y, Z[:,:,0], rstride=3, cstride=3, linewidth=1, antialiased=True,
#                     cmap=cm.viridis)
# 	ax1.view_init(55,-70)
# 	ax1.set_xticks([])
# 	ax1.set_yticks([])
# 	ax1.set_zticks([])
# 	ax1.set_xlabel(r'$x_1$')
# 	ax1.set_ylabel(r'$x_2$')
# 	ax2 = fig.add_subplot(2,1,2,projection='3d')
# 	ax2.contourf(X, Y, Z[:,:,0], zdir='z', offset=0, cmap=cm.viridis)
# 	ax2.contourf(X, Y, Z[:,:,1], zdir='z', offset=0, cmap=cm.viridis)
# 	ax2.view_init(90, 270)
# 	ax2.grid(False)
# 	ax2.set_xticks([])
# 	ax2.set_yticks([])
# 	ax2.set_zticks([])
# 	ax2.set_xlabel(r'$x_1$')
# 	ax2.set_ylabel(r'$x_2$')
# 	plt.savefig(f"{filename}_pdf.png")
# 	plt.close(fig)
# =============================================================================
