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
from sklearn.metrics import accuracy_score
from matplotlib import colors

#plt.rcParams["figure.figsize"] = (10,10)


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

def plot_pdfR(X, Y, filename, model, color_map, labels = None, n_components= 2):

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


# 	cfset = ax.contourf(xx, yy, f2, cmap='BuGn', alpha = alpha[1])
# 	cset = ax.contour(xx, yy, f2, colors='k')
# 	ax.clabel(cset, inline=1, fontsize=10)

	fig = plt.figure(figsize=(10,10))
	currentAxis = plt.gca()
	x_axis = 0
	y_axis = 1
	for i in range(n_components):

		pred1 = predictions[:,i]
		f1 = np.reshape(pred1.T, xx.shape)


		alpha = softmax(model.logits_k.numpy())

		ax = fig.gca()
		ax.set_xlim(xmin, xmax)
		ax.set_ylim(ymin, ymax)
		cfset = ax.contourf(xx, yy, f1, cmap='BuGn', alpha = alpha[i])
		# ax.imshow(np.rot90(f1), cmap='BuGn', extent=[xmin, xmax, ymin, ymax])
		cset1 = ax.contour(xx, yy, f1, colors='k', alpha= alpha[i])
		ax.clabel(cset1, inline=1, fontsize=10)

		low = model.lower.numpy()[i]
		upp = model.upper.numpy()[i]
		currentAxis.add_patch(Rectangle((low[x_axis], low[y_axis]), upp[x_axis]-low[x_axis], upp[y_axis]-low[y_axis], fill=None,
	                                    edgecolor=color_map[i], alpha=1))
		plt.scatter(*model.mu.numpy()[i,[x_axis, y_axis]], marker='^')

	if labels is not None:
		plt.scatter(X, Y, c = [color_map[labels[i]] for i in range(X.shape[0])], alpha = 0.5)
	plt.savefig(filename)
	plt.close(fig)
	plt.clf()




def plot_boundaries_hyperrect(X, y, x_axis, y_axis, file_name, color_map, sTGMA, black_box, steps=1000):

	X1 = X[:,0]
	X2 = X[:,1]
	cmap = colors.ListedColormap(list(color_map.values())[:len(np.unique(y))])
	# Define region of interest by data limits
	deltaX = (max(X1) - min(X1))/10
	deltaY = (max(X2) - min(X2))/10

	xmin, xmax = min(X1) - deltaX, max(X1) + deltaX
	ymin, ymax = min(X2) - deltaY, max(X2) + deltaY

	x_span = np.linspace(xmin, xmax, steps)
	y_span = np.linspace(ymin, ymax, steps)
	xx, yy = np.meshgrid(x_span, y_span)

	# Make predictions across region of interest
	labels_bb = black_box(np.c_[xx.ravel(), yy.ravel()])
	labels_bb = np.argmax(labels_bb, axis = 1)

	labels_stgma = sTGMA.predict(np.c_[xx.ravel(), yy.ravel()].astype(np.float32)).numpy()
	#Decision boundary sTGMA
	#plt.subplot(2, 1, 1)
	z1 = labels_stgma.reshape(xx.shape)
	
	ranges = np.linspace(z1.min(), z1.max(), len(color_map.values())+1)
	norm = colors.BoundaryNorm(ranges, cmap.N)


	plt.contourf(xx, yy, z1, alpha=0.2, cmap = cmap, norm=norm)

	plt.scatter(X1, X2, c = [color_map[y[i]] for i in range(X.shape[0])],  edgecolor='k', lw=0, cmap="Set1")
	currentAxis = plt.gca()
	for c in range(len(sTGMA.y_unique)):
		for i in range(sTGMA.n_components):
			low = sTGMA.lower[c,i].numpy()

			upp = sTGMA.upper[c,i].numpy()

			currentAxis.add_patch(Rectangle((low[x_axis], low[y_axis]), upp[x_axis]-low[x_axis], upp[y_axis]-low[y_axis], fill=None,
	                                    edgecolor=color_map[c], alpha=1))
# 			plt.scatter(*mu[i,[x_axis, y_axis]], marker='^')
	#plt.title(f"sTGMA, accuracy_score: {accuracy_score(y, sTGMA.predict(X))}")
	plt.savefig(f"{file_name}_stgma.png")
	plt.clf()
	#Decision boundary ANN
	#plt.subplot(2, 1, 2)
	z2 = labels_bb.reshape(xx.shape)
	plt.contourf(xx, yy, z2, alpha=0.2, cmap =cmap, norm = norm)
	plt.scatter(X1, X2, c = [color_map[y[i]] for i in range(X.shape[0])],  edgecolor='k', lw=0, cmap="Set1")
	#plt.title(f"ANN, accuracy_score: {accuracy_score(y, np.argmax(black_box.predict(X).numpy(), axis = 1))}")

	#plt.suptitle(f'fidelity: {accuracy_score(sTGMA.predict(X), np.argmax(black_box.predict(X).numpy(), axis = 1))}')
	print("saving boundaries************************", file_name)
	plt.savefig(f"{file_name}_bb.png")
	plt.clf()

def plot_boundary(X, y, x_axis, y_axis, file_name, color_map, model, rect = False, steps=1000):

	X1 = X[:,0]
	X2 = X[:,1]

	# Define region of interest by data limits
	deltaX = (max(X1) - min(X1))/10
	deltaY = (max(X2) - min(X2))/10

	xmin, xmax = min(X1) - deltaX, max(X1) + deltaX
	ymin, ymax = min(X2) - deltaY, max(X2) + deltaY

	x_span = np.linspace(xmin, xmax, steps)
	y_span = np.linspace(ymin, ymax, steps)
	xx, yy = np.meshgrid(x_span, y_span)

	# Make predictions across region of interest
	labels_model = model.predict(np.c_[xx.ravel(), yy.ravel()])
	if len(labels_model.shape) == 2:
		labels_model = np.argmax(labels_model, axis = 1)

	#Decision boundary sTGMA
	z1 = labels_model.reshape(xx.shape)
	plt.contourf(xx, yy, z1, alpha=0.5)
	plt.scatter(X1, X2, c = [color_map[y[i]] for i in range(X.shape[0])],  edgecolor='k', lw=0, cmap="Set1")
	if rect:
		currentAxis = plt.gca()
		for c in range(len(model.y_unique)):
			for i in range(model.n_components):
				low = model.lower[c,i].numpy()

				upp = model.upper[c,i].numpy()

				currentAxis.add_patch(Rectangle((low[x_axis], low[y_axis]), upp[x_axis]-low[x_axis], upp[y_axis]-low[y_axis], fill=None,
		                                    edgecolor=color_map[c], alpha=1))
# 			plt.scatter(*mu[i,[x_axis, y_axis]], marker='^')
	plt.title("Decision Boundary")


	#Decision boundary ANN
	plt.savefig(file_name, dpi=150)
	plt.clf()

def plot_pdf_hyperrectangles(X, Y, x_axis, y_axis, lower, upper, nb_hyperrectangles, file_name, color_map, mu):


	currentAxis = plt.gca()
	y_unique = np.unique(Y)
	for c in range(len(y_unique)):
		for i in range(nb_hyperrectangles):
			low = lower[c,i]

			upp = upper[c,i]

			currentAxis.add_patch(Rectangle((low[x_axis], low[y_axis]), upp[x_axis]-low[x_axis], upp[y_axis]-low[y_axis], fill=None,
	                                    edgecolor=color_map[i], alpha=1))
			plt.scatter(*mu[c,i,[x_axis, y_axis]], marker='^')

	plt.scatter(X[:,x_axis], X[:,y_axis], color=[color_map[i] for i in Y])
	plt.savefig(file_name, dpi = 150)
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
