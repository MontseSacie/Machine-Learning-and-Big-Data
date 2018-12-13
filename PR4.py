import matplotlib.pyplot as pl
import numpy as np
from pandas.io.parsers import read_csv
import scipy.optimize as opt
#from sklearn.preprocessing import PolynomialFeatures as pf
from scipy.io import loadmat
import checkNNGradients


def sigmoide(z):
	return 1 / (1 + np.exp(-1*z))

def derivadaSigmoide(z):
	return sigmoide(z)*(1-sigmoide(z))

def pesosAleatorios(L_in, L_out):
	return np.random.rand(L_out, 1 + L_in) * 0.24 - 0.12

def backprop(params_rn, num_entradas, num_ocultas , num_etiquetas , X, y, reg): 
	theta1 = np.reshape(params_rn[:num_ocultas*(num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
	theta2 = np.reshape(params_rn[num_ocultas*(num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
	y = np.array(y).transpose()
	# capa oculta
	cX = np.matrix(X).transpose()
	mX = np.concatenate(([np.ones(len(X))], np.array(cX)), axis=0)
	z2 = np.dot(theta1, mX)
	a2 = sigmoide(z2)
	# capa de salida
	ma2 = np.concatenate(([np.ones(len(a2[0]))], np.array(a2)), axis=0)	
	z3 = np.dot(theta2, ma2)
	h = sigmoide(z3)
	logh = np.log(h)
	log1h = np.log(1 - h)
	ys = np.zeros(logh.shape)
	#for i in range(0, len(y)):
	#	ys[y[:]-1][i] = 1
	np.put(ys, y[0]-1, np.ones(y[0]).size)
	y = ys
	print z2.shape
	# retro-propagacion
	d3 = h - y
	z2prima = derivadaSigmoide(z2)
	d2 = (np.dot(theta2.transpose(), d3)[1:][:])*z2prima
	print d2.shape, a2.transpose().shape, d3.shape,h.transpose().shape
	# regularizacion
	#d3r = d3 + reg/float(len(mX)) * theta2
	#d2r = d2 + reg/float(len(mX)) * theta1
	gradiente = np.append( np.ravel(np.dot(d2,a2.transpose())), np.ravel(np.dot(d3,h.transpose())) ) *1/float(len(X))
	#print gradiente
	regulation = (reg/(float(2*len(X))))*(np.sum(theta1**2) + np.sum(theta2**2))
	coste = (1/(float(len(X)))) * np.sum(-y*logh - (1-y)*log1h) + regulation
	return coste, gradiente

def test():
	num_entradas = 400
	num_ocultas = 25
	num_etiquetas = 10
	reg = 1
	data = loadmat('ex4data1.mat')
	y = np.ravel(data ['y'])
	X = data ['X']
	weights = loadmat( 'ex4weights.mat' )
	theta1, theta2 = weights[ 'Theta1' ], weights[ 'Theta2' ]
	#print(backprop(np.concatenate((np.ravel(theta1), np.ravel(theta2))), num_entradas, num_ocultas, num_etiquetas, X, y, reg))
	"""params = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
	fmin = opt.minimize(fun=backprop, x0=params,
 					args=(num_entradas, num_ocultas,
 					num_etiquetas, X, y, reg),
 					method='TNC', jac=True,
 					options={'maxiter': 70})"""
	print(checkNNGradients.checkNNGradients(backprop, reg))

test()