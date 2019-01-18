import matplotlib.pyplot as pl
import numpy as np
from pandas.io.parsers import read_csv
import scipy.optimize as opt
#from sklearn.preprocessing import PolynomialFeatures as pf
from scipy.io import loadmat
from sklearn import decomposition
#import pdb; pdb.set_trace()

def loadMatrix(file):
	return np.loadtxt(file)

def saveMatrix(file, X):
	return np.savetxt(file, X)

def sigmoide(z):
	return 1 / (1 + np.exp(-1*z))

def derivadaSigmoide(z):
	return sigmoide(z)*(1-sigmoide(z))

def pesosAleatorios(L_in, L_out):
	epsilon = 6**(0.5) / (L_in + L_out)
	return np.random.rand(L_in, L_out + 1) * 2 * epsilon - epsilon

def backprop(params_rn, num_entradas, num_ocultas , num_etiquetas , X, y, reg):
	theta1 = np.reshape(params_rn[:num_ocultas*(num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
	theta2 = np.reshape(params_rn[num_ocultas*(num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
	m = len(X)
	#Input
	ones_columns_input = np.array(np.ones(m))
	a1 = np.insert(X, 0,ones_columns_input, axis = 1)
	#hidden_layer
	z2 = np.dot(theta1, a1.transpose())
	a2 = sigmoide(z2)
	one_columns_hidden = np.array(np.ones(m))
	a2 = np.insert(a2, 0, one_columns_hidden, axis = 0)
	                              
	#Output_layer
	z3 = np.dot(theta2, a2)
	h = sigmoide(z3)
	#print h
	y_converted = (np.ravel(y) + 1)/2

	 #Cost
	regulation = (reg/float(2*m)) * (np.sum(theta1**2) + np.sum(theta2**2))
	J = np.sum(-y_converted * np.log(h) - (1 - y_converted)*np.log(1 - h)) * (1/float(m))
	J_regulated = J + regulation

	# Retro-Propagacion
	d3 = h - y_converted
	z2 = np.insert(z2, 0, np.ones(m), axis = 0)
	z2prima = derivadaSigmoide(z2)
	d2 = (np.dot(theta2.transpose(), d3))*z2prima

	#Gradient
	delta2 = np.dot(d3,a2.transpose())
	delta1 = np.dot(d2, a1)
	                              
	#Regularization
	D1 = (delta1[1:,:]/float(m) + theta1*reg/float(m)).ravel()
	D2 = (delta2/float(m) + theta2*reg/float(m)).ravel()
	gradient = np.r_[D1, D2]
	print J_regulated
	return J_regulated, gradient

def forwards(params, X, num_entradas, num_ocultas, num_etiquetas):
	theta1 = np.reshape(params[:num_ocultas*(num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
	theta2 = np.reshape(params[num_ocultas*(num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
	m = len(X)
	#Input
	ones_columns_input = np.array(np.ones(m))
	a1 = np.insert(X, 0,ones_columns_input, axis = 1)
	#hidden_layer
	z2 = np.dot(theta1, a1.transpose())
	a2 = sigmoide(z2)
	one_columns_hidden = np.array(np.ones(m))
	a2 = np.insert(a2, 0, one_columns_hidden, axis = 0)
	#Output_layer
	z3 = np.dot(theta2, a2)
	return sigmoide(z3)

def parametros(params, input_size, hidden_size, num_labels, X, y, reg):
	result = opt.minimize(fun=backprop, x0=params,
						 args=(input_size, hidden_size,
						 num_labels, X, y, reg),
						 method='TNC', jac=True,
						 options={'maxiter': 70})
	return result.x

def evaluacion(params, X, Y, num_entradas, num_ocultas, num_etiquetas):
	h = forwards(params,X,num_entradas,num_ocultas,num_etiquetas)
	z = (np.ravel(h) >= 0.5)
	Y = (np.ravel(Y) + 1)/2
	z = map((lambda x,y: x == y), z, Y)
	print Y
	print z
	return np.ravel(sum(z)/float(len(z))*100)[0]

def crossValidation(params, num_entradas, num_ocultas, num_etiquetas):
	X = loadMatrix('arcene_valid.data')
	pca = decomposition.PCA(n_components=100)
	pca.fit(X)
	#X = pca.transform(X)
	h = forwards(params, X, num_entradas, num_ocultas, num_etiquetas)
	z = (np.ravel(h) >= 0.5)
	return np.ravel(sum(z)/float(len(z))*100)[0]

def bestRegresion(theta1, theta2, num_entradas, num_ocultas, num_etiquetas, X, y, bound_left, bound_right, iterations):
	regs = np.linspace(bound_left, bound_right, num=iterations)
	params = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
	sol = []
	for i in regs:
		gradiente = parametros(params, num_entradas, num_ocultas, num_etiquetas, X, y, i)
		e = evaluacion(gradiente, X, y, num_entradas, num_ocultas, num_etiquetas)
		sol += [(i, e)]
	return sol

def test():
	reg = 3.5
	X = loadMatrix('arcene_train.data')

	pca = decomposition.PCA(n_components=100)
	pca.fit(X)
	#X = pca.transform(X)

	y = loadMatrix('arcene_train.labels')
	y = y.astype(int)
	num_entradas = X.shape[1]
	num_ocultas = 1000
	num_etiquetas = 1

	#theta1 = pesosAleatorios(num_ocultas, X.shape[1])
	#theta2 = pesosAleatorios(num_etiquetas, num_ocultas)
	theta1 = loadMatrix('theta1.out')
	theta2 = loadMatrix('theta2.out')

	#saveMatrix('theta1.out', theta1)
	#saveMatrix('theta2.out', theta2)

	#print bestRegresion(theta1, theta2, num_entradas, num_ocultas, num_etiquetas, X, y, 3.0, 5.0, 5)
	
	#coste, gradiente = backprop(np.concatenate((np.ravel(theta1), np.ravel(theta2))), num_entradas, num_ocultas, num_etiquetas, X, y, reg)
	gradiente = parametros(np.concatenate((np.ravel(theta1), np.ravel(theta2))), num_entradas, num_ocultas, num_etiquetas, X, y, reg)
	print evaluacion(gradiente, X, y, num_entradas, num_ocultas, num_etiquetas)
	#cvX = loadMatrix('arcene_valid.data')
	#cvY = loadMatrix('arcene_valid.labels').astype(int)
	#pca.fit(cvX)
	#cvX = pca.transform(cvX)
	#print evaluacion(gradiente, cvX, cvY, num_entradas, num_ocultas, num_etiquetas)
	

test()