import matplotlib.pyplot as plt
import numpy as np
from pandas.io.parsers import read_csv
import scipy.optimize as opt
#from sklearn.preprocessing import PolynomialFeatures as pf
from scipy.io import loadmat
from sklearn import decomposition
#import pdb; pdb.set_trace()
from sklearn import preprocessing

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
	return J_regulated, gradient

def coste(params, X, num_entradas, num_ocultas, num_etiquetas):
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
	h = coste(params,X,num_entradas,num_ocultas,num_etiquetas)
	z = (np.ravel(h) >= 0.5)
	Y = (np.ravel(Y) + 1)/2
	print z, h, Y
	z = map((lambda x,y: x == y), z, Y)
	print np.array(list(zip(z, Y))), sum(Y)
	return np.ravel(sum(z)/float(len(z))*100)[0]

def crossValidation(params, X, num_entradas, num_ocultas, num_etiquetas):
	pca = decomposition.PCA(n_components=100)
	pca.fit(X)
	#X = pca.transform(X)
	h = coste(params, X, num_entradas, num_ocultas, num_etiquetas)
	z = (np.ravel(h) >= 0.5)
	return sum(z)

def bestRegresion(theta1, theta2, num_entradas, num_ocultas, num_etiquetas, X, y, bound_left, bound_right, iterations):
	regs = np.linspace(bound_left, bound_right, num=iterations)
	params = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
	sol = []
	for i in regs:
		gradiente = parametros(params, num_entradas, num_ocultas, num_etiquetas, X, y, i)
		e = evaluacion(gradiente, X, y, num_entradas, num_ocultas, num_etiquetas)
		sol += [(i, e)]
	return sol

def curvasAprendizaje(errorT, errorVal, reg, num_ocultas, iterations):
	plt.figure()
	plt.xlabel('Numero de ejemplos de entrenamiento')
	plt.ylabel('Error')
	plt.title("$lambda$ = " + reg + ", nodos ocultos = " + num_ocultas)
	plt.plot(np.array(range(len(errorT)))*iterations, errorT, '-', color='blue', label='Entreno')
	plt.plot(np.array(range(len(errorVal)))*iterations, errorVal, '-', color='orange', label='Validacion cruzada')
	plt.legend()
	plt.savefig('curva_aprendizaje_' + reg + '_' + num_ocultas + '_' + str(iterations) + '.png')

def train(X, y, cvX, cvY, pesos, num_entradas, num_ocultas, num_etiquetas, iterations, reg):
	Jts = []
	Jcvs = []
	for i in range(1, len(X)/iterations):
		print i
		pesos = parametros(pesos, num_entradas, num_ocultas, num_etiquetas, X[0:i*iterations], y[0:i*iterations], reg)
		Jts += [backprop(pesos, num_entradas, num_ocultas, num_etiquetas, X[0:i*iterations], y[0:i*iterations], reg)[0]]
		Jcvs += [backprop(pesos, num_entradas, num_ocultas, num_etiquetas, cvX[0:i*iterations], cvY[0:i*iterations], reg)[0]]
	curvasAprendizaje(Jts, Jcvs, str(reg), str(num_ocultas), iterations)

def test():
	reg = 10
	X = loadMatrix('arcene_train.data')

	pca = decomposition.PCA(n_components=100)
	pca.fit(X)
	#X = pca.transform(X)

	y = loadMatrix('arcene_train.labels')
	y = y.astype(int)
	num_entradas = X.shape[1]
	num_ocultas = 200
	num_etiquetas = 1
	iterations = 2

	theta1 = pesosAleatorios(num_ocultas, X.shape[1])
	theta2 = pesosAleatorios(num_etiquetas, num_ocultas)
	#saveMatrix('theta1.out', theta1)
	#saveMatrix('theta2.out', theta2)

	#theta1 = loadMatrix('theta1.out')
	#theta2 = loadMatrix('theta2.out')
	cvX = loadMatrix('arcene_valid.data')
	cvY = loadMatrix('arcene_valid.labels').astype(int)
	pca.fit(cvX)
	#cvX = pca.transform(cvX)

	#print bestRegresion(theta1, theta2, num_entradas, num_ocultas, num_etiquetas, X, y, 20, 40.0, 3)
	
	#coste, gradiente = backprop(np.concatenate((np.ravel(theta1), np.ravel(theta2))), num_entradas, num_ocultas, num_etiquetas, X, y, reg)

	#pesos = loadMatrix('weights.out')
	pesos = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
	train(X, y, cvX, cvY, pesos, num_entradas, num_ocultas, num_etiquetas, iterations, reg)
	#print evaluacion(pesos, X, y, num_entradas, num_ocultas, num_etiquetas)
	#print evaluacion(pesos, cvX, cvY, num_entradas, num_ocultas, num_etiquetas)
	#tX = loadMatrix('arcene_test.data')
	#print crossValidation(pesos, tX, num_entradas, num_ocultas, num_etiquetas)
	#saveMatrix('weights.out', pesos)
	

test()