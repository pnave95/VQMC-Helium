import numpy as np
import math
#import matplotlib
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt 
from datetime import datetime

from helium import LocalEnergy, ProportionalToPi, norm, SampleVarianceLocalEnergy
from MetropolisHastings import RandomVector, MetropolisAcceptance, UniformMetropolisSampler, ApproximateMetropolisAcceptanceRate


'''
Run a grid search over values of a,b:  Na x Nb grid, with a in [amin, amax], b in [bmin, bmax].
A fixed value of delta is used (==> a fixed proposal probability kernel is used)
'''
def FineGridSearch(amin, amax, Na, bmin, bmax, Nb, M=10000, delta=0.2):

	aa = np.linspace(amin,amax,int(Na))
	bb = np.linspace(bmin,bmax,int(Nb))

	Results = np.zeros((len(aa), len(bb)))

	for i in range(len(aa)):
		for j in range(len(bb)):
			a = aa[i]
			b = bb[j]
			energy = UniformMetropolisSampler(a, b, M, delta)
			Results[i][j] = energy


	print(Results)

	#compute minimum energy	
	minEnergy = np.nanmin(Results)
	
	# compute minimum energy parameters
	i, j = np.unravel_index(np.nanargmin(Results), Results.shape)
	alphaMinimizer = aa[i]
	betaMinimizer = bb[j]
	print("Minimum Energy = " + str(minEnergy) + " = " + str(Results[i][j]))
	print("Armgin alpha, beta = " + str(alphaMinimizer) + ", " + str(betaMinimizer))
	

	# Plot results

	fig = plt.figure()
	ax = plt.axes(projection='3d')

	# define 2d planes of axes
	avals = np.outer(aa,np.ones(len(bb)))
	bvals = np.outer(np.ones(len(aa)), bb)

	ax.plot_surface(avals, bvals, Results, cmap='viridis', edgecolor='none')

	title = r"Approximate Wavefunction Energy (M=" + str(M) + ", $\delta$=" + str(delta) + ")"
	ax.set_title(title)
	ax.set_xlabel(r"$\alpha$")
	ax.set_ylabel(r"$\beta$")
	

	plt.show()

	now = datetime.now()
	#year = now.strftime("%Y")
	#month = now.strftime("%m")
	#day = now.strftime("%d")
	date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

	metastring = "GridSearch_M=" + str(M) + "_delta=" + str(delta) +  "_timestamp=" + date_time
	figname = "Graph_3D_" + metastring + ".png"
	fig.savefig(figname, dpi=fig.dpi)

	# save data
	datastring = "Data_" + metastring + ".csv"
	np.savetxt(datastring, Results, delimiter=',')


	return Results



'''
This function runs an exhaustive grid search over a 50x50 grid for a in [0, amax], b in [0, bmax].  A fixed delta is used (i.e. a fixed proposal probability function is used)
'''
def GridSearch(amax=10, bmax=10, M=1000, delta=0.2):
	return FineGridSearch(0, amax, 50, 0, bmax, 50, M, delta)
	



# This function chooses M based on delta so that M*delta is always close to 25 if possible but so that M never goes above 100 or below 10
def ChooseM(delta):
	return min(100, max(10, math.ceil(25 / delta) ) )



'''
This function attempts to choose a maximum step size delta such that the Monte Carlo acceptance rate is as close to 0.5 as possible.  It does this by randomly trying 10 different values of delta and estimating their acceptance rates.  NOTE:  "maxTries" is currently an unused variable
'''
def AdaptivelyChooseDelta(a, b, deltaGuess=0.2, maxTries = 10):
	
	# check deltaGuess is within bounds
		
	delta = deltaGuess
	M = ChooseM(delta)
	leftDelta = 0.02
	rightDelta = 3.0

	if(deltaGuess < leftDelta):
		delta = leftDelta
	elif(deltaGuess > rightDelta):
		delta = rightDelta
	M = ChooseM(delta)

	rate = ApproximateMetropolisAcceptanceRate(a, b, M, delta)
	if (abs(rate - 0.5) < 0.2):
		return delta

	# randomly sample values of delta
	
	#for i in range(maxTries):  # TODO:  fix this... python can't take an object as an integer
	for i in range(10):
		trialDelta = np.random.uniform(leftDelta, rightDelta)
		M = ChooseM(delta)
		trialRate = ApproximateMetropolisAcceptanceRate(a, b, M, trialDelta)
		if(abs(trialRate - 0.5) < 0.2):
			return trialDelta
		if(abs(trialRate - 0.5) < abs(rate - 0.5)):
			delta = trialDelta
			rate = trialRate	
	'''
	eta = 0.1

	rate = ApproximateMetropolisAcceptanceRate(a, b, M, delta)
	tries = 0

	while( (rate > 0.65 or rate < 0.35) and tries < maxTries):
		# approximate numerical derivative of acceptance rate w.r.t. delta
		if(delta )
	'''
	
	return delta




def AdaptiveGridSearch(amax=5, bmax=5, M=1000, initialDelta=0.2):

	aa = np.linspace(0,amax,50)
	bb = np.linspace(0,bmax,50)

	Results = np.zeros((len(aa), len(bb)))

	delta = initialDelta

	for i in range(len(aa)):
		for j in range(len(bb)):
			a = aa[i]
			b = bb[j]
			newDelta = AdaptivelyChooseDelta(a, b, delta)
			delta = newDelta
			energy = UniformMetropolisSampler(a, b, M, delta)
			print("delta=" + str(delta) + ", energy=" + str(energy))
			Results[i][j] = energy


	print(Results)

	#compute minimum energy	
	minEnergy = np.amin(Results)
	
	# compute minimum energy parameters
	i, j = np.unravel_index(Results.argmin(), Results.shape)
	alphaMinimizer = aa[i]
	betaMinimizer = bb[j]
	print("Minimum Energy = " + str(minEnergy) + " = " + str(Results[i][j]))
	print("Armgin alpha, beta = " + str(alphaMinimizer) + ", " + str(betaMinimizer))
	

	# Plot results

	fig = plt.figure()
	ax = plt.axes(projection='3d')

	# define 2d planes of axes
	avals = np.outer(aa,np.ones(len(bb)))
	bvals = np.outer(np.ones(len(aa)), bb)

	ax.plot_surface(avals, bvals, Results, cmap='viridis', edgecolor='none')

	title = r"Approximate Wavefunction Energy (M=" + str(M) + ", $\delta$=Variable)"
	ax.set_title(title)
	ax.set_xlabel(r"$\alpha$")
	ax.set_ylabel(r"$\beta$")


	plt.show()

	now = datetime.now()
	#year = now.strftime("%Y")
	#month = now.strftime("%m")
	#day = now.strftime("%d")
	date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

	metastring = "Approx_Energy_M=" + str(M) + "_delta=Variable_a=0-" + str(amax) + "_b=0-" + str(bmax) + "_timestamp=" + date_time
	figname = "Graph_3D_" + metastring + ".png"
	#fig.savefig(figname, dpi=fig.dpi)
	fig.savefig(figname)

	# save data
	datastring = "Grid_" + metastring + ".csv"
	np.savetxt(datastring, Results, delimiter=',')



	return Results




if __name__ == "__main__":
	print("Setting up variational quantum monte carlo for Helium....")
	

	M = 250000
	delta = 0.2
	print("M = " + str(M) + ", delta = " + str(delta))

	# try grid search test
	#results = AdaptiveGridSearch(5.0, 5.0, M, delta)

	# do a local search to refine values of alpha, beta
	#results = FineGridSearch(1.6, 1.9, 20, 0.2, 0.5, 20, M, delta)

	# test MC convergence
	a = DiagnosticMetropolisSampler(0.1, 0.3, M, delta, 1000, 5)
