import numpy as np
import math
#import matplotlib
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt 
from datetime import datetime

def norm(v):
	x, y, z = v
	return np.sqrt(x*x + y*y + z*z)

def psi_1(a, r1, r2):
	return np.exp(-a*(r1 + r2))

def psi_2(b, r12):
	return np.exp(r12 / (2*(1+b*r12)))

def psi_trial(a, b, v1, v2):
	r1 = norm(v1)
	r2 = norm(v2)
	r12 = norm(v1 - v2)
	return np.exp(-a*(r1 + r2))*np.exp(r12 / (2*(1+b*r12)))

'''
This yields the second derivative of psi_1 w.r.t. coordinate i of electron e

e = 1 or 2 = electron 1 or 2
i = 1,2, or 3 (for x,y, or z)
r_e = r_e
coord_ie = ith coordinate for electron e
rk = r_not e
'''
def d2_psi_1_variable(a, r_e, coord_ie, r_k):
	return (a / r_e)*(-1 + (coord_ie**2 / r_e)*(1/r_e + a))*psi_1(a, r_e, r_k)


def d2_psi_2_variable(b, coord_ie, coord_ik, r12):
	f1 = (coord_ie - coord_ik)**2 / (r12**2) * ( 1/(2 * (1 + b*r12)**2) - 2*b/(1 + b*r12) -1/r12 )
	return psi_2(b, r12) * (1 / (2*(1+b*r12)**2) ) * ( f1 + 1/r12) 

def d_psi_1_variable(a, r_e, coord_ie, r_k):
	return -(a * coord_ie / r_e) * psi_1(a, r_e, r_k)

def d_psi_2_variable(b, coord_ie, coord_ik, r12):
	return psi_2(b, r12) * (1 / (2*(1+b*r12)**2) ) * (coord_ie - coord_ik)/r12

def d2_psi_trial_variable(psi_1, d_psi_1, d2_psi_1, psi_2, d_psi_2, d2_psi_2):
	return d2_psi_1 * psi_2 + 2*d_psi_1*d_psi_2 + psi_1 * d2_psi_2

def Laplacian_psi_trial_variable(a, b, v_e, r_e, v_k, r_k, r12):
	psi1 = psi_1(a, r_e, r_k)
	psi2 = psi_2(b, r12)

	Laplacian = 0

	for i in range(3):
		d_psi_1_i = d_psi_1_variable(a, r_e, v_e[i], r_k)
		d_psi_2_i = d_psi_2_variable(b, v_e[i], v_k[i], r12)
		d2_psi_1_i = d2_psi_1_variable(a, r_e, v_e[i], r_k)
		d2_psi_2_i = d2_psi_2_variable(b, v_e[i], v_k[i], r12)

		d2_psi_trial_i = d2_psi_trial_variable(psi1, d_psi_1_i, d2_psi_1_i, psi2, d_psi_2_i, d2_psi_2_i)
		Laplacian += d2_psi_trial_i

	return Laplacian


def HamiltonianPsi(a, b, v1, v2, r1, r2, r12):
	Laplacian1_psi = Laplacian_psi_trial_variable(a, b, v1, r1, v2, r2, r12)
	Laplacian2_psi = Laplacian_psi_trial_variable(a, b, v2, r2, v1, r1, r12)

	return (-1/2)*Laplacian1_psi + (-1/2)*Laplacian2_psi + (-2/r1 -2/r2 +1/r12)*psi_trial(a,b,v1,v2)


def LocalEnergy(a, b, v1, v2, r1, r2, r12):
	return HamiltonianPsi(a, b, v1, v2, r1, r2, r12) / psi_trial(a,b,v1,v2)

def ProportionalToPi(a, b, v1, v2):
	return psi_trial(a,b,v1,v2)**2


# return random 3-vector with entries uniformly sampled in delta*[-1,1)
def RandomVector(delta=0.2):

	# first get random vector uniformly sampled from [0,1), then rescale and shift
	return 2.0*np.random.rand(3) - 1



# This function probabilistically chooses to accept or not accept a proposed Monte Carlo move
def MetropolisAcceptance(densityCurrent, densityProposed):
	if densityProposed >= densityCurrent:
		return True

	prob = densityProposed / densityCurrent
	if np.random.uniform() < prob:
		return True

	return False


'''
This function approximates the integral <psi(a,b)|H|psi(a,b)> by using the Metropolis-Hastings algorithm where the proposal probability (for moves in 2-electron position space) is uniform over the hypercube delta * [-1,1]^6.  M samples are taken for this approximation.
'''
def UniformMetropolisSampler(a, b, M, delta=0.2):

	# initialize energy approximation 
	#  (i.e. approximation of the integral <psi(a,b)|H|psi(a,b)> )
	approx = 0

	# randomly initialize in [-1,1]^6 (2-electron position space)
	v1 = RandomVector(1.0)
	v2 = RandomVector(1.0)

	r1 = norm(v1)
	r2 = norm(v2)
	r12 = norm(v1 - v2)

	Ecurrent = LocalEnergy(a, b, v1, v2, r1, r2, r12)
	PseudoDensityCurrent = ProportionalToPi(a, b, v1, v2)
	
	numberOfAcceptedMoves = 0

	for i in range(M):
		v1proposed = v1 + RandomVector(delta)
		v2proposed = v2 + RandomVector(delta)

		r1proposed = norm(v1proposed)
		r2proposed = norm(v2proposed)
		r12proposed = norm(v1proposed - v2proposed)

		Eproposed = LocalEnergy(a, b, v1proposed, v2proposed, r1proposed, r2proposed, r12proposed)
		PseudoDensityProposed = ProportionalToPi(a, b, v1proposed, v2proposed)

		if MetropolisAcceptance(PseudoDensityCurrent, PseudoDensityProposed):
			v1 = v1proposed
			v2 = v2proposed
			r1 = r1proposed
			r2 = r2proposed
			r12 = r12proposed
			Ecurrent = Eproposed
			PseudoDensityCurrent = PseudoDensityProposed

			numberOfAcceptedMoves += 1

		approx += Ecurrent

	print("For a,b = " + str(a) + ", " + str(b) +": acceptance rate = " + str(numberOfAcceptedMoves / M))
	return approx / M 


def ApproximateMetropolisAcceptanceRate(a, b, M, delta):

	#approx = 0
	#v1 = np.array([0.0,0.0,0.0])
	#v2 = np.array([0.0,0.0,0.0])

	# randomly initialize in [-1,1]^6
	v1 = RandomVector(1.0)
	v2 = RandomVector(1.0)

	r1 = norm(v1)
	r2 = norm(v2)
	r12 = norm(v1 - v2)

	Ecurrent = LocalEnergy(a, b, v1, v2, r1, r2, r12)
	PseudoDensityCurrent = ProportionalToPi(a, b, v1, v2)
	
	numberOfAcceptedMoves = 0

	for i in range(M):
		v1proposed = v1 + RandomVector(delta)
		v2proposed = v2 + RandomVector(delta)

		r1proposed = norm(v1proposed)
		r2proposed = norm(v2proposed)
		r12proposed = norm(v1proposed - v2proposed)

		Eproposed = LocalEnergy(a, b, v1proposed, v2proposed, r1proposed, r2proposed, r12proposed)
		PseudoDensityProposed = ProportionalToPi(a, b, v1proposed, v2proposed)

		if MetropolisAcceptance(PseudoDensityCurrent, PseudoDensityProposed):
			v1 = v1proposed
			v2 = v2proposed
			r1 = r1proposed
			r2 = r2proposed
			r12 = r12proposed
			Ecurrent = Eproposed
			PseudoDensityCurrent = PseudoDensityProposed

			numberOfAcceptedMoves += 1

		#approx += Ecurrent

	#print("For a,b = " + str(a) + ", " + str(b) +": acceptance rate = " + str(numberOfAcceptedMoves / M))
	return numberOfAcceptedMoves / M

def GridSearch(amax=10, bmax=10, M=1000, delta=0.2):

	aa = np.linspace(0,amax,50)
	bb = np.linspace(0,bmax,50)

	Results = np.zeros((len(aa), len(bb)))

	for i in range(len(aa)):
		for j in range(len(bb)):
			a = aa[i]
			b = bb[j]
			energy = UniformMetropolisSampler(a, b, M, delta)
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

	metastring = "Approx_Energy_M=" + str(M) + "_delta=" + str(delta) + "_a=0-" + str(amax) + "_b=0-" + str(bmax) + "_timestamp=" + date_time
	figname = "Graph_3D_" + metastring + ".png"
	fig.savefig(figname, dpi=fig.dpi)

	# save data
	datastring = "Grid_" + metastring + ".csv"
	np.savetxt(datastring, Results, delimiter=',')



	return Results

	# Miobium.createSchema("a:number, b:number, M:natural, delta: number")
	# Miobium.save(a,b,M,delta)



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

	metastring = "LocalSearch_Approx_Energy_M=" + str(M) + "_delta=" + str(delta) +  "_timestamp=" + date_time
	figname = "Graph_3D_" + metastring + ".png"
	fig.savefig(figname, dpi=fig.dpi)

	# save data
	datastring = "Grid_" + metastring + ".csv"
	np.savetxt(datastring, Results, delimiter=',')


	return Results



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
	fig.savefig(figname, dpi=fig.dpi)

	# save data
	datastring = "Grid_" + metastring + ".csv"
	np.savetxt(datastring, Results, delimiter=',')



	return Results




if __name__ == "__main__":
	print("Setting up variational quantum monte carlo for Helium....")
	

	M = 200000
	delta = 0.2
	print("M = " + str(M) + ", delta = " + str(delta))

	# try grid search test
	#results = AdaptiveGridSearch(5.0, 5.0, M, delta)

	# do a local search to refine values of alpha, beta
	results = FineGridSearch(1.6, 1.9, 25, 0.2, 0.5, 20, M, delta)
