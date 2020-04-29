'''
This file holds Metropolis-Hastings routines (for the specific MH setup on R^6 that we need for helium)
'''
import numpy as np
import math
import matplotlib.pyplot as plt
from helium import LocalEnergy, ProportionalToPi, norm, SampleVarianceLocalEnergy

# return random 3-vector with entries uniformly sampled in delta*[-1,1)
def RandomVector(delta=0.2):

	# first get random vector uniformly sampled from [0,1), then rescale and shift
	return 2.0*np.random.rand(3) - 1



# This function probabilistically chooses to accept or not accept a proposed Monte Carlo move based on the Metropolis condition.  Boolean returned.
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

	# initialize approximation of local energy variance
	variance = 0

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

		approx += Ecurrent  # This must come before the variance update b/c of the convention adopted in the 'SampleVarianceLocalEnergy' function
		variance = SampleVarianceLocalEnergy(Ecurrent, (approx / (i+1)), variance, i+1)

		if(i%10000 == 0):
			print("Variance at i=" + str(i) + ":  " + str(variance))

	print("For a,b = " + str(a) + ", " + str(b) +": acceptance rate = " + str(numberOfAcceptedMoves / M))
	return approx / M 




'''
This function approximates the integral <psi(a,b)|H|psi(a,b)> by using the Metropolis-Hastings algorithm where the proposal probability (for moves in 2-electron position space) is uniform over the hypercube delta * [-1,1]^6.  M samples are taken for this approximation.

This function differs from 'UniformMetropolisSampler' in that this function is designed to help diagnose the mcmc convergence.  It also produces plots to help visualize the convergence.
'''
def DiagnosticMetropolisSampler(a, b, M, delta=0.2, measurementSpacing = 100, numChains=3):
	
	chainEstimateValues = []			# an array of arrays; each element array holds the estimated (total) energy after various numbers of monte carlo steps
	
	chainIntegrandValues = []			# an array of arrays; each element array holds the sampled values of integrand (local energy) after various numbers of monte carlo steps
	
	measurementIndices = []				# the steps at which measurements are taken (zero-indexed)
	index = measurementSpacing
	while(index <= M):
		measurementIndices.append(index)
		index += measurementSpacing
	
	for chain in range(numChains):

		# initialize energy approximation 
		#  (i.e. approximation of the integral <psi(a,b)|H|psi(a,b)> )
		#energies
		approx = 0
	
		measurementValues = []		# estimated energy
		#measurementIndices = []
		measurementLocal = []		# actual local energy measurements
	
		# initialize approximation of local energy variance
		variance = 0
	
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
	
			approx += Ecurrent  # This must come before the variance update b/c of the convention adopted in the 'SampleVarianceLocalEnergy' function
			variance = SampleVarianceLocalEnergy(Ecurrent, (approx / (i+1)), variance, i+1)
	
			if((i+1)%measurementSpacing == 0):
				measurementValues.append(approx / (i+1))
				#measurementIndices.append(i)
				measurementLocal.append(Ecurrent)
	
			if(i%10000 == 0):
				print("Variance at i=" + str(i) + ":  " + str(variance))
		
		chainEstimateValues.append(measurementValues)
		chainIntegrandValues.append(measurementLocal)
		#chainIndices.append(measurementIndices)


	# Plot estimated energy vs monte carlo steps

	fig = plt.figure(1)
	#ax = plt.axes(projection='3d')
	ax = plt.axes()

	for chain in range(numChains):
		measurementValues = chainEstimateValues[chain]
		#measurementIndices = chainIndices[chain]
		ax.plot(measurementIndices, measurementValues)

	title = r"Estimated Wavefunction Energy ($\alpha$=" + str(a) + r", $\beta$=" + str(b) + ")"
	ax.set_title(title)
	ax.set_xlabel("Monte Carlo Steps")
	ax.set_ylabel("Energy (Hartrees)")
	

	plt.show()

	now = datetime.now()
	date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

	figname = "./images/MC_EstimatedEnergyPlot_a=" + str(a) + "_b=" + str(b) + "_M=" + str(M) + "_delta=" + str(delta) + "_time=" + date_time + ".png"
	fig.savefig(figname, dpi=fig.dpi)
	#fig.savefig(figname)
	
	
	# Plot local energy (integrand) vs monte carlo steps
	
	fig2 = plt.figure(2)
	#ax = plt.axes(projection='3d')
	ax2 = plt.axes()
	

	for chain in range(numChains):
		measurementValues = chainIntegrandValues[chain]
		#measurementIndices = chainIndices[chain]
		ax2.plot(measurementIndices, measurementValues)
		#print("local energy measurements = " + str(len(measurementValues)))
		#print("numIndices = " + str(len(measurementIndices)))

	title = r"Local Energy ($\alpha$=" + str(a) + r", $\beta$=" + str(b) + ")"
	ax2.set_title(title)
	ax2.set_xlabel("Monte Carlo Steps")
	ax2.set_ylabel("Local Energy")
	

	plt.show()

	now = datetime.now()
	date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

	figname = "./images/MC_LocalEnergyPlot_a=" + str(a) + "_b=" + str(b) + "M=" + str(M) + "_delta=" + str(delta) + "_time=" + date_time + ".png"
	fig2.savefig(figname, dpi=fig.dpi)
	#fig.savefig(figname)
	

	print("For a,b = " + str(a) + ", " + str(b) +": acceptance rate = " + str(numberOfAcceptedMoves / M))
	
	
	return approx / M 





'''
This function approximates the integral <psi(a,b)|H|psi(a,b)> by using the Metropolis-Hastings algorithm where the proposal probability (for moves in 2-electron position space) is uniform over the hypercube delta * [-1,1]^6.  M samples are taken for this approximation.

Unlike the function 'DiagnosticMetropolisSampler', this function throws out a specified number of moves at the beginning to reduce initialization bias.  

TODO:  Compute estimates of exponential and integrated autocovariance times 
TODO:  Merge this function with 'DiagnosticMetropolisSampler'
'''
def BurnInDiagnosticMetropolisSampler(a, b, M, delta=0.2, measurementSpacing = 100, burnIn=20000, numChains=3):
	
	chainEstimateValues = []			# an array of arrays; each element array holds the estimated (total) energy after various numbers of monte carlo steps
	
	chainIntegrandValues = []			# an array of arrays; each element array holds the sampled values of integrand (local energy) after various numbers of monte carlo steps
	
	measurementIndices = []				# the steps at which measurements are taken (zero-indexed)
	'''
	index = measurementSpacing
	while(index <= M):
		measurementIndices.append(index)
		index += measurementSpacing
	'''

	for chain in range(numChains):

		# initialize energy approximation 
		#  (i.e. approximation of the integral <psi(a,b)|H|psi(a,b)> )
		#energies
		approx = 0
	
		measurementValues = []		# estimated energy
		#measurementIndices = []
		measurementLocal = []		# actual local energy measurements
	
		# initialize approximation of local energy variance
		variance = 0
	
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
	
			samplesAfterBurnIn = (i+1) - burnIn 	# Samples after burnin
			if(samplesAfterBurnIn > 0):
				approx += Ecurrent  # This must come before the variance update b/c of the convention adopted in the 'SampleVarianceLocalEnergy' function
				variance = SampleVarianceLocalEnergy(Ecurrent, (approx / samplesAfterBurnIn), variance, samplesAfterBurnIn)
	
				if(samplesAfterBurnIn%measurementSpacing == 0):
					measurementValues.append(approx / samplesAfterBurnIn)
					#measurementIndices.append(i)
					measurementLocal.append(Ecurrent)
					if(chain == 0):
						measurementIndices.append(i)
			
	
			if(i%10000 == 0):
				print("Variance at i=" + str(i) + ":  " + str(variance))
		
		chainEstimateValues.append(measurementValues)
		chainIntegrandValues.append(measurementLocal)
		#chainIndices.append(measurementIndices)


	# Plot estimated energy vs monte carlo steps

	fig = plt.figure(1)
	#ax = plt.axes(projection='3d')
	ax = plt.axes()

	for chain in range(numChains):
		measurementValues = chainEstimateValues[chain]
		#measurementIndices = chainIndices[chain]
		ax.plot(measurementIndices, measurementValues)

	title = r"Estimated Wavefunction Energy ($\alpha$=" + str(a) + r", $\beta$=" + str(b) + ")"
	ax.set_title(title)
	ax.set_xlabel("Monte Carlo Steps")
	ax.set_ylabel("Energy (Hartrees)")
	

	plt.show()

	now = datetime.now()
	date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

	figname = "./images/MC_BurnIn_EnergyPlot_a=" + str(a) + "_b=" + str(b) + "_M=" + str(M) + "_delta=" + str(delta) + "_time=" + date_time + ".png"
	fig.savefig(figname, dpi=fig.dpi)
	#fig.savefig(figname)
	
	
	# Plot local energy (integrand) vs monte carlo steps
	
	fig2 = plt.figure(2)
	#ax = plt.axes(projection='3d')
	ax2 = plt.axes()
	

	for chain in range(numChains):
		measurementValues = chainIntegrandValues[chain]
		#measurementIndices = chainIndices[chain]
		ax2.plot(measurementIndices, measurementValues)
		#print("local energy measurements = " + str(len(measurementValues)))
		#print("numIndices = " + str(len(measurementIndices)))

	title = r"Local Energy ($\alpha$=" + str(a) + r", $\beta$=" + str(b) + ")"
	ax2.set_title(title)
	ax2.set_xlabel("Monte Carlo Steps")
	ax2.set_ylabel("Local Energy")
	

	plt.show()

	now = datetime.now()
	date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

	figname = "./images/MC_BurnIn_LocalEPlot_a=" + str(a) + "_b=" + str(b) + "M=" + str(M) + "_delta=" + str(delta) + "_time=" + date_time + ".png"
	fig2.savefig(figname, dpi=fig.dpi)
	#fig.savefig(figname)
	

	print("For a,b = " + str(a) + ", " + str(b) +": acceptance rate = " + str(numberOfAcceptedMoves / M))
	
	
	return approx / M 


'''
This function approximates the metropolis algorithm acceptance rate for given variational parameters a,b and given delta > 0 which determines the proposal probability distribution.  This function is provided as a subroutine which samplers can build upon to create procedures that adaptively choose delta based on the values of a and b in order to achieve an acceptance rate in some desired range.
'''
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



if __name__ == '__main__':

	E = BurnInDiagnosticMetropolisSampler(5, 0.3, 250000, 0.2, 1000, 150000, 5)
	#E = BurnInDiagnosticMetropolisSampler(5, 0.3, 2500, 0.2, 100, 1000, 3)