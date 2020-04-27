'''
This file contains functions for the variational wavefunction of helium, the local energy of that wavefunction, and the result of the helium Hamiltonian applied to the variational wavefunction.  It also contains many intermediate helper functions.
'''


import numpy as np
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt 
from datetime import datetime


# Euclidean norm of a 3-vector
def norm(v):
	x, y, z = v
	return np.sqrt(x*x + y*y + z*z)


# first factor of the trial wavefunction
def psi_1(a, r1, r2):
	return np.exp(-a*(r1 + r2))


# second factor of the trial wavefunction
def psi_2(b, r12):
	return np.exp(r12 / (2*(1+b*r12)))


# This is the total (unnormalized) wave function with variational parameters a=alpha, b=beta
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


# first derivative of psi_1 w.r.t. coord_i of electron e
def d_psi_1_variable(a, r_e, coord_ie, r_k):
	return -(a * coord_ie / r_e) * psi_1(a, r_e, r_k)


# first derivative of psi_1 w.r.t. coord_i of electron e
def d_psi_2_variable(b, coord_ie, coord_ik, r12):
	return psi_2(b, r12) * (1 / (2*(1+b*r12)**2) ) * (coord_ie - coord_ik)/r12


# second partial derivative of psi_trial (with respect to some variable)
def d2_psi_trial_variable(psi_1, d_psi_1, d2_psi_1, psi_2, d_psi_2, d2_psi_2):
	return d2_psi_1 * psi_2 + 2*d_psi_1*d_psi_2 + psi_1 * d2_psi_2


# Laplacian_e of psi_trial (that is, Laplacian for electron e coordinates)
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


# H | psi_trial(v1,v2; a,b) >
def HamiltonianPsi(a, b, v1, v2, r1, r2, r12):
	Laplacian1_psi = Laplacian_psi_trial_variable(a, b, v1, r1, v2, r2, r12)
	Laplacian2_psi = Laplacian_psi_trial_variable(a, b, v2, r2, v1, r1, r12)

	return (-1/2)*Laplacian1_psi + (-1/2)*Laplacian2_psi + (-2/r1 -2/r2 +1/r12)*psi_trial(a,b,v1,v2)


# Local energy of a psi_trial 
def LocalEnergy(a, b, v1, v2, r1, r2, r12):
	return HamiltonianPsi(a, b, v1, v2, r1, r2, r12) / psi_trial(a,b,v1,v2)



# proportional to magnitude of normalized wavefunction squared (only proportional because the normalization factor needed for psi_trial is unknown).  Used for Metropolis-Hastings algorithm.
def ProportionalToPi(a, b, v1, v2):
	return psi_trial(a,b,v1,v2)**2


'''
Input:
	MeanElocal:  sample mean local energy computed over N samples
	N:  number of samples used to estimate mean local energy
	Elocal:  N+1 sample of Elocal
Output:
	Mean local energy computed over N+1 samples
'''
def MeanLocalEnergy(MeanElocal, N, Elocal):
	return (N*MeanElocal + Elocal) / (N+1)


'''
Input:
	Elocal:  local energy of sample N+1
	MeanElocal:  calculated over the first N+1 samples
	VarianceElocal:  variance of the first N local energy samples
'''
def SampleVarianceLocalEnergy(Elocal, MeanElocal, VarianceElocal, N):
	return (N*VarianceElocal + (Elocal - MeanElocal)**2) / (N+1)