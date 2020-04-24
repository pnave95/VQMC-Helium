import numpy as np
import sys
import re

#print(sys.argv)

csvfile = sys.argv[1]  # The name of this script is the first (zeroth) element of the list

datagrid = np.genfromtxt(csvfile, delimiter=',')


minEnergy = np.nanmin(datagrid)
i, j = np.unravel_index(np.nanargmin(datagrid), datagrid.shape)

print("Min energy = " + str(minEnergy))

#need to parse csv file name to get the alpha, beta values
numAs, numBs = datagrid.shape

print(numAs)

amaxRegex = '(?<=(a\=0-)).'
result = re.search(amaxRegex, csvfile)
amax = float(result.group(0))
#print(amax)

bmaxRegex = '(?<=(b\=0-)).'
result2 = re.search(bmaxRegex, csvfile)
bmax = float(result2.group(0))

aa = np.linspace(0, amax, numAs)
bb = np.linspace(0, bmax, numBs)
alpha = aa[i]
beta = bb[j]

print("Minimized at a,b = " + str(alpha) + ", " + str(beta))


