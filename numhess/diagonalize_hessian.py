

#--------------------------------------------------------------
# PythonScript for Building Hessian Input Files
#--------------------------------------------------------------
# Written: July 2023
# Updated: July 2025


#-------------------------------
# Importations
#-------------------------------

import sys
import os
import numpy
import scipy
import sympy
import math
import pyscf
import string
import time
import math
import copy

numpy.set_printoptions(threshold=sys.maxsize, precision=3, linewidth=300)

#-------------------------------
# Sub-Importations
#-------------------------------

from pyscf import neo

#-------------------------------
# Define functions
#-------------------------------


natoms = 7
ndims = 3

atom1_mass = 1.007825
atom2_mass = 15.994915
atom3_mass = 15.994915
atom4_mass = 1.007825
atom5_mass = 1.007825
atom6_mass = 1.007825
atom7_mass = 1.007825

#natoms=3
#atom1_mass=15.994915
#atom2_mass=1.007825
#atom3_mass=1.007825


#natoms = 4
#ndims = 3

#atom1_mass = 1.007825
#atom2_mass = 18.99840316273
#atom3_mass = 1.007825
#atom4_mass = 18.99840316273



atom_mass_list = [atom1_mass, atom2_mass, atom3_mass, atom4_mass, atom5_mass, atom6_mass, atom7_mass]

#atom_mass_list = [atom1_mass, atom2_mass, atom3_mass]

#atom_mass_list = [atom1_mass, atom2_mass, atom3_mass, atom4_mass]

total_vars = natoms*ndims
#Hessian = numpy.ones((total_vars,total_vars))
#Hessian = numpy.loadtxt("zundel_atz_numerical_hessian_5_5_2025.npy", dtype=float)
#Hessian = numpy.loadtxt("zundel_atz_scnumhess_15_5_2025.npy", dtype=float)
Hessian = numpy.loadtxt("zundel-cneomp2-atz-uc-on-uc-geom-bohr.txt", dtype=float)

mass_mat = numpy.identity(natoms*ndims)
mass_weight_hessian=numpy.zeros((total_vars, total_vars))
print(mass_mat.shape)


for i in range(total_vars):
    for j in range(total_vars):

        
        atom1 = math.ceil((i+1)/ndims)
        atom2 = math.ceil((j+1)/ndims)
        mass_mat[i][j] = (1/math.sqrt(atom_mass_list[atom1-1]*atom_mass_list[atom2-1]))
        print(i, j, atom1, atom2)
        

        
#for i in range(total_vars):
#    for j in range(total_vars):
       # if (i==j):
       #
       #     for a in range(natoms):
       #     
       #         if i>((a*ndims)-1) and i<((a+1)*ndims):
       #             mass_mat[i][j] = (1/math.sqrt(atom_mass_list[a]))
       #         else: 
       #             pass

# mass_weight_hessian = mass_mat @ Hessian @ mass_mat



# The below commented out lines only worked for diatomic systems...
#       if (i<ndims) and (j<ndims):
#           mass_weight_hessian[i,j] = Hessian[i,j]*(1/(math.sqrt(atom1_mass*atom1_mass))) 
#           print(i,j,'block HH')
#       elif (i>(ndims-1)) and (j<ndims):
#           mass_weight_hessian[i,j] = Hessian[i,j]*(1/(math.sqrt(atom2_mass*atom1_mass)))
#           print(i,j,'block FH')
#       elif (i<ndims) and (j>(ndims-1)):
#           mass_weight_hessian[i,j] = Hessian[i,j]*(1/(math.sqrt(atom1_mass*atom2_mass)))
#           print(i,j,'block HF')
#       elif (i>(ndims-1)) and (j>(ndims-1)):
#           mass_weight_hessian[i,j] = Hessian[i,j]*(1/(math.sqrt(atom2_mass*atom2_mass)))
#           print(i,j, 'block FF')
 
#print(sys.version)     
for i in range(total_vars):
    for j in range(total_vars):

        mass_weight_hessian[i][j] = mass_mat[i][j]*Hessian[i][j] 
#print(Hessian)              
print(mass_weight_hessian)


eigval, eigvec = numpy.linalg.eig(mass_weight_hessian)
#eigval2, eigvec2 = numpy.linalg.eig(Test_Hessian)
print('***')
print(eigvec)
print('***')

diagonal_Hessian = eigvec.T @ mass_weight_hessian @ eigvec
print('diagonal Hessian: ')
print(diagonal_Hessian)
#eigl, eigc = numpy.linalg.eig(Hessian)

print('eigenvalues and frequencies at 0K: ')
print(eigval)

#eigval = 3.571067673 * eigval # Convert Hartree/Ang^2 to Hartree/Bohr^2

for i in range(len(eigval)):
    #print(eigval[i])
    if eigval[i]>(0.0000000001):
        print(eigval[i], '->', (((math.sqrt(eigval[i]*(5.4858*(10**(-4))))/(2*math.pi))/(137))*(1/(5.2917724900001*(10**(-9))))))
    else:
        print(eigval[i])

#print(eigl)

# Convert to wavenumbers in cm^-1
#eigval_wavenumbers = (((math.sqrt(eigval * (5.4858*(10**(-4))))/(2*math.pi))/(137))*(1/(5.2917724900001*(10**(-9)))))

#print('Vibrational Frequencies')
#print(eigval_wavenumbers)



