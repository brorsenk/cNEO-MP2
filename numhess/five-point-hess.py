

#--------------------------------------------------------------
# PythonScript for Building Hessian Input Files
#--------------------------------------------------------------
# Written: July 2023
# Updated: July 2025


#-------------------------------
# Importations
#-------------------------------

import string
import time
import copy

#----------------------------------------
# For troubleshooting
#----------------------------------------
import os
import sys
import inspect
import importlib
print(sys.version)

#----------------------------------------
# Math dependencies
#----------------------------------------
import math
import numpy
import sympy
import scipy

#----------------------------------------
# Chemistry dependencies
#----------------------------------------
import pyscf
import pyscf.ao2mo as ao2mo

from pyscf.neo import cneomp2_diis
from pyscf.neo.cneomp2_diis import *
from pyscf import ao2mo, mp, dft, data, scf, neo, lib, gto
from pyscf.neo.ao2mo import *
from pyscf.neo import cdft
from pyscf.dft import numint

import cymods
from cymods.t_amps_e_only.t_amps_e_only import t_amps_e_only
from cymods.t_amps_en_only.t_amps_en_only import t_amps_en_only
from cymods.t_amps_n_only.t_amps_n_only import t_amps_n_only
from cymods.mp2_dens.mp2_density import mp2_density_one
from cymods.hylleraas_e.hylleraas_e import Hylleraas_energy_e
from cymods.hylleraas_en.hylleraas_en import Hylleraas_energy_en
from cymods.hylleraas_n.hylleraas_n import Hylleraas_energy_n
from cymods.rmsd.rmsd import RMSD_e
from cymods.rmsd.rmsd import RMSD_en
from cymods.rmsd.rmsd import RMSD_n



#-------------------------------
# Sub-Importations
#-------------------------------

from pyscf import neo
from pyscf.neo import cneomp2 

#import importlib
mod = importlib.import_module("methods.multi-component.cneomp2.constrained-t.fix-speed.conv-adpt.zundel-one-h-constraint-correlated-atz")
#from methods.multi-component.cneomp2.constrained-t.run-hessian import zundel_one_h

#-------------------------------
# Define functions
#-------------------------------

def energy(coordinates):

    # Defining variables:
    dimensions = 3

    # Building the mole object:

    # [Note:] First atom input into the mole object for a triatomic molecule must be the central atom because of the way that
    #         the distances are being computed.

    #mol = gto.Mole()
    mol = neo.Mole()
    mol.build(atom=[['H',coordinates[0:dimensions]],
                    ['O',coordinates[dimensions:2*dimensions]],
                    ['O',coordinates[2*dimensions:3*dimensions]],
                    ['H',coordinates[3*dimensions:4*dimensions]],
                    ['H',coordinates[4*dimensions:5*dimensions]],
                    ['H',coordinates[5*dimensions:6*dimensions]],
                    ['H',coordinates[6*dimensions:7*dimensions]]], basis='aug-cc-pvtz', charge=1, unit='Bohr', quantum_nuc = [0])


    # For Troubleshooting:
    # Printing Cartesian Coordinates:
    #        print(coordinates)
    #        print(mol.atom_coords(unit='ANG'))
    #        print('The variable equals:', mol.nuc_num)

    # Calculating the Bond Length to Graph later:
    # These lines can be commented out when running the PESs, but are necessary for the geometry optimization data.
    R = mol.atom_coords(unit='ANG')
    RB = mol.atom_coords(unit='BOHR') 
    #distance1 = math.sqrt(((R[1,0] - R[0,0])**2)+((R[1,1] - R[0,1])**2)+((R[1,2] - R[0,2])**2))
    #distance2 = math.sqrt(((R[2,0] - R[0,0])**2)+((R[2,1] - R[0,1])**2)+((R[2,2] - R[0,2])**2))
    #distance3
    #distance4
    #distance5

    # Defining Objects to pass into the Class:
    reg = neo.HF(mol)
    con = neo.cdft.CDFT(mol)
   
    # For single component:
    #reg = scf.HF(mol)
    #con = scf.HF(mol)

    reg.verbose = 10
    con.verbose = 10
    reg.verbose = False
    con.verbose = False
    # Calculating energies without MP2 correction:
    energy_con = con.scf()
    #print('\n\n\n\n\n\n\n\n\n\nCONSTRAINED CALCULATION CONVERGED')


    # Making an object/instance of the Class and calculating MP2 correction terms:
    mp2 = mod.cNEOMP2(reg, con, energy_con, restart=False)
    #mp2 = cneomp2.cNEOMP2(reg, con, energy_con)
    final_e, mp2_iter_e, mp2_iter_en, mp2_iter_n, lagrange = mp2.kernel()


    # Definition of the Total Energy:
    totale = energy_con+mp2_iter_e+mp2_iter_en+mp2_iter_n


    print('outputline', con.converged, totale)
    print('ANG COORDS')
    print(R)
    print('BOHR COORDS')
    print(RB)
    # Return Total Energy:
    return totale



#-------------------------------
# Define variables
#-------------------------------
natoms = 7 
dimensions = 3

term1_coords = numpy.zeros((natoms, dimensions))
term2_coords = numpy.zeros((natoms, dimensions))
term3_coords = numpy.zeros((natoms, dimensions))
term4_coords = numpy.zeros((natoms, dimensions))
term5_coords = numpy.zeros((natoms, dimensions))
term6_coords = numpy.zeros((natoms, dimensions))
term7_coords = numpy.zeros((natoms, dimensions))
term8_coords = numpy.zeros((natoms, dimensions))

set_i = numpy.zeros((natoms, dimensions))
set_i_minus = numpy.zeros((natoms, dimensions))
set_i_plus = numpy.zeros((natoms, dimensions))
set_i_two_minus = numpy.zeros((natoms, dimensions))
set_i_two_plus = numpy.zeros((natoms, dimensions))

total_vars = 21
step_size = 0.005*numpy.ones((natoms, dimensions)) # Angstroms
step_size_scalar = 0.005
start_coords = numpy.loadtxt('zundel-augtz-unconstrained-correlated-bohr.xyz')

#----------------------------------------------
# Fill coordinate set (i) 
#----------------------------------------------

set_i = start_coords

#----------------------------------------------
# Fill coordinate set (i-1) 
#----------------------------------------------

set_i_minus = set_i - step_size

#----------------------------------------------
# Fill coordinate set (i+1) 
#----------------------------------------------

set_i_plus = set_i + step_size

#----------------------------------------------
# Fill coordinate set (i-2h)
#----------------------------------------------

set_i_two_minus = set_i - 2*step_size

#----------------------------------------------
# Fill coordinate set (i+2h)
#----------------------------------------------

set_i_two_plus = set_i + 2*step_size

term1_coords = term1_coords.flatten()
term2_coords = term2_coords.flatten()
term3_coords = term3_coords.flatten()
term4_coords = term4_coords.flatten()
term5_coords = term5_coords.flatten()
term6_coords = term6_coords.flatten()
term7_coords = term7_coords.flatten()
term8_coords = term8_coords.flatten()

set_i = set_i.flatten()
set_i_minus = set_i_minus.flatten()
set_i_plus = set_i_plus.flatten()
set_i_two_minus = set_i_two_minus.flatten()
set_i_two_plus = set_i_two_plus.flatten()

Hessian = numpy.zeros((total_vars, total_vars))

term1 = 0
term2 = 0
term3 = 0
term4 = 0
term5 = 0
term6 = 0
term7 = 0
term8 = 0

for i in range(total_vars):

    term1 = 0
    term2 = 0
    term3 = 0
    term4 = 0

    term1_coords = numpy.zeros((natoms, dimensions))
    term2_coords = numpy.zeros((natoms, dimensions))
    term3_coords = numpy.zeros((natoms, dimensions))
    term4_coords = numpy.zeros((natoms, dimensions))
    term5_coords = numpy.zeros((natoms, dimensions))

    term1_coords = term1_coords.flatten()
    term2_coords = term2_coords.flatten()
    term3_coords = term3_coords.flatten()
    term4_coords = term4_coords.flatten()
    term5_coords = term5_coords.flatten()

    # For term1
    term1_coords = copy.copy(set_i)
    term1_coords[i] = set_i_two_plus[i]
    term1 = energy(term1_coords)

    # For term2
    term2_coords = copy.copy(set_i)
    term2_coords[i] = set_i_plus[i]
    term2 = energy(term2_coords)

    # For term3
    term3_coords = copy.copy(set_i)
    term3 = energy(term3_coords)

    # For term4
    term4_coords = copy.copy(set_i)
    term4_coords[i] = set_i_minus[i]
    term4 = energy(term4_coords)

    # For term5
    term5_coords = copy.copy(set_i)
    term5_coords[i] = set_i_two_minus[i]
    term5 = energy(term5_coords)

    Hessian[i,i] = (-term1 + 16*term2 - 30*term3 + 16*term4 - term5)/(12*(step_size_scalar**2))
    print(i,i,Hessian[i,i])


    for j in range(total_vars):

        if (i==j):
            pass

        else:

            term1 = 0
            term2 = 0
            term3 = 0
            term4 = 0
            term5 = 0
            term6 = 0
            term7 = 0
            term8 = 0

            term1_coords = numpy.zeros((natoms, dimensions))
            term2_coords = numpy.zeros((natoms, dimensions))
            term3_coords = numpy.zeros((natoms, dimensions))
            term4_coords = numpy.zeros((natoms, dimensions))
            term5_coords = numpy.zeros((natoms, dimensions))
            term6_coords = numpy.zeros((natoms, dimensions))
            term7_coords = numpy.zeros((natoms, dimensions))
            term8_coords = numpy.zeros((natoms, dimensions))

            term1_coords = term1_coords.flatten()
            term2_coords = term2_coords.flatten()
            term3_coords = term3_coords.flatten()
            term4_coords = term4_coords.flatten()
            term5_coords = term5_coords.flatten()
            term6_coords = term6_coords.flatten()
            term7_coords = term7_coords.flatten()
            term8_coords = term8_coords.flatten()


            # For term1
            term1_coords = copy.copy(set_i)
            term1_coords[i] = set_i_two_plus[i]
            term1_coords[j] = set_i_two_plus[j]

            term1 = energy(term1_coords)

            # For term2
            term2_coords = copy.copy(set_i)
            term2_coords[i] = set_i_plus[i]
            term2_coords[j] = set_i_plus[j]

            term2 = energy(term2_coords)

            # For term3
            term3_coords = copy.copy(set_i)
            term3_coords[i] = set_i_two_plus[i]
            term3_coords[j] = set_i_two_minus[j]

            term3 = energy(term3_coords)

            # For term4
            term4_coords = copy.copy(set_i)
            term4_coords[i] = set_i_plus[i]
            term4_coords[j] = set_i_minus[j]

            term4 = energy(term4_coords)

            # For term5
            term5_coords = copy.copy(set_i)
            term5_coords[i] = set_i_two_minus[i]
            term5_coords[j] = set_i_two_plus[j]

            term5 = energy(term5_coords)

            # For term6
            term6_coords = copy.copy(set_i)
            term6_coords[i] = set_i_minus[i]
            term6_coords[j] = set_i_plus[j]

            term6 = energy(term6_coords)

            # For term7
            term7_coords = copy.copy(set_i)
            term7_coords[i] = set_i_two_minus[i]
            term7_coords[j] = set_i_two_minus[j]

            term7 = energy(term7_coords)

            # For term8
            term8_coords = copy.copy(set_i)
            term8_coords[i] = set_i_minus[i]
            term8_coords[j] = set_i_minus[j]

            term8 = energy(term8_coords)


            Hessian[i,j] = (-term1 + 16*term2 + term3 - 16*term4 + term5 - 16*term6 - term7 + 16*term8)/(48*step_size_scalar*step_size_scalar)
            print(i,j,Hessian[i,j])


print('**********************HURRAY*************************')
print('The Hessian calculation is succesfully complete!')
print('*****************************************************')


print(Hessian)              


#eigvec, eigval = sympy.diagonalize(Hessian)

#sympy.pprint(eigval)



    



