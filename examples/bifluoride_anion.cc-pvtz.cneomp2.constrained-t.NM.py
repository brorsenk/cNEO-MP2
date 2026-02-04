#-------------------------------------------------------------------------------
# cNEOMP2 Module
#-------------------------------------------------------------------------------
# INFO
# INFO
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Importations
#-------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------------
# For troubleshooting
#----------------------------------------
import os
import sys
import inspect
from timeit import default_timer as timer

print(sys.version)
#sys.stdout = open('bifluoride_anion.cc-pvtz.cneomp2.constrained-t.NM.log', 'wt')


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
#from cymods.density.density_comp import one_dim_density_on_axis
#from cymods.density.density_comp import one_dim_density_off_axis
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#-------------------------------------------------------------------------------
# Options
#-------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------------
# Numpy Options
#----------------------------------------
numpy.set_printoptions(suppress=True, precision = 8, linewidth=sys.maxsize,threshold=sys.maxsize)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#-------------------------------------------------------------------------------
# Defined Global Variables
#-------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
line = ('---------------------------------------------')
asterisk = ('*********************************************')
inf = float('inf')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#-------------------------------------------------------------------------------
# Defined Global Functions
#-------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If any are used place these here.


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#-------------------------------------------------------------------------------
# Table of Contents for the cNEO-MP2 Class
#-------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# [1] Initialization and instantiation of class attributes
# [2] Kernel (class method)
#    *[2.1] Assignment of variables for class attributes 
#    *[2.2] Definition of variables for MP2 calculations 
#    *[2.3] Construction of Fock matrices 
#    *[2.4] Lagrangian constraint upon the MP2 nuclear densities
#    *[2.5] Electronic t-amplitude function 
#    *[2.6] Electronic-nuclear t-amplitude function
#    *[2.7] Nuclear t-amplitude function
#    *[2.8] Nuclear single-particle MP2 density matrix function 
#    *[2.9] Hylleraag electronic energy function
#    *[2.10] Hylleraas electronic-nuclear energy function
#    *[2.11] Hylleraas nuclear energy function
#    *[2.12] SCF optimization procedure
#    *[2.13] RMSD calculations
#    *[2.14] Check all convergence criteria
#    *[2.15] Calculate and compare HF density with final optimized MP2 density
#    *[2.16] Return variables and complete kernel function processes
# [3]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#-------------------------------------------------------------------------------
# The cNEO-MP2 Class
#-------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class cNEOMP2(lib.StreamObject):


#-------------------------------------------------------------------------------
# [1] Initialization and instantiation of class attributes
#-------------------------------------------------------------------------------
# Within this function the attributes of the class are initialized/instantiated.
# The new attributes for the cNEOMP2 class are the t-amplitudes and the lagrange multipliers for the new constrained optimization.
# Other attributes are previously existing for the NEO-HF and CNEO-DFT classes, but are simply redefined here.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, reg, con, base_energy, restart):
 
        start_timer_1 = timer()

        self.mol = neo.Mole()

        # The reg object is the regular NEO-HF class treated mole object.
        # The con object is the NEO-c-DFT(HF) class treated mole object. 
  
        self.reg = reg
        self.con = con

        # The base_energy is the cHF energy used as input to the cNEOMP2 class.
        self.base_energy = base_energy

        self.mp2_density_converged = [False]*len(self.con.mol.nuc)

        # The electronic and nuclear density matrix attributes
        # and mo coefficients of the reg mole object are defined 
        # here to be the same as those of the con mole object.    
        
        # redefinition of electronic class attributes        
        self.reg.dm_elec = self.con.dm_elec
        self.reg.mf_elec.mo_coeff = self.con.mf_elec.mo_coeff


        self.posn_ints = []

        # redefinition of nuclear class attributes
        for i in range(len(self.reg.mol.nuc)):
            self.reg.dm_nuc[i] = self.con.dm_nuc[i]
            self.reg.mf_nuc[i].mo_coeff = self.con.mf_nuc[i].mo_coeff

        self.R_mp2 = self.con.mol.atom_coords(unit='ANG')

        self.posn_ints = [None] * con.mol.nuc_num
        for i in range(len(self.con.mol.nuc)):
            hk = self.con.mf_nuc[i]
            s1n = hk.get_ovlp(hk.mol)

            self.posn_ints[i] = hk.mol.intor_symmetric('int1e_r', comp=3)
#            for x in range(3):
#                self.posn_ints[i][x,:,:] = hk.mo_coeff.T @ self.posn_ints[i][x,:,:] @ hk.mo_coeff
#            print((hk.nuclei_expect_position)) 
 
            self.posn_ints[i] = (self.posn_ints[i] \
                         - numpy.array([hk.nuclei_expect_position[i] * s1n for i in range(3)])) 
#                         - numpy.array([self.R_mp2[i][x] * s1n for x in range(3)])) #numpy.array([hk.nuclei_expect_position[i] * s1n for i in range(3)]))


        # Lagrange multipliers for the constrained optimization of the nuclear and electronic-nuclear MP2 t-amplitudes
        self.lagr = numpy.zeros((len(self.con.mol.nuc), 3))
        self.try_lagr = numpy.zeros((len(self.con.mol.nuc), 3))


        # new electronic class attributes
        self.e_nocc = con.mf_elec.mo_coeff[:,con.mf_elec.mo_occ>0].shape[1]
        self.e_tot  = con.mf_elec.mo_coeff[0,:].shape[0]
        self.e_nvir = self.e_tot - self.e_nocc

        self.t_elec = numpy.zeros((self.e_nocc, self.e_nvir, self.e_nocc, self.e_nvir))
        self.l_elec = numpy.zeros((self.e_nvir, self.e_nocc, self.e_nvir, self.e_nocc))
             
        # new nuclear class attributes
        self.num_ovt = []

        # A list of arrays is built to store the information on number of occupied, virtual, and total orbitals for each nucleus.
        for i in range(len(self.con.mol.nuc)):

             nuc_occ = self.con.mf_nuc[i].mo_coeff[:,self.con.mf_nuc[i].mo_occ>0].shape[1]
             nuc_tot = self.con.mf_nuc[i].mo_coeff[0,:].shape[0]
             nuc_vir = nuc_tot - nuc_occ

             ovt_array = numpy.zeros((1,3))

             ovt_array[0,0] = nuc_occ
             ovt_array[0,1] = nuc_vir
             ovt_array[0,2] = nuc_tot
          
             self.num_ovt.append(ovt_array)             


        self.t_nuc = numpy.empty((len(self.con.mol.nuc),len(self.con.mol.nuc)), dtype=object)
        self.t_opt_n = numpy.empty((len(self.con.mol.nuc),len(self.con.mol.nuc)), dtype=object)
        self.t_nuc_test= numpy.empty((len(self.con.mol.nuc),len(self.con.mol.nuc)), dtype=object)
        self.l_nuc = numpy.empty((len(self.con.mol.nuc),len(self.con.mol.nuc)), dtype=object)
        self.l_opt_n = numpy.empty((len(self.con.mol.nuc),len(self.con.mol.nuc)), dtype=object)
        self.l_nuc_test = numpy.empty((len(self.con.mol.nuc),len(self.con.mol.nuc)), dtype=object)

        for i in range(len(self.con.mol.nuc)):
            for j in range(len(self.con.mol.nuc)):

                self.t_nuc[i][j] = numpy.zeros((int(self.num_ovt[i][0,0]), int(self.num_ovt[i][0,1]), int(self.num_ovt[j][0,0]), int(self.num_ovt[j][0,1])))
                self.t_opt_n[i][j] = numpy.zeros((int(self.num_ovt[i][0,0]), int(self.num_ovt[i][0,1]), int(self.num_ovt[j][0,0]), int(self.num_ovt[j][0,1])))
                self.t_nuc_test[i][j] = numpy.zeros((int(self.num_ovt[i][0,0]), int(self.num_ovt[i][0,1]), int(self.num_ovt[j][0,0]), int(self.num_ovt[j][0,1])))
                self.l_nuc[i][j] = numpy.zeros((int(self.num_ovt[i][0,1]), int(self.num_ovt[i][0,0]), int(self.num_ovt[j][0,1]), int(self.num_ovt[j][0,0])))
                self.l_opt_n[i][j] = numpy.zeros((int(self.num_ovt[i][0,1]), int(self.num_ovt[i][0,0]), int(self.num_ovt[j][0,1]), int(self.num_ovt[j][0,0])))
                self.l_nuc_test[i][j] = numpy.zeros((int(self.num_ovt[i][0,1]), int(self.num_ovt[i][0,0]), int(self.num_ovt[j][0,1]), int(self.num_ovt[j][0,0])))

        # new electronic-nuclear class attributes
        self.t_elecnuc = []
        self.t_opt_en = []
        self.t_elecnuc_test = []
        self.l_elecnuc = []
        self.l_opt_en = []
        self.l_elecnuc_test = []

        for i in range(len(self.con.mol.nuc)):

            t_element = numpy.zeros((self.e_nocc, self.e_nvir, int(self.num_ovt[i][0,0]), int(self.num_ovt[i][0,1])))
            l_element = numpy.zeros((self.e_nvir, self.e_nocc, int(self.num_ovt[i][0,1]), int(self.num_ovt[i][0,0])))
            self.t_elecnuc.append(t_element)
            self.t_opt_en.append(t_element)
            self.t_elecnuc_test.append(t_element)
            self.l_elecnuc.append(l_element) 
            self.l_opt_en.append(l_element)
            self.l_elecnuc_test.append(l_element)

        #If loading in t-amplitudes from a previous run that didn't finish or converge, change the value of restart below to True.
        restart=True 

        if (restart == True):
            self.t_elec = numpy.load('bifluoride_anion.cc-pvtz.cneomp2.t_elec.npy'+'.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII', max_header_size=10000)
            self.l_elec = numpy.load('bifluoride_anion.cc-pvtz.cneomp2.l_elec.npy'+'.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII', max_header_size=10000)

            for i in range(len(self.con.mol.nuc)):
                 self.t_elecnuc[i] = numpy.load('bifluoride_anion.cc-pvtz.cneomp2.t_elecnuc.npy'+'.'+str(i)+'.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII', max_header_size=10000)
                 self.l_elecnuc[i] = numpy.load('bifluoride_anion.cc-pvtz.cneomp2.l_elecnuc.npy'+'.'+str(i)+'.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII', max_header_size=10000)


            for i in range(len(self.con.mol.nuc)):
                for j in range(len(self.con.mol.nuc)):
                    self.t_nuc[i][j] = numpy.load('bifluoride_anion.cc-pvtz.cneomp2.t_nuc.npy'+'.'+str(i)+str(j)+'.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII', max_header_size=10000)
                    self.l_nuc[i][j] = numpy.load('bifluoride_anion.cc-pvtz.cneomp2.l_nuc.npy'+'.'+str(i)+str(j)+'.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII', max_header_size=10000)
        else:
            pass

        finish_timer_1 = timer()

        print('timer[1]: timing of the init function: ', finish_timer_1 - start_timer_1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
#-------------------------------------------------------------------------------
# [2] Kernel (class method)
#-------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def kernel(self):

        start_timer_2 = timer()

        #-------------------------------------------------------------------------------
        # [2.1] Assignment of variables for class attributes
        #-------------------------------------------------------------------------------
        # Here at the beginning of the kernel function in which the construction of the MP2 densitites, t-amplitudes, and energies 
        # is carried out, the class attributes are assigned to variables.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #---------------------------
        # For NEO Mole Objects
        #---------------------------
        mol = self.mol
 
        reg = self.reg
        con = self.con

        reg.dm_elec = self.con.dm_elec
        reg.mf_elec.mo_coeff = self.con.mf_elec.mo_coeff

        for i in range(len(self.reg.mol.nuc)):
            reg.dm_nuc[i] = self.con.dm_nuc[i]
            reg.mf_nuc[i].mo_coeff = self.con.mf_nuc[i].mo_coeff

        e_nocc = self.e_nocc
        e_nvir = self.e_nvir
        e_tot = self.e_tot

        n_ovt = self.num_ovt

        #--------------------------------------------------------
        # Lagrange multipliers for the constrained optimization.
        #--------------------------------------------------------

        #self.try_lagr = self.x0 ####DELETE THIS AFTER TROUBLESHOOTING!
        lagr = self.lagr
    

        #--------------------------------------------------------
        # t-amplitudes for the constrained optimization.
        #--------------------------------------------------------
        t_nuc = self.t_nuc
        t_elec = self.t_elec
        t_elecnuc = self.t_elecnuc
        

        #----------------------------------------------------------
        # CHF energy (base energy before MP2 corrections are added)
        #----------------------------------------------------------
        base_energy = self.base_energy
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #-------------------------------------------------------------------------------        
        # [2.2] Definition of variables for MP2 calculations 
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #--------------------------------------------------------
        # Variables to return at end of kernel function
        #--------------------------------------------------------
        total_n = 0
        total_e = 0
        total_en = 0

        #--------------------------------------------------------
        # Convergence criteria
        #--------------------------------------------------------
        t_conv_tol = 10**(-8)
        rho_conv_tol = 10**(-8)
        e_conv_tol = 10**(-8)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #-------------------------------------------------------------------------------        
        # [2.3] Construction of Fock matrices 
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Notes on Notation Usage:
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # An -(_e) suffix indicates an electronic object.
        # An -(_n) suffix indicates a nuclear object.
        # An -(_en) suffix indicates an electronic-nuclear object.
        # A (c)- prefix indicates an object build using the NEO - CDFT class with an HF class object input. 
        # An (s)- prefix indicates that the NEO-CDFT(HF) Fock Matrix has been semicanonicalized.
        # An (n)- or (nc)- prefix indicates a noncanonical object.

        # Capital letters denote matrices.

        # (H) denotes core Hamiltonian matrices.
        # (V) denotes potential matrices, built from density, Coulomb, and exchange terms. 
        # (S) indicates overlap matrices.
        # (F) indicates the Fock matrices.
        # (_AO) denotes the Fock matrix in the atomic orbital basis.
        # (_MO) denotes the Fock matrix in the molecular orbital basis after transformation using the molecular orbital coefficients.
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        start_timer_3 = timer()

        #------------------------------------------------
        # Canonical NEO-HF(mol) Electronic Objects 
        #------------------------------------------------
        # (These are actually noncanonical now due to setting the dms and coeffs etc. above.)
        # (As such these could be used in the calculations, but I decided to continue using the noncanonical)
        # (objects with the (n)- prefix defined below in the calculations instead.)
        H_e = reg.mf_elec.get_hcore(reg.mol.elec)
        V_e = reg.mf_elec.get_veff(reg.mol.elec, reg.dm_elec)
        S_e = reg.mf_elec.get_ovlp(reg.mol.elec)
        F_e = reg.mf_elec.get_fock(H_e, S_e, V_e, reg.dm_elec)
        F_eMO = reg.mf_elec.mo_coeff.T @ F_e @ reg.mf_elec.mo_coeff

        #---------------------------------------------------
        # Canonical NEO-CDFT(NEO-HF(mol) Electronic Objects
        #---------------------------------------------------
        cH_e = con.mf_elec.get_hcore(con.mol.elec)
        cV_e = con.mf_elec.get_veff(con.mol.elec, con.dm_elec)
        cS_e = con.mf_elec.get_ovlp(con.mol.elec)
        cF_e = con.mf_elec.get_fock(cH_e, cS_e, cV_e, con.dm_elec)
        cF_eMO = con.mf_elec.mo_coeff.T @ cF_e @ con.mf_elec.mo_coeff

        #-----------------------------------------
        # Noncanonical Electronic Objects
        #-----------------------------------------
        ncH_e = reg.mf_elec.get_hcore(reg.mol.elec)
        ncV_e = reg.mf_elec.get_veff(reg.mol.elec, con.dm_elec)
        ncF_eAO = ncH_e + ncV_e
        ncF_eMO = con.mf_elec.mo_coeff.T @ ncF_eAO @ con.mf_elec.mo_coeff

        #---------------------------------------------------------------------------
        # NEO-HF Nuclear Objects - Initialization of Lists and Arrays for Iteration
        #---------------------------------------------------------------------------
        H_n = []
        V_n = []
        S_n = []
        F_n = [None] * reg.mol.nuc_num
        F_nMO = [None] * reg.mol.nuc_num

        #---------------------------------------------------------------------------------
        # NEO-CDFT(HF) Nuclear Objects - Initialization of Lists and Arrays for Iteration
        #---------------------------------------------------------------------------------
        cH_n = []
        cV_n = []
        cS_n = []
        cF_n = [None] * con.mol.nuc_num
        cF_nMO = [None] * con.mol.nuc_num

        #-----------------------------------------------------------------------------------------------------
        # Noncanonical NEO-HF Electronic Fock in MO basis with orbitals obtained from constrained version
        #-----------------------------------------------------------------------------------------------------  
        ncF_nAO = [None]*con.mol.nuc_num
        ncF_nMO = [None]*con.mol.nuc_num

        #---------------------------------
        # NEO-HF Nuclear Objects
        #---------------------------------
        for i in range(len(reg.mol.nuc)):
           H_n.append(reg.mf_nuc[i].get_hcore(reg.mol.nuc[i]))
           V_n.append(reg.mf_nuc[i].get_veff(reg.mol.nuc[i], reg.dm_nuc[i]))
           S_n.append(reg.mf_nuc[i].get_ovlp(reg.mol.nuc[i]))


        #---------------------------------
        # NEO-CDFT(HF) Nuclear Objects
        #---------------------------------
        for i in range(len(con.mol.nuc)):
           cH_n.append(con.mf_nuc[i].get_hcore(con.mol.nuc[i]))
           cV_n.append(con.mf_nuc[i].get_veff(con.mol.nuc[i], con.dm_nuc[i]))
           cS_n.append(con.mf_nuc[i].get_ovlp(con.mol.nuc[i]))

        #---------------------------------
        # NEO-HF Nuclear Objects
        #---------------------------------
        for i in range(len(reg.mol.nuc)):
           F_n[i] = reg.mf_nuc[i].get_fock(H_n[i], S_n[i], V_n[i], reg.dm_nuc[i])
           F_nMO[i] = reg.mf_nuc[i].mo_coeff.T @ F_n[i] @ reg.mf_nuc[i].mo_coeff

        #---------------------------------
        # NEO-CDFT(HF) Nuclear Objects
        #---------------------------------
        for i in range(len(con.mol.nuc)):
           cF_n[i] = con.mf_nuc[i].get_fock(cH_n[i], cS_n[i], cV_n[i], con.dm_nuc[i])
           cF_nMO[i] = con.mf_nuc[i].mo_coeff.T @ cF_n[i] @ con.mf_nuc[i].mo_coeff

        #----------------------------------
        # Noncanonical Nuclear Objects
        #----------------------------------
        ncV_n = []

        for i in range(len(self.reg.mol.nuc)):
            ncV_n.append(reg.mf_nuc[i].get_veff(reg.mol.nuc[i], con.dm_nuc[i]))

        for i in range(len(reg.mol.nuc)):
           ncF_nAO[i] = H_n[i] + ncV_n[i]

        for i in range(len(reg.mol.nuc)):
            ncF_nMO[i] = con.mf_nuc[i].mo_coeff.T @ ncF_nAO[i] @ con.mf_nuc[i].mo_coeff

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        # Constructing the lamba tensors and making sure that the axes are in the right locations.
        # [Note:] I remember I did this because of indexing issues, but I can't remember exactly what had been wrong.         
        #         Since everthing previously was backwards I will definitely have to figure out the right way to fix these.   
        # [Note:] Nevermind, I have since realized that it is because when the electronic nuclear tensors get transposed it   
        #         gives the columns with the electronic and nuclear occupied and electronic and nuclear virtual in the wrong  
        #         order and this fixes everything. 


        eri_ee_full = reg.mol.elec.intor('int2e',aosym='s8')
        co_e= con.mf_elec.mo_coeff[:,:e_nocc]
        cv_e= con.mf_elec.mo_coeff[:,e_nocc:e_tot]
        eri_ee = ao2mo.incore.general(eri_ee_full,(co_e,cv_e,co_e,cv_e), compact=False)

        # Two electron integral tensor:
        TEIMO_t = eri_ee.reshape(e_nocc, e_nvir, e_nocc, e_nvir)

        TEIMO_l = TEIMO_t.T

        # Unnecessary for the electronic case:
        # TEIMO_lambda = numpy.swapaxes(TEIMO.T, 0,1) 
        # TEIMO_lambda = numpy.swapaxes(TEIMO_lambda, 2,3) 

        TPIMO_t = []
        TPIMO_l = []
        for j in range(len(self.con.mol.nuc)):

            p_nocc = int(self.num_ovt[j][0,0])
            p_nvir = int(self.num_ovt[j][0,1])
            p_tot = int(self.num_ovt[j][0,2])

            eri_ep = neo.ao2mo.ep_ovov(self.con, self.con, j)
            TPIMO_en = eri_ep.reshape(e_nocc, e_nvir, p_nocc, p_nvir)
            TPIMO_lambda_en = TPIMO_en.T
            TPIMO_lambda_en = numpy.swapaxes(TPIMO_en.T, 0,2)
            TPIMO_lambda_en = numpy.swapaxes(TPIMO_lambda_en, 1,3)
            TPIMO_t.append(TPIMO_en)
            TPIMO_l.append(TPIMO_lambda_en)

#        T_lambda_en = t_amps_en[j].T
#        T_lambda_en = numpy.swapaxes(t_amps_en[j].T, 0,2)
#        T_lambda_en = numpy.swapaxes(T_lambda_en, 1,3)    

        TNIMO_t = numpy.empty((len(self.con.mol.nuc),len(self.con.mol.nuc)), dtype=object)
        TNIMO_l = numpy.empty((len(self.con.mol.nuc),len(self.con.mol.nuc)), dtype=object)

        for i in range(len(self.con.mol.nuc)):
            for j in range(len(self.con.mol.nuc)):

                p_nocc_i = int(self.num_ovt[i][0,0])
                p_nvir_i = int(self.num_ovt[i][0,1])
                p_tot_i = int(self.num_ovt[i][0,2])
                p_nocc_j = int(self.num_ovt[j][0,0])
                p_nvir_j = int(self.num_ovt[j][0,1])
                p_tot_j = int(self.num_ovt[j][0,2])

                eri_pp = neo.ao2mo.pp_ovov(self.con, self.con, i, j)
                TNIMO = eri_pp.reshape(p_nocc_i, p_nvir_i, p_nocc_j, p_nvir_j)

                TNIMO_lambda = numpy.swapaxes(TNIMO.T, 0,2)
                TNIMO_lambda = numpy.swapaxes(TNIMO_lambda, 1,3)
                TNIMO_t[i,j] = TNIMO
                TNIMO_l[i,j] = TNIMO_lambda


       # T_lambda_n = numpy.swapaxes(t_amps_n[i][j].T, 0,2)
       # T_lambda_n = numpy.swapaxes(T_lambda_n, 1,3)
        finish_timer_3 = timer()

        print('timer[3]: total time for building matrices and integral tensors: ', finish_timer_3  - start_timer_3)

        #-------------------------------------------------------------------------------        
        # [2.4] Lagrangian constraint upon the MP2 nuclear densities
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #----------------------------------
        # Nuclear position integrals
        #----------------------------------
        integrals_r = [None] * con.mol.nuc_num
        for i in range(len(self.con.mol.nuc)):

            integrals_r[i] = self.con.mf_nuc[i].mol.intor_symmetric('int1e_r', comp=3)
        #----------------------------------------------------------------
        # Transformation of position integrals to the  MO basis...:
        #----------------------------------------------------------------
#        for i in range(len(self.con.mol.nuc)):
#            for x in range(3):

#                integrals_r[i][x,:,:] = self.con.mf_nuc[i].mo_coeff.T @ integrals_r[i][x,:,:] @ self.con.mf_nuc[i].mo_coeff
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        def Lagrangian_constraint(lagr_update, self, nuc_idx, t_nuclear, t_electronic_nuclear, position_ints):
        
            start_timer_4 = timer()

            print(self.lagr)
            print(lagr_update)

            self.try_lagr[nuc_idx] = lagr_update
   
            cycles = 100
            for t in range(cycles):
            
                start_timer_5 = timer()

                t_update_en = t_amps_en_only(self, nuc_idx, lagr_update, t_electronic_nuclear[nuc_idx], self.l_elecnuc[nuc_idx], TPIMO_t[nuc_idx], ncF_eMO, ncF_nMO[nuc_idx], integrals_r[nuc_idx])

                self.t_opt_en[nuc_idx] = t_update_en 
                self.l_opt_en[nuc_idx] = t_update_en.T
                intermediate_l_opt_en = numpy.swapaxes(self.l_opt_en[nuc_idx], 0,2)
                self.l_opt_en[nuc_idx]= numpy.swapaxes(intermediate_l_opt_en, 1,3)

                finish_timer_5 = timer()

                print('timer[5]: time to update electronic-nuclear amplitude: ', finish_timer_5 - start_timer_5)

                start_timer_6 = timer()

                for nuc_2_idx in range(len(self.con.mol.nuc)):
                     
                    t_update_n = t_amps_n_only(self, nuc_idx, nuc_2_idx, lagr_update, self.try_lagr[nuc_2_idx], t_nuclear[nuc_idx][nuc_2_idx], self.l_nuc[nuc_idx][nuc_2_idx], TNIMO_t[nuc_idx][nuc_2_idx], ncF_nMO[nuc_idx], ncF_nMO[nuc_2_idx], integrals_r[nuc_idx], integrals_r[nuc_2_idx])
                    t_update_n_swap_int = numpy.swapaxes(t_update_n, 0,2)
                    t_update_n_swap = numpy.swapaxes(t_update_n_swap_int, 1,3)
                    self.t_opt_n[nuc_idx][nuc_2_idx] = t_update_n
                    self.t_opt_n[nuc_2_idx][nuc_idx] = t_update_n_swap           

                    self.l_opt_n[nuc_idx][nuc_2_idx] = t_update_n.T
                    intermediate_l_opt_n = numpy.swapaxes(self.l_opt_n[nuc_idx][nuc_2_idx], 0,2)
                    self.l_opt_n[nuc_idx][nuc_2_idx] = numpy.swapaxes(intermediate_l_opt_n, 1,3)
                    l_update_n_swap_int = numpy.swapaxes(self.l_opt_n[nuc_idx][nuc_2_idx], 0,2)
                    l_update_n_swap = numpy.swapaxes(l_update_n_swap_int, 1,3)
                    self.l_opt_n[nuc_2_idx][nuc_idx] = l_update_n_swap

                finish_timer_6 = timer()

                print('timer[6]: time to update nuclear amplitudes: ', finish_timer_6 - start_timer_6)



                #June 17th - what happens if I don't update the self.t_elecnuc and self.t_nuc objects yet?  Tried this (i.e. commenting these lines out) but didn't change anything. 
                start_timer_7 = timer()

                self.t_elecnuc_test[nuc_idx] = self.t_opt_en[nuc_idx]
                self.l_elecnuc_test[nuc_idx] = self.l_opt_en[nuc_idx]

                for nuc_2_idx in range(len(self.con.mol.nuc)):

                    self.t_nuc_test[nuc_idx][nuc_2_idx] = self.t_opt_n[nuc_idx][nuc_2_idx]
                    t_nuc_test_swap_int = numpy.swapaxes(self.t_nuc_test[nuc_idx][nuc_2_idx], 0,2)
                    t_nuc_test_swap = numpy.swapaxes(t_nuc_test_swap_int, 1,3)
                    self.t_nuc_test[nuc_2_idx][nuc_idx] = t_nuc_test_swap
                    self.l_nuc_test[nuc_idx][nuc_2_idx] = self.l_opt_n[nuc_idx][nuc_2_idx]
                    self.l_nuc_test[nuc_2_idx][nuc_idx] = self.l_opt_n[nuc_2_idx][nuc_idx]

                finish_timer_7 = timer()

                print('timer[7]: total time for redefinitions and axes swaps: ', finish_timer_7 - start_timer_7)

                start_timer_8 = timer()

                t_diis_opt_en = (cneomp2_diis(t_update_en, e_nocc, e_tot, int(n_ovt[nuc_idx][0,0]), int(n_ovt[nuc_idx][0,2])).kernel())
                self.t_opt_en[nuc_idx] = t_diis_opt_en
                self.l_opt_en[nuc_idx] = t_diis_opt_en.T
                intermediate_l_diis_en = numpy.swapaxes(self.l_opt_en[nuc_idx], 0,2)
                self.l_opt_en[nuc_idx]= numpy.swapaxes(intermediate_l_diis_en, 1,3)

                finish_timer_8 = timer()

                print('timer[8]: total time for electronic-nuclear diis inside LC function: ', finish_timer_8 - start_timer_8)

                start_timer_9 = timer()

                for nuc_2_idx in range(len(self.con.mol.nuc)):

                    t_diis_opt_n = cneomp2_diis(self.t_opt_n[nuc_idx][nuc_2_idx], int(n_ovt[nuc_idx][0,0]), int(n_ovt[nuc_idx][0,2]), int(n_ovt[nuc_2_idx][0,0]), int(n_ovt[nuc_2_idx][0,2])).kernel()
                    t_diis_n_swap_int = numpy.swapaxes(t_diis_opt_n, 0,2)
                    t_diis_n_swap = numpy.swapaxes(t_diis_n_swap_int, 1,3)
                    self.t_opt_n[nuc_idx][nuc_2_idx] = t_diis_opt_n
                    self.t_opt_n[nuc_2_idx][nuc_idx] = t_diis_n_swap

                    self.l_opt_n[nuc_idx][nuc_2_idx] = t_diis_opt_n.T
                    intermediate_l_diis_n = numpy.swapaxes(self.l_opt_n[nuc_idx][nuc_2_idx], 0,2)
                    self.l_opt_n[nuc_idx][nuc_2_idx] = numpy.swapaxes(intermediate_l_diis_n, 1,3)
                    l_diis_n_swap_int = numpy.swapaxes(self.l_opt_n[nuc_idx][nuc_2_idx], 0,2)
                    l_diis_n_swap = numpy.swapaxes(l_diis_n_swap_int, 1,3)
                    self.l_opt_n[nuc_2_idx][nuc_idx] = l_diis_n_swap


                finish_timer_9 = timer()

                print('timer[9]: total time for nuclear diis inside LC function: ', finish_timer_9 - start_timer_9)

                start_timer_10 = timer()

                t_new_lambda_en = t_amps_en_only(self, nuc_idx, self.try_lagr[nuc_idx], self.t_opt_en[nuc_idx], self.l_opt_en[nuc_idx], TPIMO_t[nuc_idx], ncF_eMO, ncF_nMO[nuc_idx], integrals_r[nuc_idx])
                self.t_opt_en[nuc_idx] = t_new_lambda_en
                l_new_lambda_en = t_new_lambda_en.T
                intermediate_l_lambda_en = numpy.swapaxes(l_new_lambda_en, 0,2)
                self.l_opt_en[nuc_idx]= numpy.swapaxes(intermediate_l_lambda_en, 1,3)

                finish_timer_10 = timer()

                print('timer[10]: total time for rebuilding electronic-nuclear t-amps after diis inside LC function: ', finish_timer_10 - start_timer_10)

                start_timer_11 = timer()

                for nuc_2_idx in range(len(self.con.mol.nuc)):
                    
                    t_new_lambda_n = t_amps_n_only(self, nuc_idx, nuc_2_idx, self.try_lagr[nuc_idx], self.lagr[nuc_2_idx], self.t_opt_n[nuc_idx][nuc_2_idx], self.l_opt_n[nuc_idx][nuc_2_idx], TNIMO_t[nuc_idx][nuc_2_idx], ncF_nMO[nuc_idx], ncF_nMO[nuc_2_idx], integrals_r[nuc_idx], integrals_r[nuc_2_idx])
                    
                    t_new_n_swap_int = numpy.swapaxes(t_new_lambda_n, 0,2)
                    t_new_n_swap = numpy.swapaxes(t_new_n_swap_int, 1,3)
                    self.t_opt_n[nuc_idx][nuc_2_idx] = t_new_lambda_n  
                    self.t_opt_n[nuc_2_idx][nuc_idx] = t_new_n_swap
            
                    l_new_lambda_n = t_new_lambda_n.T
                    l_new_n_int = numpy.swapaxes(l_new_lambda_n, 0,2)
                    self.l_opt_n[nuc_idx][nuc_2_idx] = numpy.swapaxes(l_new_n_int, 1,3)
                    l_new_n_swap_int = numpy.swapaxes(self.l_opt_n[nuc_idx][nuc_2_idx], 0,2)
                    l_new_n_swap = numpy.swapaxes(l_new_n_swap_int, 1,3)
                    self.l_opt_n[nuc_2_idx][nuc_idx] = l_new_n_swap

                finish_timer_11 = timer()

                print('timer[11]: total time for rebuilding nuclear t-amps after diis inside LC function: ', finish_timer_11 - start_timer_11)

                start_timer_12 = timer()

                diff_E_opt_en = abs(Hylleraas_energy_en(self, self.t_opt_en, self.l_opt_en, TPIMO_t, TPIMO_l, ncF_eMO, ncF_nMO) - Hylleraas_energy_en(self, self.t_elecnuc_test, self.l_elecnuc_test, TPIMO_t, TPIMO_l, ncF_eMO, ncF_nMO))

                diff_E_opt_n = abs(Hylleraas_energy_n(self, self.t_opt_n, self.l_opt_n, TNIMO_t, TNIMO_l, ncF_nMO) - Hylleraas_energy_n(self, self.t_nuc_test, self.l_nuc_test, TNIMO_t, TNIMO_l, ncF_nMO))
              
                finish_timer_12 = timer()

                print('timer[12]: total time for energy convergence checks inside LC function: ', finish_timer_12 - start_timer_12)
 
                start_timer_13 = timer()
              
                opt_RMSD_en = RMSD_en(self, self.t_opt_en, self.t_elecnuc_test)
                opt_RMSD_n = RMSD_n(self, self.t_opt_n, self.t_nuc_test)

                finish_timer_13 = timer()

                print('timer[13]: total time for RMSD checks inside LC function: ', finish_timer_13 - start_timer_13)

                start_timer_14 = timer()

                hyll_en_opt = Hylleraas_energy_en(self, self.t_opt_en, self.l_opt_en, TPIMO_t, TPIMO_l, ncF_eMO, ncF_nMO)
                hyll_n_opt = Hylleraas_energy_n(self, self.t_opt_n, self.l_opt_n, TNIMO_t, TNIMO_l, ncF_nMO)

                finish_timer_14 = timer()

                print('timer[14]: total time for calculating final energies inside LC function: ', finish_timer_14 - start_timer_14)
                

                print('SUBCYCLE: ', i, 'ITERATION: ', t, '-------', 'E(MP2)_e: ', hyll_e, 'E(MP2)_en: ', hyll_en_opt, 'E(MP2)_n: ', hyll_n_opt)
                print(line)


                print('RMSD: ', ' en: ', opt_RMSD_en, ' n: ', opt_RMSD_n)
                print(line)

                print('diff_E: ', ' en: ', diff_E_opt_en, ' n: ', diff_E_opt_n)
                print(line)

                self.t_elecnuc_test = self.t_opt_en
                self.t_nuc_test = self.t_opt_n
                self.l_elecnuc_test = self.l_opt_en
                self.l_nuc_test = self.l_opt_n

                #if (opt_RMSD_en < t_conv_tol) and (opt_RMSD_n < t_conv_tol) and (diff_E_opt_en < e_conv_tol) and (diff_E_opt_n < e_conv_tol):
                if (opt_RMSD_en < conv_current) and (opt_RMSD_n < conv_current) and (diff_E_opt_en < conv_current) and (diff_E_opt_n < conv_current):
                    print('SUCCESSFUL CONVERGENCE INNER LOOP')
                    self.mp2_density_converged[nuc_idx] = True
                    #hyll_en = Hylleraas_energy_en(self, self.t_elecnuc, ncF_eMO, ncF_nMO)
                    #hyll_n = Hylleraas_energy_n(self, self.t_nuc, ncF_nMO)
                    break
                     
                else:
                    if t>98: print('WARNING, NOT CONVERGED')
                    else: 
                        print('in opt, continuing to next subcycle of t amplitude convergence for a single iteration of lambda LM.')                

            start_timer_15 = timer()

            density_matrix = mp2_density_one(self, nuc_idx, self.t_nuc_test, self.l_nuc_test, self.t_elecnuc_test, self.l_elecnuc_test)

            finish_timer_15 = timer()

            print('timer[15]: total time for calculating mp2 nuclear single-particle density matrix inside LC function: ', finish_timer_15 - start_timer_15)

            start_timer_16 = timer()
             
            constraint = numpy.einsum('xij, ji->x', position_ints[nuc_idx], density_matrix)
            
            finish_timer_16 = timer()

            print('timer[16]: total time for calculation of the constraint itself inside LC function: ', finish_timer_16 - start_timer_16)

            if (constraint.all() < 0.002):
                self.t_elecnuc = self.t_opt_en
                self.t_nuc = self.t_opt_n
 
            else:
                pass

            print('...')
            print('Constraint: ', constraint)
            print('...')

            finish_timer_4 = timer()

            print('timer[4]: total runtime for the LC function: ', finish_timer_4 - start_timer_4)

            return numpy.einsum('xij, ji->x', position_ints[nuc_idx], density_matrix) 

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #-------------------------------------------------------------------------------        
        # [2.5] Electronic t-amplitude function 
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # See cymods.t_amps_e_only module.

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #-------------------------------------------------------------------------------        
        # [2.6] Electronic-nuclear t-amplitude function 
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # See cymods.t_amps_en_only module.

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #-------------------------------------------------------------------------------        
        # [2.7] Nuclear t-amplitude function 
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # See cymods.t_amps_n_only module.

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        #-------------------------------------------------------------------------------        
        # [2.8] Nuclear single-particle MP2 density matrix functions 
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # See cymods.mp2_dens module.

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        #-------------------------------------------------------------------------------        
        # [2.9] Hylleraas electronic energy function 
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # See cymods.hylleraas_e module.

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        #-------------------------------------------------------------------------------        
        # [2.10] Hylleraas electronic-nuclear energy function 
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # See cymods.hylleraas_en module.

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        #-------------------------------------------------------------------------------        
        # [2.11] Hylleraas nuclear energy function 
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # See cymods.hylleraas_n module.

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        #-------------------------------------------------------------------------------        
        # [2.12] SCF optimization procedure 
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        max_iteration_cycles = 300

        conv_start = 10**(-8)
        conv_current = conv_start
        conv_tol = 10**(-8)

        constraint_opt = [None]*len(con.mol.nuc)
        opt_RMSD_en = 0
        opt_RMSD_n = 0
        diff_E_opt_en = 0
        diff_E_opt_n = 0

        enuc_t_holder_1 = numpy.empty(len(self.con.mol.nuc), object)
        enuc_t_holder_2 = numpy.empty(len(self.con.mol.nuc), object)
        enuc_t_holder_3 = numpy.empty(len(self.con.mol.nuc), object)

        enuc_l_holder_1 = numpy.empty(len(self.con.mol.nuc), object)
        enuc_l_holder_2 = numpy.empty(len(self.con.mol.nuc), object)
        enuc_l_holder_3 = numpy.empty(len(self.con.mol.nuc), object)
        
        nuc_t_holder_1 = numpy.empty((len(self.con.mol.nuc), len(self.con.mol.nuc)), object)
        nuc_t_holder_2 = numpy.empty((len(self.con.mol.nuc), len(self.con.mol.nuc)), object)
        nuc_t_holder_2 = numpy.empty((len(self.con.mol.nuc), len(self.con.mol.nuc)), object)

        nuc_l_holder_1 = numpy.empty((len(self.con.mol.nuc), len(self.con.mol.nuc)), object)
        nuc_l_holder_2 = numpy.empty((len(self.con.mol.nuc), len(self.con.mol.nuc)), object)
        nuc_l_holder_3 = numpy.empty((len(self.con.mol.nuc), len(self.con.mol.nuc)), object)

        start_timer_17 = timer()

        for t in range(max_iteration_cycles):



            #if (t > 0):
            #    if t%2==0 and (conv_current>conv_tol):
            #        conv_current=conv_current*0.1
            #        print('current convergence tolerance: ', conv_current)
            #    else:
            #        print('current convergence tolerance: ', conv_current)
            #        pass
            #else:
            #    pass

            print(asterisk)
            print('THIS IS SCF CYCLE NO: ', t)

            for j in range(len(self.con.mol.nuc)):
                self.l_elecnuc[j] = self.t_elecnuc[j].T
                intermediate_l_en = numpy.swapaxes(self.l_elecnuc[j], 0,2)
                self.l_elecnuc[j]= numpy.swapaxes(intermediate_l_en, 1,3)  
            for i in range(len(self.con.mol.nuc)):
                for j in range(len(self.con.mol.nuc)):
                    self.l_nuc[i][j] = self.t_nuc[i][j].T
                    intermediate_l_n = numpy.swapaxes(self.l_nuc[i][j], 0,2)
                    self.l_nuc[i][j] = numpy.swapaxes(intermediate_l_n, 1,3)
  

            R_mp2 = self.con.mol.atom_coords(unit='ANG')

            t_old_n = self.t_nuc
            t_old_e = self.t_elec
            t_old_en = self.t_elecnuc
            
            l_old_n = self.l_nuc
            l_old_e = self.l_elec
            l_old_en = self.l_elecnuc

            start_timer_18 = timer()

            old_hyll_n = Hylleraas_energy_n(self, t_old_n, l_old_n, TNIMO_t, TNIMO_l, ncF_nMO)
            old_hyll_e = Hylleraas_energy_e(self, t_old_e, l_old_e, TEIMO_t, TEIMO_l, ncF_eMO)
            old_hyll_en = Hylleraas_energy_en(self, t_old_en, l_old_en, TPIMO_t, TPIMO_l, ncF_eMO, ncF_nMO)
            
            finish_timer_18 = timer()

            print('timer[18]: total time for calculation of initial/previous iteration energies: ', finish_timer_18 - start_timer_18)

            start_timer_19 = timer()

            if (t>3):

                t_diis_int_e = cneomp2_diis(t_old_e, e_nocc, e_tot, e_nocc, e_tot).kernel()

            
                t_diis_int_en = []
                for i in range(len(self.con.mol.nuc)):
                    t_diis_int_en.append(cneomp2_diis(t_old_en[i], e_nocc, e_tot, int(n_ovt[i][0,0]), int(n_ovt[i][0,2])).kernel())
               
                t_diis_int_n = numpy.empty((len(self.con.mol.nuc),len(self.con.mol.nuc)), dtype=object)
                for i in range(len(self.con.mol.nuc)):
                    for j in range(len(self.con.mol.nuc)):
                        t_diis_int_n[i][j] = cneomp2_diis(t_old_n[i][j], int(n_ovt[i][0,0]), int(n_ovt[i][0,2]), int(n_ovt[j][0,0]), int(n_ovt[j][0,2])).kernel()

                self.t_elec = t_diis_int_e
                self.t_elenuc = t_diis_int_en
                self.t_nuc = t_diis_int_n
                self.l_elec = t_diis_int_e

                for j in range(len(self.con.mol.nuc)):
                    self.l_elecnuc[j] = self.t_elecnuc[j].T
                    intermediate_l_en = numpy.swapaxes(self.l_elecnuc[j], 0,2)
                    self.l_elecnuc[j]= numpy.swapaxes(intermediate_l_en, 1,3)
                for i in range(len(self.con.mol.nuc)):
                    for j in range(len(self.con.mol.nuc)):
                        self.l_nuc[i][j] = self.t_nuc[i][j].T
                        intermediate_l_n = numpy.swapaxes(self.l_nuc[i][j], 0,2)
                        self.l_nuc[i][j] = numpy.swapaxes(intermediate_l_n, 1,3)

            else: 
   
                pass

            finish_timer_19 = timer()

            print('timer[19]: total time for diis in main iteration loop: ', finish_timer_19 - start_timer_19)


            start_timer_20 = timer()

            t_new_e = t_amps_e_only(self, self.t_elec, TEIMO_t, ncF_eMO)
            l_new_e = t_new_e.T
            
            finish_timer_20 = timer()

            print('timer[20]: total time for calculation of new electronic t amps in main interation loop: ', finish_timer_20 - start_timer_20)


            t_new_en = [None]*len(self.con.mol.nuc)
            l_new_en = [None]*len(self.con.mol.nuc)

            start_timer_21 = timer()

            for i in range(len(self.con.mol.nuc)):

                t_new_en[i] = t_amps_en_only(self, i, self.lagr[i], self.t_elecnuc[i], self.l_elecnuc[i], TPIMO_t[i], ncF_eMO, ncF_nMO[i], integrals_r[i])
                l_new_en[i] = t_new_en[i].T
                l_new_intermediate_en = numpy.swapaxes(l_new_en[i], 0,2)
                l_new_en[i] = numpy.swapaxes(l_new_intermediate_en, 1,3)

            finish_timer_21 = timer()

            print('timer[21]: total time for calculation of new electronic nuclear t amps in main iteration loop: ', finish_timer_21 - start_timer_21)

            start_timer_22 = timer()

            t_new_n = numpy.empty((len(self.con.mol.nuc),len(self.con.mol.nuc)), dtype=object)
            l_new_n = numpy.empty((len(self.con.mol.nuc),len(self.con.mol.nuc)), dtype=object)
            for i in range(len(self.con.mol.nuc)):
                for j in range(len(self.con.mol.nuc)):

                    t_new_n[i][j] = t_amps_n_only(self, i, j, self.lagr[i], self.lagr[j], self.t_nuc[i][j], self.l_nuc[i][j], TNIMO_t[i][j], ncF_nMO[i], ncF_nMO[j], integrals_r[i], integrals_r[j])
                    l_new_n[i][j] = t_new_n[i][j].T
                    l_new_intermediate_n = numpy.swapaxes(l_new_n[i][j], 0,2)
                    l_new_n[i][j] = numpy.swapaxes(l_new_intermediate_n, 1,3)

            finish_timer_22 = timer()

            print('timer[22]: total time for calculation of new nuclear t amps in main iteration loop: ', finish_timer_22 - start_timer_22)


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        #-------------------------------------------------------------------------------        
        # [2.13] RMSD calculations  
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # The following section calculates the root mean square differences in the t-amplitudes
        #---------------------------------------------------------------------------------------------
            start_timer_23 = timer()

            check_RMSD_e = RMSD_e(self, t_old_e, t_new_e)
            check_RMSD_en = RMSD_en(self, t_old_en, t_new_en)
            check_RMSD_n = RMSD_n(self, t_old_n, t_new_n)
 
            finish_timer_23 = timer()

            print('timer[23]: total time for checking RMSD in main iteration loop: ', finish_timer_23 - start_timer_23)


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        #-------------------------------------------------------------------------------        
        # [2.14] Check all convergence criteria 
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.t_nuc = t_new_n
            self.t_elec = t_new_e
            self.t_elecnuc = t_new_en
            self.l_nuc = l_new_n
            self.l_elec = t_new_e.T
            self.l_elecnuc = l_new_en


            self.t_opt_en = self.t_elecnuc
            self.t_opt_n = self.t_nuc
            self.l_opt_en = self.l_elecnuc
            self.l_opt_n = self.l_nuc

            #-------------------------------------------------------------------------------
            # Check for convergence and satisfaction of constraint:
            #-------------------------------------------------------------------------------
   
            start_timer_24 = timer()

            start_timer_25 = timer()
            hyll_n = Hylleraas_energy_n(self, t_new_n, l_new_n, TNIMO_t, TNIMO_l, ncF_nMO)
            finish_timer_25 = timer()
            print('timer[25]: total time for calculation of new nuclear energy in main iteration loop: ', finish_timer_25 - start_timer_25)
            
            start_timer_26 = timer()
            hyll_e = Hylleraas_energy_e(self, t_new_e, l_new_e, TEIMO_t, TEIMO_l, ncF_eMO)
            finish_timer_26 = timer()
            print('timer[26]: total time for calculation of new electronic energy in main iteration loop: ', finish_timer_26 - start_timer_26)

            start_timer_27 = timer()
            hyll_en = Hylleraas_energy_en(self, t_new_en, l_new_en, TPIMO_t, TPIMO_l, ncF_eMO, ncF_nMO)
            finish_timer_27 = timer()
            print('timer[27]: total time for calculation of new electronic nuclear energy in main iteration loop: ', finish_timer_27 - start_timer_27)


            finish_timer_24 = timer()

            print('timer[24]: total time for calculation of new energies in main convergence loop: ', finish_timer_24 - start_timer_24)
 
            start_timer_28 = timer()
            diff_E_n = hyll_n - old_hyll_n
            diff_E_e = hyll_e - old_hyll_e
            diff_E_en = hyll_en - old_hyll_en
            finish_timer_28 = timer()
            print('timer[28]: total time for calculation of energy convergence checks in main iteration loop: ', finish_timer_28 - start_timer_28)

            print(asterisk)

            print('ITERATION: ', t, '-------', 'E(MP2)_e: ', hyll_e, 'E(MP2)_en: ', hyll_en, 'E(MP2)_n: ', hyll_n)
            print(line)


            print('RMSD: ', ' e :', check_RMSD_e, ' en: ', check_RMSD_en, ' n: ', check_RMSD_n)
            print(line)

            print('diff_E: ', ' e: ', diff_E_e, ' en: ', diff_E_en, ' n: ', diff_E_n)
            print(line)

           # if (check_RMSD_e < t_conv_tol) and (check_RMSD_n < t_conv_tol) and (check_RMSD_en < t_conv_tol) and (abs(diff_E_n) < e_conv_tol) and (abs(diff_E_e) < e_conv_tol) and (abs(diff_E_en) < e_conv_tol):
            if (check_RMSD_e < conv_current) and (check_RMSD_n < conv_current) and (check_RMSD_en < conv_current) and (abs(diff_E_n) < conv_current) and (abs(diff_E_e) < conv_current) and (abs(diff_E_en) < conv_current):
                start_timer_29 = timer()
                print('t-amplitudes and mp2 energies are converged!')

                enuc_t_holder_1[:] = self.t_elecnuc
                enuc_l_holder_1[:] = self.l_elecnuc

                nuc_t_holder_1 = self.t_nuc
                nuc_l_holder_1 = self.l_nuc


                numpy.save('bifluoride_anion.cc-pvtz.cneomp2.t_elec.npy''.npy', self.t_elec, allow_pickle=True, fix_imports=True)
                numpy.save('bifluoride_anion.cc-pvtz.cneomp2.l_elec.npy'+'.npy', self.l_elec, allow_pickle=True, fix_imports=True)

                for i in range(len(self.con.mol.nuc)):

                    numpy.save('bifluoride_anion.cc-pvtz.cneomp2.t_elecnuc.npy'+'.'+str(i)+'.npy', enuc_t_holder_1[i], allow_pickle=True, fix_imports=True)
                    numpy.save('bifluoride_anion.cc-pvtz.cneomp2.l_elecnuc.npy'+'.'+str(i)+'.npy', enuc_l_holder_1[i], allow_pickle=True, fix_imports=True)

                for i in range(len(self.con.mol.nuc)):
                    for j in range(len(self.con.mol.nuc)):

                        numpy.save('bifluoride_anion.cc-pvtz.cneomp2.t_nuc.npy'+'.'+str(i)+str(j)+'.npy', nuc_t_holder_1[i][j], allow_pickle=True, fix_imports=True)
                        numpy.save('bifluoride_anion.cc-pvtz.cneomp2.l_nuc.npy'+'.'+str(i)+str(j)+'.npy', nuc_l_holder_1[i][j], allow_pickle=True, fix_imports=True)


                finish_timer_29 = timer()
                print('timer[29]: total time for saving amplitudes after initial convergence in main iteration loop: ', finish_timer_29 - start_timer_29)

                for i in range(len(self.con.mol.nuc)):

                    start_timer_30 = timer()
                    constraint_opt[i] = scipy.optimize.root(Lagrangian_constraint, self.try_lagr[i].flatten(), args=(self, i, self.t_nuc, self.t_elecnuc, self.posn_ints), method='lm', jac=False, tol=1e-5, options={'col_deriv':True, 'ftol':1e-5, 'xtol':1e-5, 'gtol':1e-5, 'maxiter':1000, })         
                    print(asterisk)
                    print(asterisk)
                    print(asterisk)
                    print(constraint_opt[i].status)
                    print(constraint_opt[i].message)
                    print(asterisk)
                    print(asterisk)
                    print(asterisk)
                    finish_timer_30 = timer()
                    print('timer[30]: total runtime for call to scipy optimize (this is a per nucleus time): ', finish_timer_30 - start_timer_30)


                    if (constraint_opt[i].status == 1) or (constraint_opt[i].status == 2):

                        self.lagr[i] = constraint_opt[i].x
 
                    else:
                        pass
                

#                    t_opt_en = self.t_opt_en 
#
#                    t_opt_n = self.t_opt_n 
                 
#
#                    t_diis_opt_en = []
#                    t_diis_int_en.append(cneomp2_diis(t_opt_en[i], e_nocc, e_tot, int(n_ovt[i][0,0]), int(n_ovt[i][0,2])).kernel())
#
#                    t_diis_opt_n = numpy.empty((len(self.con.mol.nuc),len(self.con.mol.nuc)), dtype=object)
#                    for j in range(len(self.con.mol.nuc)):
#                        t_diis_int_n[i][j] = cneomp2_diis(t_opt_n[i][j], int(n_ovt[i][0,0]), int(n_ovt[i][0,2]), int(n_ovt[j][0,0]), int(n_ovt[j][0,2])).kernel()

#
#                    self.t_opt_en[i] = t_diis_int_en[i]
                   
#                    for j in range(len(self.con.mol.nuc)):
#                        self.t_opt_n[i][j] = t_diis_int_n[i][j]
#                        self.t_opt_n[j][i] = t_diis_int_n[j][i]

#                    t_new_lambda_en[i] = t_amps_en_only(self, i, self.try_lagr[i], self.t_opt_en[i], ncF_eMO, ncF_nMO[i], integrals_r[i]) 

#                    for j in range(len(self.con.mol.nuc)):
#                        t_new_lambda_n[i][j] = t_amps_n_only(self, i, j, self.try_lagr[i], self.try_lagr[j], self.t_nuc[i][j], ncF_nMO[i], ncF_nMO[j], integrals_r[i], integrals_r[j])


#                    diff_E_opt_en = abs(Hylleraas_energy_en(self, t_new_lambda_en, ncF_eMO, ncF_nMO) - Hylleraas_energy_en(self, t_new_en, ncF_eMO, ncF_nMO))

#                    diff_E_opt_n = abs(Hylleraas_energy_n(self, t_new_lambda_n, ncF_nMO) - Hylleraas_energy_n(self, t_new_n, ncF_nMO))

                    start_timer_31 = timer()

                    opt_RMSD_en_mloop = RMSD_en(self, self.t_elecnuc, t_new_en)
                    opt_RMSD_n_mloop = RMSD_n(self, self.t_nuc, t_new_n)

                    hyll_en_opt = Hylleraas_energy_en(self, self.t_elecnuc, self.l_elecnuc, TPIMO_t, TPIMO_l, ncF_eMO, ncF_nMO)
                    hyll_n_opt = Hylleraas_energy_n(self, self.t_nuc, self.l_nuc, TNIMO_t, TNIMO_l, ncF_nMO) 

                    diff_E_opt_en = hyll_en_opt - hyll_en
                    diff_E_opt_n = hyll_n_opt - hyll_n

                    finish_timer_31 = timer()

                    print('timer[31]: total time for calculation of new vs newer optimized en and n objects and energies convergence checks in main iteration loop: ', finish_timer_31 - start_timer_31)
#                    print('SUBCYCLE: ', i, 'ITERATION: ', t, '-------', 'E(MP2)_e: ', hyll_e, 'E(MP2)_en: ', hyll_en_opt, 'E(MP2)_n: ', hyll_n_opt)
#                    print(line)


#                    print('RMSD: ', ' en: ', opt_RMSD_en, ' n: ', opt_RMSD_n)
#                    print(line)

#                    print('diff_E: ', ' en: ', diff_E_opt_en, ' n: ', diff_E_opt_n)
#                    print(line)

                    if (constraint_opt[i].status == 1) or (constraint_opt[i].status == 2):
                        if (opt_RMSD_en_mloop < t_conv_tol) and (opt_RMSD_n_mloop < t_conv_tol) and (diff_E_opt_en < e_conv_tol) and (diff_E_opt_n < e_conv_tol):

                            print('SUCCESSFUL CONVERGENCE OUTER LOOP')
                            self.mp2_density_converged[i] = True
                            hyll_en = Hylleraas_energy_en(self, self.t_elecnuc, self.l_elecnuc, TPIMO_t, TPIMO_l, ncF_eMO, ncF_nMO)
                            hyll_n = Hylleraas_energy_n(self, self.t_nuc, self.l_nuc, TNIMO_t, TNIMO_l, ncF_nMO)

                            enuc_t_holder_2[:] = self.t_elecnuc
                            enuc_l_holder_2[:] = self.l_elecnuc

                            nuc_t_holder_2 = self.t_nuc
                            nuc_l_holder_2 = self.l_nuc

                            numpy.save('bifluoride_anion.cc-pvtz.cneomp2.t_elec.npy'+'.npy', self.t_elec, allow_pickle=True, fix_imports=True)
                            numpy.save('bifluoride_anion.cc-pvtz.cneomp2.l_elec.npy'+'.npy', self.l_elec, allow_pickle=True, fix_imports=True)

                            for i in range(len(self.con.mol.nuc)):

                                numpy.save('bifluoride_anion.cc-pvtz.cneomp2.t_elecnuc.npy'+'.'+str(i)+'.npy', enuc_t_holder_2[i], allow_pickle=True, fix_imports=True)
                                numpy.save('bifluoride_anion.cc-pvtz.cneomp2.l_elecnuc.npy'+'.'+str(i)+'.npy', enuc_l_holder_2[i], allow_pickle=True, fix_imports=True)

                            for i in range(len(self.con.mol.nuc)):
                                for j in range(len(self.con.mol.nuc)):

                                    numpy.save('bifluoride_anion.cc-pvtz.cneomp2.t_nuc.npy'+'.'+str(i)+str(j)+'.npy', nuc_t_holder_2[i][j], allow_pickle=True, fix_imports=True)
                                    numpy.save('bifluoride_anion.cc-pvtz.cneomp2.l_nuc.npy'+'.'+str(i)+str(j)+'.npy', nuc_l_holder_2[i][j], allow_pickle=True, fix_imports=True)

                    else:
 
                        self.mp2_density_converged[i] = False

                if all(self.mp2_density_converged):
                    for i in range(len(self.con.mol.nuc)):
                        print(constraint_opt[i])
                        print('SUCCESSFUL CONVERGENCE ACHIEVED FOR ALL NUCLEAR AND ELECTRONIC NUCLEAR T-amplitudes...EXITING ITERATION CYCLE...HURRAY!')
                        print('[',i,'[', self.lagr[i])

                        enuc_t_holder_3[:] = self.t_elecnuc
                        enuc_l_holder_3[:] = self.l_elecnuc

                        nuc_t_holder_3 = self.t_nuc
                        nuc_l_holder_3 = self.l_nuc


                        numpy.save('bifluoride_anion.cc-pvtz.cneomp2.t_elec.npy'+'.npy', self.t_elec, allow_pickle=True, fix_imports=True)
                        numpy.save('bifluoride_anion.cc-pvtz.cneomp2.l_elec.npy'+'.npy', self.l_elec, allow_pickle=True, fix_imports=True)

                        for i in range(len(self.con.mol.nuc)):

                            numpy.save('bifluoride_anion.cc-pvtz.cneomp2.t_elecnuc.npy'+'.'+str(i)+'.npy', enuc_t_holder_3[i], allow_pickle=True, fix_imports=True)
                            numpy.save('bifluoride_anion.cc-pvtz.cneomp2.l_elecnuc.npy'+'.'+str(i)+'.npy', enuc_l_holder_3[i], allow_pickle=True, fix_imports=True)

                        for i in range(len(self.con.mol.nuc)):
                            for j in range(len(self.con.mol.nuc)):

                                numpy.save('bifluoride_anion.cc-pvtz.cneomp2.t_nuc.npy'+'.'+str(i)+str(j)+'.npy', nuc_t_holder_3[i][j], allow_pickle=True, fix_imports=True)
                                numpy.save('bifluoride_anion.cc-pvtz.cneomp2.l_nuc.npy'+'.'+str(i)+str(j)+'.npy', nuc_l_holder_3[i][j], allow_pickle=True, fix_imports=True)

                    break
               
                else:

                    pass    

            else:

                if t < (max_iteration_cycles - 1):

                    print('Not converged, continuing to next iteration...')

                    print(asterisk)

                    pass

                else:

                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('WARNING!...WARNING!...WARNING!...WARNING!...WARNING!')
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('One or more sets of t-amplitudes has failed to converge!')
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('WARNING!...WARNING!...WARNING!...WARNING!...WARNING!')
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                    pass



        finish_timer_17 = timer()

        print('timer[17]: total time for the overall t-amplitude and energy convergence after all iterations/cycles: ', finish_timer_17 - start_timer_17)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        
        #-------------------------------------------------------------------------------        
        # [2.15] Calculate and compare HF density with final optimized MP2 density   
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #number_of_points = 301 
        #compare = one_dim_density_on_axis(self, number_of_points)
        #for i in range(len(self.con.mol.nuc)):
        #    print(numpy.asarray(compare[i]))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        finish_timer_2 = timer()

        print('timer[2]: total timing of the kernel function: ', finish_timer_2 - start_timer_2)

        #-------------------------------------------------------------------------------        
        # [2.16] Return variables and complete kernel function processes  
        #-------------------------------------------------------------------------------
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return self.base_energy, hyll_e, hyll_en, hyll_n, self.try_lagr

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == '__main__':


#    mol = neo.Mole()
#    mol.build(atom = '''H1 0 0 0.375; H2 0 0 -0.375''', basis = 'cc-pvtz', charge = 0, quantum_nuc = [0,1])

#    mol = neo.Mole()
#     mol.build(atom = '''C1 0 0 0; H2 0.6276 0.6276 0.6276; H3 0.6276 -0.6276 -0.6276; H4 -0.6276 0.6276 -0.6276; H5 -0.6276 -0.6276 0.6276''', basis = 'cc-pvdz', charge = 0, quantum_nuc = [0,1,2,3,4])

#    mol = neo.Mole() 
#    mol.build(atom = '''He1 0 0 0.388; H2 0 0 -0.388''', basis = 'sto-3g', charge = 1, quantum_nuc = [0,1])

#    mol = neo.Mole()
#    mol.build(atom = '''Li1 0 0 -1.05; Ne2 0 0 1.05''', basis = 'sto-3g', charge = 1, quantum_nuc = [0,1])

#    mol = neo.Mole()
#    mol.build(atom = '''C1 0 0 -0.564; O2 0 0 0.564''', basis = 'sto-3g', charge = 0, quantum_nuc = [0,1])


#    mol = neo.Mole()
#    mol.build(atom = '''H1 0 0 -0.4585; F2 0 0 0.4585''', basis = 'cc-pvdz', charge = 0, quantum_nuc = [0,1])

#    mol = neo.Mole()
#    mol.build(atom = '''H1 0 0 -1.064; C2 0 0 0; N3 0 0 1.156''', basis = 'cc-pvdz', charge = 0, quantum_nuc = [0,1,2])

#    mol = neo.Mole()
#    mol.build(atom = '''H1 0.0000 0.5011 0.0000; H2 0.4340 -0.2505 0.0000; H3 -0.4340 -0.2505 0.0000''', basis = 'cc-pvdz', charge = 1, quantum_nuc = [0,1,2])

#    mol = neo.Mole() 
#    mol.build(atom = '''C1 0 0 -1.277; C2 0 0 0; C3 0 0 1.277''', basis = 'sto-3g', charge = 0, quantum_nuc = [0,1,2])

#    mol = neo.Mole()
#    mol.build(atom = '''F1 0 0 -1.135; H2 0 0 0; F3 0 0 1.135''', basis = 'sto-3g', charge = -1, quantum_nuc = [0,1,2])    

#    reg = neo.HF(mol)
#    con = neo.cdft.CDFT(mol)

#    energy_con = con.scf()
#
#    final_e, mp2_iter_n, mp2_iter_e, mp2_iter_en, lagrange_m = cNEOMP2(reg, con, energy_con).kernel()
#
#    print(line)
#    print('Converged c-NEOHF-SCF Energy = ', energy_con)
#    print(line)
#    print('iterative e_mp2 = ', mp2_iter_e)
#    print(line)
#    print('iterative en_mp2 = ', mp2_iter_en)
#    print(line)
#    print('iterative n_mp2 = ', mp2_iter_n)
#    print(line)
#    print('total iterative neo-mp2 = ', energy_con + mp2_iter_e + mp2_iter_en + mp2_iter_n)
#
#    print('total from Lagrangian = ', final_e)
#    print('Lagrange multipliers = ', lagrange_m)


#---------------------------------------------------------------------------------------------------------------
# This section is to be used for calculations requiring optimization or generating potential energy surfaces.
#---------------------------------------------------------------------------------------------------------------

#------------------------------
# Diatomics
#------------------------------
#    noatoms = 2
#    dimensions = 3
#    coordinate_array = numpy.zeros((noatoms,dimensions))

#    dist = (stdist/2)

#    coordinate_array[0,0] = 0.0000
#    coordinate_array[0,1] = 0.0000
#    coordinate_array[0,2] = -dist
#    coordinate_array[1,0] = 0.0000
#    coordinate_array[1,1] = 0.0000
#    coordinate_array[1,2] = dist

#    input_coordinates = coordinate_array.flatten()


#------------------------------
# Triatomics
#------------------------------

    noatoms = 3 
    dimensions = 3

    coordinate_array = numpy.loadtxt('bifluoride_anion-coords.txt')

    input_coordinates = coordinate_array.flatten()


    def calc_distances(ivec, jvec):

        quant_one = (ivec[0] - jvec[0])**2
        quant_two = (ivec[1] - jvec[1])**2
        quant_three = (ivec[2] - jvec[2])**2

        return math.sqrt(quant_one + quant_two + quant_three)


    def calc_angles(ivec, jvec, kvec):

        ijvec = -(jvec - ivec)
        jkvec = -(kvec - jvec)
        ijvec = ijvec / numpy.linalg.norm(ijvec)
        jkvec = jkvec / numpy.linalg.norm(jkvec)

        return numpy.degrees(numpy.arccos(numpy.dot(ijvec, jkvec)))


    #------------------------------------------------
    # Defining the Energy Function for Optimization
    #------------------------------------------------

    def energy(coordinates):

        # Defining variables:
        dimensions = 3
        noatoms = 3 
        # Building the mole object:

        # [Note:] First atom input into the mole object for a triatomic molecule must be the central atom because of the way that
        #         the distances are being computed.

        mol = neo.Mole()
        mol.build(atom=[['H1',coordinates[0:dimensions]],
                        ['F2',coordinates[dimensions:2*dimensions]],
                        ['F3',coordinates[2*dimensions:3*dimensions]]], basis='cc-pvtz', charge=-1, quantum_nuc = [0,1,2])



        # For Troubleshooting:
        # Printing Cartesian Coordinates:
        #        print(coordinates)
        #        print(mol.atom_coords(unit='ANG'))
        #        print('The variable equals:', mol.nuc_num)

        # Calculating the Bond Length to Graph later:
        # These lines can be commented out when running the PESs, but are necessary for the geometry optimization data.

        R = mol.atom_coords(unit='ANG')

        for i in range(noatoms):
            for j in range(noatoms):
                
                if (i!=j):
                    print('distance', i, j, calc_distances(R[i], R[j]))
                else:
                    pass

        for i in range(noatoms):
            for j in range(noatoms):
                for k in range(noatoms):

                     if (i==j) or (i==k) or (j==k):
                         pass
                     else:
                         print('angle', i, j, k, calc_angles(R[i], R[j], R[k]))

        # Defining Objects to pass into the Class:
        reg = neo.HF(mol)
        con = neo.cdft.CDFT(mol)
#        reg.verbose = 10
#        con.verbose = 10
        reg.verbose = False
        con.verbose = False
        # Calculating energies without MP2 correction:
        energy_con = con.scf()

        restart=False

        # Making an object/instance of the Class and calculating MP2 correction terms:
        mp2 = cNEOMP2(reg, con, energy_con, restart)
        final_e, mp2_iter_e, mp2_iter_en, mp2_iter_n, lagrange = mp2.kernel()


        # Definition of the Total Energy:
        totale = energy_con+mp2_iter_e+mp2_iter_en+mp2_iter_n

        print('outputline', con.converged, totale, mp2_iter_e, mp2_iter_en, mp2_iter_n)
        print('Lagrange multipliers', lagrange)
        print('------------------------')
        print(R)
        print('------------------------')
        print('basis', mol.basis)
        print('charge', mol.charge)
        print('quantum nuclei', mol.quantum_nuc)
        print('method cneomp2')

        # Return Total Energy:
        return totale



#---------------------------------------------------------------
# Manual Optimization
#---------------------------------------------------------------

#    print(energy(input_coordinates))

#---------------------------------------------------------------
# Optimization Methods: [1] BFGS or [2] Nelder-Mead
#---------------------------------------------------------------
# Test Optimization with Broyden-Fletcher-Goldfarb-Shanno Method:
    optimal_coordinates = scipy.optimize.minimize(energy, coordinate_array, method='BFGS', jac=None, tol=1e-6, callback=None, options={'gtol': 1e-06, 'norm': inf, 'eps': 1e-06, 'maxiter': None, 'disp': True, 'return_all': False, 'finite_diff_rel_step': None})
    print(optimal_coordinates)

# Test Optimization with Nelder-Mead Method:
#    optimal_coordinates = scipy.optimize.minimize(energy, coordinate_array, method='Nelder-Mead', bounds=None, tol=1e-6, callback=None, options={'disp': True, 'xatol': 0.000001, 'fatol': 0.000001})
#    print(optimal_coordinates)

