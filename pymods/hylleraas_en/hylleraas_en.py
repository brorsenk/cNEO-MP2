import tracemalloc
import numpy
import pyscf
from pyscf import neo, ao2mo

#-------------------------------------------------------------------------------        
# [2.10] Hylleraas electronic-nuclear energy function 
#-------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def Hylleraas_energy_en(self, t_amps_en, l_amps_en, two_int_en_t, two_int_en_l, fock_e, fock_n):

    print('calling Hylleraas_energy_en from pymods', flush=True)
    print(tracemalloc.get_traced_memory(), flush=True)
    #----------------------------------------------------
    # Electronic-Nuclear Iterative MP2 Energy Correction
    #----------------------------------------------------

    reg = self.reg
    con = self.con
 
    nuclei = len(self.con.mol.nuc)

    e_nocc = self.e_nocc
    e_nvir = self.e_nvir
    e_tot = self.e_tot

    # Initializing variables for electronic-nuclear terms.
    c_sum_en_total = 0
    k_sum_en_total = 0
    C_sum_en_total = 0
    K_sum_en_total = 0
    gt_sum_en_total = 0
    gl_sum_en_total = 0

    iterative_mp2_electronic_nuclear_list = []
    iterative_mp2_electronic_nuclear_new = 0

    two = 2

    #--------------------------------------------------------------------------
    # Loop over nuclei to construct each electronic-nuclear energy correction:
    #--------------------------------------------------------------------------    

    for j in range(nuclei):

        p_nocc = self.con.mf_nuc[j].mo_coeff[:,self.con.mf_nuc[j].mo_occ>0].shape[1]
        p_tot  = self.con.mf_nuc[j].mo_coeff[0,:].shape[0]
        p_nvir = p_tot - p_nocc

#        eri_ep = neo.ao2mo.ep_ovov(self.con, self.con, j)
#        TPIMO_en = eri_ep.reshape(e_nocc, e_nvir, p_nocc, p_nvir)

#        T_lambda_en = t_amps_en[j].T
#        TPIMO_lambda_en = TPIMO_en.T

#        T_lambda_en = numpy.swapaxes(t_amps_en[j].T, 0,2)
#        T_lambda_en = numpy.swapaxes(T_lambda_en, 1,3)
#        TPIMO_lambda_en = numpy.swapaxes(TPIMO_en.T, 0,2)
#        TPIMO_lambda_en = numpy.swapaxes(TPIMO_lambda_en, 1,3)

        c_sum_en_total = 0
        k_sum_en_total = 0
        C_sum_en_total = 0
        K_sum_en_total = 0
        gt_sum_en_total = 0
        gl_sum_en_total = 0

        #-----------------------------------------------------------------
        # Set/reset the energy value equal to zero: 
        #-----------------------------------------------------------------
        iterative_mp2_electronic_nuclear_new = 0

        #-----------------------------------------------------------------
        # Outer Summation (indexed by a,A occupied, i,I virtual)
        #-----------------------------------------------------------------
        for i in range(e_nocc):
            for a in range(e_nvir):
                for I in range(p_nocc):
                    for A in range (p_nvir):

                        print(tracemalloc.get_traced_memory(), flush=True)
                        #----------------------------------------------------------------------
                        # Construct products of g and t terms.
                        #----------------------------------------------------------------------
                        gt_sum_en_total += (two*((two_int_en_l[j][a,i,A,I])*(t_amps_en[j][i,a,I,A])))

                        #----------------------------------------------------------------------
                        # Construct products of g and lambda terms.
                        #----------------------------------------------------------------------
                        gl_sum_en_total += (two*((two_int_en_t[j][i,a,I,A])*(l_amps_en[j][a,i,A,I])))

                        #----------------------------------------------------------------------
                        # Inner summation over electronic occupied orbitals (indexed by c):
                        #----------------------------------------------------------------------
                        for c in range(e_nvir):
                            c_sum_en_total += (two*(((fock_e[a+e_nocc,c+e_nocc])*(l_amps_en[j][a,i,A,I])*(t_amps_en[j][i,c,I,A]))))

                        #----------------------------------------------------------------------
                        # Inner summation over electronic virtual orbitals (indexed by k):
                        #----------------------------------------------------------------------
                        for k in range(e_nocc):
                            k_sum_en_total += (two*((fock_e[i,k])*(l_amps_en[j][a,k,A,I])*(t_amps_en[j][i,a,I,A])))

                        #----------------------------------------------------------------------
                        # Inner summation over nuclear occupied orbitals (indexed by C):
                        #----------------------------------------------------------------------
                        for C in range(p_nvir):
                            C_sum_en_total += (two*((fock_n[j][A+p_nocc,C+p_nocc])*(l_amps_en[j][a,i,A,I])*(t_amps_en[j][i,a,I,C])))

                        #----------------------------------------------------------------------
                        # Inner summation over nuclear virtual orbitals (indexed by K):
                        #----------------------------------------------------------------------
                        for K in range(p_nocc):
                            K_sum_en_total += (two*((fock_n[j][I,K])*(l_amps_en[j][a,i,A,K])*(t_amps_en[j][i,a,I,A])))

        #-----------------------------------------------------------------
        # Combine all terms into the total energy expression:
        #-----------------------------------------------------------------
        iterative_mp2_electronic_nuclear_new = (((c_sum_en_total - k_sum_en_total + C_sum_en_total - K_sum_en_total - gt_sum_en_total - gl_sum_en_total)))

        #-----------------------------------------------------------------------------------------------
        # Append electronic nuclear energy corrections to a list which will be indexed by nucleus [j]:
        #-----------------------------------------------------------------------------------------------
        iterative_mp2_electronic_nuclear_list.append(iterative_mp2_electronic_nuclear_new)

    print(tracemalloc.get_traced_memory(), flush=True)
    total_en = 0
    for i in range(nuclei):
        total_en += iterative_mp2_electronic_nuclear_list[i]

    return total_en
    print(tracemalloc.get_traced_memory(), flush=True)
