import tracemalloc
import numpy
import pyscf
from pyscf import neo, ao2mo

#-------------------------------------------------------------------------------        
# [2.11] Hylleraas nuclear energy function 
#-------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def Hylleraas_energy_n(self, t_amps_n, l_amps_n, two_int_n_t, two_int_n_l, fock_n):

    print('calling Hylleraas_energy_n from pymods', flush=True)
    print(tracemalloc.get_traced_memory(), flush=True)

    reg = self.reg
    con = self.con

    nuclei = len(self.con.mol.nuc)

    #--------------------------------------------------------------------------
    # Nuclear MP2 energy correction:
    #--------------------------------------------------------------------------

    # Initialize variables used to calculate nuclear energy correction.
    gt_sum_n_total = 0
    gl_sum_n_total = 0
    Ci_sum_n_total = 0
    Cj_sum_n_total = 0
    Ki_sum_n_total = 0
    Kj_sum_n_total = 0
    iterative_mp2_nuclear_new = 0

    zero = 0

    # Build empty nested lists to store energy corrections indexed by nuclei i and j.

    iterative_mp2_n = numpy.empty((nuclei,nuclei), dtype=object)
    for i in range(nuclei):
        for j in range(nuclei):
            iterative_mp2_n[i][j] = zero

    #--------------------------------------------------------------------------
    # Loop over nuclei to construct nuclear energy correction for each pair:
    #--------------------------------------------------------------------------   
    for i in range(nuclei):

        for j in range(nuclei):

            if (i == j):

                pass

            else:

                p_nocc_i = self.con.mf_nuc[i].mo_coeff[:,self.con.mf_nuc[i].mo_occ>0].shape[1]
                p_tot_i  = self.con.mf_nuc[i].mo_coeff[0,:].shape[0]
                p_nvir_i = p_tot_i - p_nocc_i
                p_nocc_j = self.con.mf_nuc[j].mo_coeff[:,self.con.mf_nuc[j].mo_occ>0].shape[1]
                p_tot_j  = self.con.mf_nuc[j].mo_coeff[0,:].shape[0]
                p_nvir_j = p_tot_j - p_nocc_j

                #------------------------------------------------------------------------
                # Set/reset the energy value equal to zero: 
                #------------------------------------------------------------------------
                gt_sum_n_total = 0
                gl_sum_n_total = 0
                C_sum_n_total = 0
                K_sum_n_total = 0
                Ci_sum_n_total = 0
                Cj_sum_n_total = 0
                Ki_sum_n_total = 0
                Kj_sum_n_total = 0
                iterative_mp2_nuclear_new = 0

                #-----------------------------------------------------------------
                # Outer Summation (indexed by I,J occupied, A,B virtual)
                #-----------------------------------------------------------------
                for I in range(p_nocc_i):
                    for A in range(p_nvir_i):
                        for J in range(p_nocc_j):
                            for B in range(p_nvir_j):

                                print(tracemalloc.get_traced_memory(), flush=True)
                                #-----------------------------------------------------------------------
                                # Set/reset the inner summations equal to zero to begin the iterations:
                                #-----------------------------------------------------------------------

                                # Troubleshooting February 8th 2023 - perhaps the sums over Cj and Kj are
                                # double counting?
                                # Actually they might be necessary it seems since the ij interactions are
                                # possibly different than the ji interactions?

                                #-----------------------------------------------------------------------
                                # Construct products of g and t terms.
                                #-----------------------------------------------------------------------
                                gt_sum_n_total += two_int_n_l[i][j][A,I,B,J]*(t_amps_n[i][j][I,A,J,B])

                                #-----------------------------------------------------------------------
                                # Construct products of g and lambda terms.
                                #-----------------------------------------------------------------------
                                gl_sum_n_total += two_int_n_t[i][j][I,A,J,B]*(l_amps_n[i][j][A,I,B,J])

                                #-----------------------------------------------------------------------
                                # Inner summation over virtual orbitals for nucleus [i] (indexed by Ci):
                                #-----------------------------------------------------------------------                                    
                                for Ci in range(p_nvir_i):
                                    Ci_sum_n_total += ((fock_n[i][A+p_nocc_i,Ci+p_nocc_i])*(l_amps_n[i][j][A,I,B,J])*(t_amps_n[i][j][I,Ci,J,B]))

                                #-----------------------------------------------------------------------
                                # Inner summation over virtual orbitals for nucleus [j] (indexed by Cj):
                                #-----------------------------------------------------------------------
                                for Cj in range(p_nvir_j):
                                    Cj_sum_n_total += ((fock_n[j][B+p_nocc_j,Cj+p_nocc_j])*(l_amps_n[i][j][A,I,B,J])*(t_amps_n[i][j][I,A,J,Cj]))

                                #-----------------------------------------------------------------------
                                # Inner summation over occupied orbitals for nucleus [i] (indexed by Ki):
                                #-----------------------------------------------------------------------
                                for Ki in range(p_nocc_i):
                                    Ki_sum_n_total += ((fock_n[i][I,Ki])*(l_amps_n[i][j][A,Ki,B,J])*(t_amps_n[i][j][I,A,J,B]))

                                #-----------------------------------------------------------------------
                                # Inner summation over occupied orbitals for nucleus [j] (indexed by Kj):
                                #-----------------------------------------------------------------------
                                for Kj in range(p_nocc_j):
                                    Kj_sum_n_total += ((fock_n[j][J,Kj])*(l_amps_n[i][j][A,I,B,Kj])*(t_amps_n[i][j][I,A,J,B]))


                #-----------------------------------------------------------------------
                # Combine all terms into the total energy expression:
                #-----------------------------------------------------------------------
                iterative_mp2_nuclear_new += ((gt_sum_n_total) + (gl_sum_n_total) + (Ci_sum_n_total+Cj_sum_n_total) - (Ki_sum_n_total+Kj_sum_n_total))

            #-------------------------------------------------------------------------------
            # Store final values for each correction, indexed by nuclear pair [i][j]:
            #-------------------------------------------------------------------------------
            iterative_mp2_n[i][j] = iterative_mp2_nuclear_new

    print(tracemalloc.get_traced_memory(), flush=True)
    total_n = 0
    for i in range(nuclei):
        for j in range(i):
            total_n += iterative_mp2_n[i][j]

    return total_n

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print(tracemalloc.get_traced_memory(), flush=True)
