cimport cython
import numpy
import pyscf
cimport numpy
from pyscf import neo
#from cython.parallel import prange

#-------------------------------------------------------------------------------        
# [2.8] Nuclear single-particle MP2 density matrix functions 
#-------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
def mp2_density_one(self, int i_idx, T_n, L_n, T_en, L_en):

    print('calling mp2_density_one function from cymods...')

    con = self.con

    cdef int nuclei = len(self.con.mol.nuc)

    cdef int e_nocc = self.e_nocc
    cdef int e_tot = self.e_tot
    cdef int e_nvir = self.e_nvir

    #----------------------------------------------------------------------------  
    # [Notes:] 
    #----------------------------------------------------------------------------
    # Perhaps putting these here will make it work?
    # It worked, but the first matrix is all zero. I still need to find a way to
    # index and store them correctly. [Complete] 
    # The occupied and virtual values must be defined in this part of the loop
    # in order to be able to initialize empty density matrices of the appropriate
    # dimensions. [Complete]
    #----------------------------------------------------------------------------

    cdef int j_idx
    cdef int p_nocc_i = int(self.num_ovt[i_idx][0,0])
    cdef int p_nvir_i  = int(self.num_ovt[i_idx][0,1])
    cdef int p_tot_i = int(self.num_ovt[i_idx][0,2])
    cdef int p_nocc_j
    cdef int p_nvir_j
    cdef int p_tot_j 
    
    cdef double Sum_over_iIa = 0
    cdef double Sum_over_aAi = 0
    cdef double Sum_over_C = 0
    cdef double Sum_over_K = 0
    cdef double virt_en_contr = 0
    cdef double occu_en_contr = 0

    cdef int A, B, I, J, ie, ae, C, K 

    #----------------------------------------------------------------------------
    # Initialization of the empty single particle density matrices.
    #----------------------------------------------------------------------------
    cdef double[:,:] gamma_vir = numpy.zeros((p_nvir_i,p_nvir_i))
    cdef double[:,:] gamma_occ = numpy.zeros((p_nocc_i,p_nocc_i))
#    cdef double[:,:,:,:] T_en[i_idx]
#    cdef double[:,:,:,:] L_en[i_idx]

    for j_idx in range(nuclei):
        T_n[i_idx][j_idx]: double[:,:,:,:]
        L_n[i_idx][j_idx]: double[:,:,:,:]
        T_en[j_idx]: double[:,:,:,:]
        L_en[j_idx]: double[:,:,:,:]
    #-------------------------------------------------------------------------------------
    # [Notes:]
    #-------------------------------------------------------------------------------------
    # I think this new if/else statement and extra loops may solve the problem! I hope!
    # [Note:] It seems that this did indeed solve the problem.
    #-------------------------------------------------------------------------------------


    if (nuclei == 1):

#        T_lambda_en_i = numpy.swapaxes(T_en[i_idx].T, 0,2)
#        T_lambda_en_i = numpy.swapaxes(T_lambda_en_i, 1,3)

        #--------------------------------------------------------------------------
        # Building of the single particle virtual density matrix:
        #--------------------------------------------------------------------------
        # Variables which index the matrix:                 
        #------------------------------------
        for A in range(p_nvir_i):
            for B in range(p_nvir_i):

                # This is a contraction over the indices i, I, and a to produce gamma_vir[A,B]

                Sum_over_iIa = 0

                #------------------------------------
                # Variables which are summed over:
                #------------------------------------
                for I in range(p_nocc_i):
                    for ie in range(e_nocc):
                        for ae in range(e_nvir):

                            Sum_over_iIa += ((L_en[i_idx][ae,ie,A,I])*(T_en[i_idx][ie,ae,I,B]))

                gamma_vir[A,B] += Sum_over_iIa

        #--------------------------------------------------------------------------
        # Building of the single particle occupied density matrix:
        #--------------------------------------------------------------------------
        # Variables which index the matrix:
        #------------------------------------
        for I in range(p_nocc_i):
            for J in range(p_nocc_i):

                # This is a contraction over the indices a, A, and i to produce gamma_occ[I,J]

                Sum_over_aAi = 0


                #------------------------------------
                # Variables which are summed over:
                #------------------------------------
                for A in range(p_nvir_i):
                    for ie in range(e_nocc):
                        for ae in range(e_nvir):

                            Sum_over_aAi +=  -((L_en[i_idx][ae,ie,A,J])*(T_en[i_idx][ie,ae,I,A]))

                gamma_occ[I,J] += Sum_over_aAi

    else:

    #-------------------------------------------------------------------------------------------------------------------
    # This begins the loop over all of the other nuclei if more than one nucleus is being treated quantum mechanically.
    #-------------------------------------------------------------------------------------------------------------------
        for j_idx in range(nuclei):

                if (i_idx == j_idx):
                    pass


                else:
                    p_nocc_j = int(self.num_ovt[j_idx][0,0])
                    p_nvir_j = int(self.num_ovt[j_idx][0,1])
                    p_tot_j = int(self.num_ovt[j_idx][0,2])


                  #  T_lambda_n = numpy.swapaxes(T_n[i_idx][j_idx].T, 0,2)
                  #  T_lambda_n = numpy.swapaxes(T_lambda_n, 1,3)
                  #  T_lambda_en_i = numpy.swapaxes(T_en[i_idx].T, 0,2)
                  #  T_lambda_en_i = numpy.swapaxes(T_lambda_en_i, 1,3)
                  #  T_lambda_en_j = numpy.swapaxes(T_en[j_idx].T, 0,2)
                  #  T_lambda_en_j = numpy.swapaxes(T_lambda_en_j, 1,3)

                    #-----------------------------------------------------------
                    # Building of the single particle virtual density matrix:
                    #-----------------------------------------------------------
                    # Variables which index the matrix:                 
                    #------------------------------------
                    for A in range(p_nvir_i):
                        for B in range(p_nvir_i):

                            Sum_over_C = 0
                            virt_en_contr = 0

                            #------------------------------------
                            # Variables which are summed over:
                            #------------------------------------
                            for I in range(p_nocc_i):
                                for J in range(p_nocc_j):
                                    for C in range(p_nvir_j):
                                    
                                        Sum_over_C += ((L_n[i_idx][j_idx][A,I,C,J])*(T_n[i_idx][j_idx][I,B,J,C]))

                                    for ie in range(e_nocc):
                                        for ae in range(e_nvir):

                                            virt_en_contr += ((L_en[i_idx][ae,ie,A,I])*(T_en[i_idx][ie,ae,I,B]))


                            gamma_vir[A,B] += Sum_over_C + virt_en_contr


                    #-----------------------------------------------------------
                    # Building of the single particle occupied density matrix:
                    #-----------------------------------------------------------
                    # Variables which index the matrix:
                    #------------------------------------
                    for I in range(p_nocc_i):
                        for J in range(p_nocc_i):

                            Sum_over_K = 0
                            occu_en_contr = 0

                            #------------------------------------
                            # Variables which are summed over:
                            #------------------------------------
                            for A in range(p_nvir_i):
                                for B in range(p_nvir_j):
                                    for K in range(p_nocc_j):

                                        Sum_over_K += ((-1)*(L_n[i_idx][j_idx][A,J,B,K])*(T_n[i_idx][j_idx][I,A,K,B]))
                                    
                                    for ie in range(e_nocc):
                                        for ae in range(e_nvir):

                                            occu_en_contr += ((-1)*((L_en[i_idx][ae,ie,A,J])*(T_en[i_idx][ie,ae,I,A])))


                            gamma_occ[I,J] += Sum_over_K + occu_en_contr



    cdef double[:,:] gamma_total = numpy.zeros((p_tot_i, p_tot_i))
    gamma_total[:p_nocc_i, :p_nocc_i] = gamma_occ
    gamma_total[p_nocc_i:, p_nocc_i:] = gamma_vir

    return gamma_total

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

