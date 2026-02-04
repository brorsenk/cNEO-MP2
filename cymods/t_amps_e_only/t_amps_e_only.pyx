cimport cython
import numpy
import pyscf
cimport numpy
from pyscf import ao2mo


@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
def t_amps_e_only(self, double[:,:,:,:] t_electronic, double[:,:,:,:] two_int_e_t, double[:,:] ncF_eMO):

    print('calling t_amps_e_only from cymods')
    #-------------------------------------------------------
    # Build items for t-amplitude electronic terms
    #-------------------------------------------------------
    cdef int e_nocc = self.e_nocc
    cdef int e_nvir = self.e_nvir
    cdef int e_tot = self.e_tot
    
    moe_occupied = ncF_eMO[:e_nocc,:e_nocc]
    moe_virtual = ncF_eMO[e_nocc:,e_nocc:]

    # Empty orbital energy tensor:
    cdef double[:,:,:,:] e = numpy.zeros((e_nocc, e_nvir, e_nocc, e_nvir))

    cdef int i, a, j, b, c, k

    # Building the orbital energy difference tensor.
    for i in range(e_nocc):
        for a in range(e_nvir):
            for j in range(e_nocc):
                for b in range(e_nvir):
                    e[i,a,j,b] = (moe_occupied[i,i] + moe_occupied[j,j] - moe_virtual[a,a] - moe_virtual[b,b])


    # Initializing T(abij) {abab} tensors to perform iterations with.
    cdef double[:,:,:,:] T_old_e = t_electronic
    cdef double[:,:,:,:] T_new_e = numpy.zeros((e_nocc, e_nvir, e_nocc, e_nvir))

    # Initializing variables for electronic t terms.
    cdef double coulomb_term = 0
    cdef double sum_over_c_not_a = 0
    cdef double sum_over_c_not_b = 0
    cdef double sum_over_k_not_i = 0
    cdef double sum_over_k_not_j = 0

#    eeee
#   e    e  #------------------------------------------------
#   e eee   # Beginning of electronic t-amplitude cycle:
#   e     e #------------------------------------------------ 
#    eeee

    #----------------------------------------------------------------
    # Outer Summation (indexed by i,j occupied, a,b virtual):
    #----------------------------------------------------------------
    for i in range(e_nocc):
        for a in range(e_nvir):
            for j in range(e_nocc):
                for b in range (e_nvir):

                    #------------------------------------------------------------------------------
                    # Set/reset inner summation terms equal to zero to begin the iteration:
                    #------------------------------------------------------------------------------
                    sum_over_c_not_a = 0
                    sum_over_c_not_b = 0
                    sum_over_k_not_i = 0
                    sum_over_k_not_j = 0

                    #------------------------------------------------------------------------------
                    # Constant Term: 
                    #------------------------------------------------------------------------------
                    coulomb_term = (two_int_e_t[i,a,j,b])

                    #------------------------------------------------------------------------------                              
                    # Inner Virtual Summations (indexed by c):
                    #------------------------------------------------------------------------------

                    for c in range(e_nvir):
                        if (c != a):
                            sum_over_c_not_a += (ncF_eMO[a+e_nocc,c+e_nocc]*T_old_e[i,c,j,b])
                        if (c != b):
                            sum_over_c_not_b += (ncF_eMO[b+e_nocc,c+e_nocc]*T_old_e[i,a,j,c])

                    #------------------------------------------------------------------------------
                    # Inner Occupied Summations (indexed by k):
                    #------------------------------------------------------------------------------

                    for k in range(e_nocc):
                        if (k != i):
                            sum_over_k_not_i += (ncF_eMO[k,i]*T_old_e[k,a,j,b])
                        if (k != j):
                            sum_over_k_not_j += (ncF_eMO[k,j]*T_old_e[i,a,k,b])


                    #------------------------------------------------------------------------------
                    # Build elements of T(abij){abab}:
                    #------------------------------------------------------------------------------

                    T_new_e[i,a,j,b] += ((coulomb_term + sum_over_c_not_a + sum_over_c_not_b - sum_over_k_not_i - sum_over_k_not_j)/(((e[i,a,j,b]))))



    return T_new_e
