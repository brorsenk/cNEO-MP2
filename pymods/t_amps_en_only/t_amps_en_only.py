import tracemalloc
import numpy
import pyscf
from pyscf import neo, ao2mo

#-------------------------------------------------------------------------------        
# [2.6] Electronic-nuclear t-amplitude function 
#-------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
def t_amps_en_only(self, j_idx, lagr_multipliers, t_electronic_nuclear, l_electronic_nuclear, two_int_en_t, ncF_eMO, ncF_nMO_j, integrals_r_j):
    '''
    Note that the lagr_multipliers, t_electronic_nuclear, integrals, and ncF_nMO should all be indexed by the nucleus when being fed in to the function.
    '''

    print('calling t_amps_en_only from pymods...') 
    print(tracemalloc.get_traced_memory(), flush=True)
    #-----------------------------------------------------------------
    # Build items for t-amplitude electronic-nuclear terms
    #----------------------------------------------------------------
                    
    # Initializing variables for electronic-nuclear t terms.
    coulomb_term_en = 0
    sum_over_c_not_a_en = 0
    sum_over_C_not_A_en = 0
    sum_over_k_not_i_en = 0
    sum_over_K_not_I_en = 0

    #T_en = [None] 
    #e_en = [None]
    #T_old_en = [None] 
    #T_new_en = [None]
                    
    #------------------------------------------------
    # Iterate the entire process over each nuclei:
    #------------------------------------------------

    p_nocc = int(self.num_ovt[j_idx][0,0])
    p_nvir = int(self.num_ovt[j_idx][0,1])
    p_tot = int(self.num_ovt[j_idx][0,2])

    e_nocc = self.e_nocc
    e_nvir = self.e_nvir
    e_tot = self.e_tot

    # Defining electronic and nuclear occupied and virtual orbital energies to build the orbital energy difference tensor.
    moe_nj_occupied = ncF_nMO_j[:p_nocc,:p_nocc]
    moe_nj_virtual = ncF_nMO_j[p_nocc:,p_nocc:]
    moe_occupied = ncF_eMO[:e_nocc,:e_nocc]
    moe_virtual = ncF_eMO[e_nocc:,e_nocc:]


    # Initialization and building of the orbital energy difference tensor.
    e_en = numpy.zeros((e_nocc, e_nvir, p_nocc, p_nvir))
    for i in range(e_nocc):
        for a in range(e_nvir):
            for I in range(p_nocc):
                for A in range(p_nvir):
                    e_en[i,a,I,A] = (moe_occupied[i,i] + moe_nj_occupied[I,I] - moe_virtual[a,a] - moe_nj_virtual[A,A])

    # Initializing T(abij) {abab} tensors to perform iterations with.
    T_old_en = t_electronic_nuclear
    T_new_en = numpy.zeros((e_nocc, e_nvir, p_nocc, p_nvir))
    L_old_en = l_electronic_nuclear


# eee 
#eeeee      #------------------------------------------------
#e    n nnn # Beginning of elecronic-nuclear t-amplitude sub-cycle:
# eee nn  n #------------------------------------------------ 
#     n   n

    # Two particle integral tensor:
#    eri_ep = neo.ao2mo.ep_ovov(self.con, self.con, j_idx)
#    TPIMO_en = eri_ep.reshape(e_nocc, e_nvir, p_nocc, p_nvir)

    virtual_gradient_term_en = numpy.zeros((3,))
    occupied_gradient_term_en = numpy.zeros((3,))

    # I NEED TO FIND OUT WHETHER THE LAMBDA AMPLITUDES ARE SIMPLY THE COMPLEX
    # CONJUGATES OF THE T-AMPLITUDES, OR WHETHER THE LAMBDA AMPLITUDES ARE THE
    # HERMITIAN CONJUGATES OF THE T-AMPLITUDES.

#    T_old_lambda_en_partial = numpy.swapaxes(T_old_en.T, 0,2)
#    T_old_lambda_en = numpy.swapaxes(T_old_lambda_en_partial, 1,3)

    print(tracemalloc.get_traced_memory(), flush=True)
    #----------------------------------------------------------------
    # Outer Summation (indexed by i,I occupied, a,A virtual)
    #----------------------------------------------------------------
    for i in range(e_nocc):
        for a in range(e_nvir):
            for I in range(p_nocc):
                for A in range (p_nvir):

                    #------------------------------------------------------------------------------
                    # Set/reset inner summation terms equal to zero to begin the iteration:
                    #------------------------------------------------------------------------------
                    sum_over_c_not_a_en = 0
                    sum_over_C_not_A_en = 0
                    sum_over_k_not_i_en = 0
                    sum_over_K_not_I_en = 0

                    virtual_gradient_term_en = numpy.zeros((3,))
                    occupied_gradient_term_en = numpy.zeros((3,))

                    #------------------------------------------------------------------------------
                    # Constant Term: 
                    #------------------------------------------------------------------------------
                    coulomb_term_en = ((-1)*two_int_en_t[i,a,I,A])

                    #------------------------------------------------------------------------------                              
                    # Inner Virtual Summations (indexed by c and C)
                    #------------------------------------------------------------------------------
                    for c in range(e_nvir):
                        if (c != a):
                            sum_over_c_not_a_en += (ncF_eMO[a+e_nocc,c+e_nocc]*(T_old_en[i,c,I,A]))
                    for C in range(p_nvir):
                        if (C != A):
                            sum_over_C_not_A_en += (ncF_nMO_j[A+p_nocc,C+p_nocc]*(T_old_en[i,a,I,C]))

                    #------------------------------------------------------------------------------
                    # Inner Occupied Summations (indexed by k and K)
                    #------------------------------------------------------------------------------
                    for k in range(e_nocc):
                        if (k != i):
                            sum_over_k_not_i_en += (ncF_eMO[k,i]*(T_old_en[k,a,I,A]))
                    for K in range(p_nocc):
                        if (K != I):
                            sum_over_K_not_I_en += (ncF_nMO_j[K,I]*(T_old_en[i,a,K,A]))

                    #------------------------------------------------------------------------------                              
                    # Virtual electronic-nuclear terms given by gradient of nuclear MP2 densities
                    #------------------------------------------------------------------------------
                    for R in range(p_nvir):
                        for T in range(p_nvir):
                            for x in range(3):

                                virtual_gradient_term_en[x] += (L_old_en[a,i,R,I] + T_old_en[i,a,I,T])*integrals_r_j[x,R,T]

                    #print(virtual_gradient_term_en)

                    lagr_dot_gradient_vir_en = numpy.dot(lagr_multipliers, virtual_gradient_term_en)
                    #print('vir_dot: ', lagr_dot_gradient_vir_en)

                    #------------------------------------------------------------------------------                              
                    # Occupied electronic-nuclear terms given by gradient of nuclear MP2 densities
                    #------------------------------------------------------------------------------
                    for P in range(p_nocc):
                        for Q in range(p_nocc):
                            for x in range(3):

                                occupied_gradient_term_en[x] += (L_old_en[a,i,A,P] + T_old_en[i,a,Q,A])*integrals_r_j[x,P,Q]

                    #print(occupied_gradient_term_en)

                    lagr_dot_gradient_occ_en = numpy.dot(lagr_multipliers, occupied_gradient_term_en)
                    #print('occ_dot: ', lagr_dot_gradient_occ_en)

                    #------------------------------------------------------------------------------                              
                    # Combined virtual and occupied MP2 nuclear density gradient contribution
                    #------------------------------------------------------------------------------
                    total_dot_product_of_lm_and_grad_en = (lagr_dot_gradient_vir_en) - (lagr_dot_gradient_occ_en)

                    #------------------------------------------------------------------------------
                    # Build elements of T(abij){abab}:
                    #------------------------------------------------------------------------------
                    T_new_en[i,a,I,A] = (((coulomb_term_en + sum_over_c_not_a_en + sum_over_C_not_A_en - sum_over_k_not_i_en - sum_over_K_not_I_en + total_dot_product_of_lm_and_grad_en)/(((e_en[i,a,I,A])))))

    print(tracemalloc.get_traced_memory(), flush=True)
    return T_new_en


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
