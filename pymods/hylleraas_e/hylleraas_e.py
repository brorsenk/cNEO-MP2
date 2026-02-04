import tracemalloc
import numpy
import pyscf
from pyscf import ao2mo

#-------------------------------------------------------------------------------        
# [2.9] Hylleraas electronic energy function 
#-------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Note: I need to make the lambda amplitudes something that gets passed into the function as well!

def Hylleraas_energy_e(self, t_amps_e, l_amps_e, two_int_e_t, two_int_e_l, fock_e):

    print('calling Hylleraas_energy_e from pymods', flush=True)
    print(tracemalloc.get_traced_memory(), flush=True)
    #--------------------------------------------------------------------------
    # Electronic MP2 energy correction:
    #--------------------------------------------------------------------------
    reg = self.reg
    con = self.con
            
    e_nocc = self.e_nocc
    e_nvir = self.e_nvir
    e_tot = self.e_tot
            
    # Initializing variables for electronic terms.        
    gt_sum_total = 0
    gl_sum_total = 0 
    c_sum_total = 0
    k_sum_total = 0
    iterative_mp2_electronic_new = 0

    two = 2
    half = 0.5
    quarter = 0.25  

    # On spin-indexing:
    # [Note:] Remember each of these sums will be written in terms of the {abab} t(i,j,a,b) 
    #         expressions because that is what the t(abij) computed above are solved in terms of.

    #------------------------------------------------------------------------
    # Beginning of the iterations for the electronic MP2 energy correction:
    #------------------------------------------------------------------------
    print(t_amps_e.shape, flush=True)
    print(l_amps_e.shape, flush=True)
    print(two_int_e_t.shape, flush=True)
    print(two_int_e_l.shape, flush=True)
    #------------------------------------------------------------------------
    # Set/reset the energy value equal to zero: 
    #------------------------------------------------------------------------
    iterative_mp2_electronic_new = 0

    #----------------------------------------------------------------
    # Outer Summation (indexed by i,j occupied, a,b virtual):
    #----------------------------------------------------------------
    for i in range(e_nocc):
        for a in range(e_nvir):
            for j in range(e_nocc):
                for b in range(e_nvir):
                    print(tracemalloc.get_traced_memory(), flush=True)
                    #-------------------------------------------------------------------
                    # Construct products of g and t terms.
                    #-------------------------------------------------------------------
                    gt_sum_total += (two*((two_int_e_l[a,i,b,j] - two_int_e_l[a,j,b,i])*(t_amps_e[i,a,j,b] - t_amps_e[i,b,j,a])))
                    gt_sum_total += (two*((two_int_e_l[a,i,b,j])*(t_amps_e[i,a,j,b])))
                    gt_sum_total += (two*((-two_int_e_l[a,j,b,i])*(-t_amps_e[i,b,j,a])))

                    #-------------------------------------------------------------------
                    # Construct products of g and l terms.
                    #-------------------------------------------------------------------                     
                    gl_sum_total += (two*((two_int_e_t[i,a,j,b] - two_int_e_t[i,b,j,a])*(l_amps_e[a,i,b,j] - l_amps_e[a,j,b,i])))
                    gl_sum_total += ((two*two_int_e_t[i,a,j,b])*(l_amps_e[a,i,b,j]))
                    gl_sum_total += (two*((-two_int_e_t[i,b,j,a])*(-l_amps_e[a,j,b,i])))

                    #------------------------------------------------------------------
                    # Inner summation over occupied orbitals (indexed by c):
                    #------------------------------------------------------------------
                    for c in range(e_nvir):
                        c_sum_total += (two*((fock_e[a+e_nocc,c+e_nocc])*(l_amps_e[a,i,b,j] - l_amps_e[a,j,b,i])*(t_amps_e[i,c,j,b] - t_amps_e[i,b,j,c])))
                        c_sum_total += (two*((fock_e[a+e_nocc,c+e_nocc])*(l_amps_e[a,i,b,j])*(t_amps_e[i,c,j,b])))
                        c_sum_total += (two*((fock_e[a+e_nocc,c+e_nocc])*(-l_amps_e[a,j,b,i])*(-t_amps_e[i,b,j,c])))

                    #------------------------------------------------------------------
                    # Inner summation over virtual orbitals (indexed by k):
                    #------------------------------------------------------------------
                    for k in range(e_nocc):
                        k_sum_total += (two*((fock_e[i,k])*(l_amps_e[a,k,b,j]-l_amps_e[a,j,b,k])*(t_amps_e[i,a,j,b] - t_amps_e[i,b,j,a])))
                        k_sum_total += (two*((fock_e[i,k])*(l_amps_e[a,k,b,j])*(t_amps_e[i,a,j,b])))
                        k_sum_total += (two*((fock_e[i,k])*(-l_amps_e[a,j,b,k])*(-t_amps_e[i,b,j,a])))

    print(tracemalloc.get_traced_memory(), flush=True)
    #-----------------------------------------------------------------
    # Combine all terms into the total energy expression:
    #-----------------------------------------------------------------
    iterative_mp2_electronic_new = (((half)*c_sum_total) - ((half)*k_sum_total) + ((quarter)*gt_sum_total) + ((quarter)*gl_sum_total))

    total_e = 0
    total_e = iterative_mp2_electronic_new

    return total_e

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print(tracemalloc.get_traced_memory(), flush=True)

