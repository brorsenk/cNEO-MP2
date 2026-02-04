import tracemalloc
import numpy
import pyscf
import math

def RMSD_e(self, t_old, t_new):

    print('calling RMSD_e from pymods')
    print(tracemalloc.get_traced_memory(), flush=True)
    e_nocc = self.e_nocc
    e_nvir = self.e_nvir
    e_tot = self.e_tot

    sum_pieces = 0

    for i in range(e_nocc):
        for a in range(e_nvir):
            for j in range(e_nocc):
                for b in range(e_nvir):

                    sum_pieces += ((t_new[i,a,j,b]-t_old[i,a,j,b])**2)
    print(tracemalloc.get_traced_memory(), flush=True)
    RMSD_t = math.sqrt(sum_pieces)
    return RMSD_t

def RMSD_en(self, t_old, t_new):

    print('calling RMSD_en from pymods')
    print(tracemalloc.get_traced_memory(), flush=True)
    nuclei = len(self.con.mol.nuc)

    e_nocc = self.e_nocc
    e_nvir = self.e_nvir
    e_tot = self.e_tot

    sum_pieces = 0
    RMSD_t = []
    total_RMSD = 0

    for i in range(nuclei):
        RMSD_t.append([None])

    for j in range(nuclei):
 
        sum_pieces = 0

        p_nocc = self.con.mf_nuc[j].mo_coeff[:,self.con.mf_nuc[j].mo_occ>0].shape[1]
        p_tot  = self.con.mf_nuc[j].mo_coeff[0,:].shape[0]
        p_nvir = p_tot - p_nocc


        for i in range(e_nocc):
            for a in range(e_nvir):
                for I in range(p_nocc):
                    for A in range(p_nvir):

                        sum_pieces += ((t_new[j][i,a,I,A]-t_old[j][i,a,I,A])**2)
        print(tracemalloc.get_traced_memory(), flush=True)
        RMSD_t[j] = math.sqrt(sum_pieces)
        total_RMSD += RMSD_t[j]

    return total_RMSD

def RMSD_n(self, t_old, t_new):
    
    print('calling RMSD_n from pymods')
    print(tracemalloc.get_traced_memory(), flush=True)
    nuclei = len(self.con.mol.nuc)

    RMSD_t = numpy.empty((nuclei,nuclei), dtype=object)
    for i in range(nuclei):
        for j in range(nuclei):
            RMSD_t[i][j] = [None]


    sum_pieces = 0
    
    for i in range(nuclei):

        if nuclei==1:
            pass

        else:

            for j in range(nuclei):

                sum_pieces = 0

                p_nocc_i = self.con.mf_nuc[i].mo_coeff[:,self.con.mf_nuc[i].mo_occ>0].shape[1]
                p_tot_i  = self.con.mf_nuc[i].mo_coeff[0,:].shape[0]
                p_nvir_i = p_tot_i - p_nocc_i
                p_nocc_j = self.con.mf_nuc[j].mo_coeff[:,self.con.mf_nuc[j].mo_occ>0].shape[1]
                p_tot_j  = self.con.mf_nuc[j].mo_coeff[0,:].shape[0]
                p_nvir_j = p_tot_j - p_nocc_j
                
                for I in range(p_nocc_i):
                    for A in range(p_nvir_i):
                        for J in range(p_nocc_j):
                            for B in range(p_nvir_j):

                                sum_pieces += ((t_new[i][j][I,A,J,B]-t_old[i][j][I,A,J,B])**2)

                RMSD_t[i][j] =  math.sqrt(sum_pieces)


    print(tracemalloc.get_traced_memory(), flush=True)
    total_RMSD = 0

    for i in range(nuclei):
        for j in range(i):

            total_RMSD += RMSD_t[i][j]

    return total_RMSD
