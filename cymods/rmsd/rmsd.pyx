cimport cython
import numpy
import pyscf
cimport numpy
import math

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
def RMSD_e(self, double[:,:,:,:] t_old, double[:,:,:,:] t_new):

    print('calling RMSD_e from cymods')

    cdef int e_nocc = self.e_nocc
    cdef int e_nvir = self.e_nvir
    cdef int e_tot = self.e_tot

    cdef double sum_pieces = 0
    cdef int i, a, j, b

    for i in range(e_nocc):
        for a in range(e_nvir):
            for j in range(e_nocc):
                for b in range(e_nvir):

                    sum_pieces += ((t_new[i,a,j,b]-t_old[i,a,j,b])**2)

    RMSD_t = math.sqrt(sum_pieces)
    return RMSD_t

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
def RMSD_en(self, t_old, t_new):

    print('calling RMSD_en from cymods')
    cdef int nuclei = len(self.con.mol.nuc)

    cdef int e_nocc = self.e_nocc
    cdef int e_nvir = self.e_nvir
    cdef int e_tot = self.e_tot

    cdef int p_nocc
    cdef int p_nvir 
    cdef int p_tot

    cdef double sum_pieces = 0
    RMSD_t = []
    cdef double total_RMSD = 0

    cdef int i, j, a, I, A

    for i in range(nuclei):
        RMSD_t.append([None])

    for j in range(nuclei):
 
        t_old[j]: double[:,:,:,:]
        t_new[j]: double[:,:,:,:]

        sum_pieces = 0

        p_nocc = self.con.mf_nuc[j].mo_coeff[:,self.con.mf_nuc[j].mo_occ>0].shape[1]
        p_tot  = self.con.mf_nuc[j].mo_coeff[0,:].shape[0]
        p_nvir = p_tot - p_nocc


        for i in range(e_nocc):
            for a in range(e_nvir):
                for I in range(p_nocc):
                    for A in range(p_nvir):

                        sum_pieces += ((t_new[j][i,a,I,A]-t_old[j][i,a,I,A])**2)

        RMSD_t[j] = math.sqrt(sum_pieces)
        total_RMSD += RMSD_t[j]

    return total_RMSD

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
def RMSD_n(self, t_old, t_new):
    
    print('calling RMSD_n from cymods')

    cdef int nuclei = len(self.con.mol.nuc)
    cdef int i, j, I, A, J, B

    RMSD_t = numpy.empty((nuclei,nuclei), dtype=object)
    for i in range(nuclei):
        for j in range(nuclei):
            RMSD_t[i][j] = [None]


    cdef double sum_pieces = 0
    
    for i in range(nuclei):

        if nuclei==1:
            pass

        else:

            for j in range(nuclei):

                sum_pieces = 0

                p_nocc_i: int = self.con.mf_nuc[i].mo_coeff[:,self.con.mf_nuc[i].mo_occ>0].shape[1]
                p_tot_i: int  = self.con.mf_nuc[i].mo_coeff[0,:].shape[0]
                p_nvir_i: int = p_tot_i - p_nocc_i
                p_nocc_j: int = self.con.mf_nuc[j].mo_coeff[:,self.con.mf_nuc[j].mo_occ>0].shape[1]
                p_tot_j: int  = self.con.mf_nuc[j].mo_coeff[0,:].shape[0]
                p_nvir_j: int  = p_tot_j - p_nocc_j
                
                t_old[i][j]: double[:,:,:,:]
                t_new[i][j]: double[:,:,:,:]

                for I in range(p_nocc_i):
                    for A in range(p_nvir_i):
                        for J in range(p_nocc_j):
                            for B in range(p_nvir_j):

                                sum_pieces += ((t_new[i][j][I,A,J,B]-t_old[i][j][I,A,J,B])**2)

                RMSD_t[i][j] =  math.sqrt(sum_pieces)



    cdef double total_RMSD = 0

    for i in range(nuclei):
        for j in range(i):

            total_RMSD += RMSD_t[i][j]

    return total_RMSD
