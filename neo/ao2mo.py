#!/usr/bin/env python
# Author: Kurt Brorsen (brorsenk@missouri.edu)

import pyscf.gto
import numpy
import pyscf.ao2mo as ao2mo
from timeit import default_timer as timer

def ep_setup(mf, mf2,  i=0, j=0, ep=True):

    if(ep==True):

        mol_tot = mf.mol.elec + mf.mol.nuc[i] 
        tot1  = mf.mf_elec.mo_coeff[0,:].shape[0]
        tot2  = mf.mf_nuc[i].mo_coeff[0,:].shape[0]

    else:

        mol_tot = mf.mol.nuc[i] + mf.mol.nuc[j] 
        tot1  = mf.mf_nuc[i].mo_coeff[0,:].shape[0]
        tot2  = mf.mf_nuc[j].mo_coeff[0,:].shape[0]

    eri = mol_tot.intor('int2e',aosym='s8')

    mo_coeff_tot = numpy.zeros((tot1+tot2,tot1+tot2))

    if(ep==True):

        mo_coeff_tot[:tot1,:tot1] = mf2.mf_elec.mo_coeff
        mo_coeff_tot[tot1:,tot1:] = mf2.mf_nuc[i].mo_coeff

    else:

        mo_coeff_tot[:tot1,:tot1] = mf2.mf_nuc[i].mo_coeff
        mo_coeff_tot[tot1:,tot1:] = mf2.mf_nuc[j].mo_coeff

    return eri, mo_coeff_tot


def pp_setup(mf, mf2, i=0, j=0, pp=True):

    if(pp==True):

        mol_tot_p = mf.mol.nuc[i] + mf.mol.nuc[j]
        ptot1 = mf.mf_nuc[i].mo_coeff[0,:].shape[0]
        ptot2 = mf.mf_nuc[j].mo_coeff[0,:].shape[0]
   
    else:

        pass   
         
    eri = mol_tot_p.intor('int2e',aosym='s8')

    mo_coeff_tot_p = numpy.zeros((ptot1+ptot2,ptot1+ptot2))

    if(pp==True):

        mo_coeff_tot_p[:ptot1,:ptot1] = mf2.mf_nuc[i].mo_coeff
        mo_coeff_tot_p[ptot1:,ptot1:] = mf2.mf_nuc[j].mo_coeff   

    else:

        pass

    return eri, mo_coeff_tot_p

#end1

def ep_full(mf, mf2, i=0):

    print('calling ep_full')
    eri, mo_coeff_tot = ep_setup(mf,mf2,i)

    e_tot  = mf.mf_elec.mo_coeff[0,:].shape[0]
    p_tot  = mf.mf_nuc[i].mo_coeff[0,:].shape[0]

#    c_e= mo_coeff_tot[:,:e_tot]
#    c_n= mo_coeff_tot[:,e_tot:]

    eri_ep = ao2mo.incore.full(eri, mo_coeff_tot,compact=False)
    charge_i_ep =  mf.mol.nuc[i].super_mol.atom_charge(mf.mol.nuc[i].atom_index)
    scaled_eri_ep = eri_ep*(charge_i_ep)

    return scaled_eri_ep

#start2

def pp_full(mf, mf2, i=0, j=0):

   eri, mo_coeff_tot_p = pp_setup(mf, mf2, i, j)

#   p_tot_i = mf.mf_nuc[i].mo_coeff[0,:].shape[0]
#   p_tot_j = mf.mf_nuc[j].mo_coeff[0,:].shape[0]

   eri_pp = ao2mo.incore.full(eri, mo_coeff_tot_p, compact=False)
  
   charge_i_pp = mf.mol.nuc[i].super_mol.atom_charge(mf.mol.nuc[i].atom_index)
   charge_j_pp = mf.mol.nuc[j].super_mol.atom_charge(mf.mol.nuc[j].atom_index)

   scaled_eri_pp = eri_pp*(charge_i_pp*charge_j_pp)
 
   return scaled_eri_pp

#end2

def ep_ovov(mf,mf2, i=0):

    eri, mo_coeff_tot = ep_setup(mf, mf2, i)

    e_nocc = mf.mf_elec.mo_coeff[:,mf.mf_elec.mo_occ>0].shape[1]
    e_tot  = mf.mf_elec.mo_coeff[0,:].shape[0]
    e_nvir = e_tot - e_nocc

    p_nocc = mf.mf_nuc[i].mo_coeff[:,mf.mf_nuc[i].mo_occ>0].shape[1]
    p_tot  = mf.mf_nuc[i].mo_coeff[0,:].shape[0]
    p_nvir = p_tot - p_nocc
    
    
    co_e= mo_coeff_tot[:,:e_nocc]
    cv_e= mo_coeff_tot[:,e_nocc:e_tot]

    co_n=mo_coeff_tot[:,e_tot:e_tot+p_nocc]
    cv_n=mo_coeff_tot[:,e_tot+p_nocc:]

    charge_i_ep =  mf.mol.nuc[i].super_mol.atom_charge(mf.mol.nuc[i].atom_index)

    start = timer()
    eri_ep = ao2mo.incore.general(eri,(co_e,cv_e,co_n,cv_n), compact=False)
    finish = timer()

#    print('real time for ovov integral transformation = ',finish-start)
    scaled_eri_ep = eri_ep*(charge_i_ep)
    return scaled_eri_ep


def pp_ovov(mf, mf2, i=0, j=0):

    eri, mo_coeff_tot_p = pp_setup(mf,mf2,i,j,True)

    p_nocc_i = mf.mf_nuc[i].mo_coeff[:,mf.mf_nuc[i].mo_occ>0].shape[1]
    p_tot_i = mf.mf_nuc[i].mo_coeff[0,:].shape[0]
    p_nvir_i = p_tot_i - p_nocc_i

    charge_i_pp = mf.mol.nuc[i].super_mol.atom_charge(mf.mol.nuc[i].atom_index)

    p_nocc_j = mf.mf_nuc[j].mo_coeff[:,mf.mf_nuc[j].mo_occ>0].shape[1]
    p_tot_j = mf.mf_nuc[j].mo_coeff[0,:].shape[0]
    p_nvir_j = p_tot_j - p_nocc_j

    charge_j_pp = mf.mol.nuc[j].super_mol.atom_charge(mf.mol.nuc[j].atom_index) 

    co_ni = mo_coeff_tot_p[:,:p_nocc_i]
    cv_ni = mo_coeff_tot_p[:,p_nocc_i:p_tot_i]
   
    co_nj = mo_coeff_tot_p[:,p_tot_i:p_tot_i+p_nocc_j]
    cv_nj = mo_coeff_tot_p[:,p_tot_i+p_nocc_j:]
    
    start = timer()
    eri_pp = ao2mo.incore.general(eri,(co_ni,cv_ni,co_nj,cv_nj), compact=False)
    scaled_eri_pp = eri_pp*(charge_i_pp*charge_j_pp)
    finish = timer()

    return scaled_eri_pp



