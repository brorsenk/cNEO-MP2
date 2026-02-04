cimport cython
import pyscf
import numpy
cimport numpy
from pyscf import neo, ao2mo
from pyscf.dft import numint
import cymods
from cymods.mp2_dens import mp2_density
from cymods.mp2_dens.mp2_density import mp2_density_one

#--------------------------------------------------------------------
# Comparison of CHF and Total (with MP2) Densities
#--------------------------------------------------------------------
# [Notes:] This will be used to treat both the CHF and CHFMP2 cases and compare.
#
#          [October 7th, 2022] Resolving the AttributeError for mol.cart?
#
#          The error that it is currently giving is an attribute error -> that the mole object list has no attribute "cart"...
#          Adding cart = True to the inside of the mol.build function did not work, this gave errors with arrays of inconsistent sizes.
#          Trying to add an attribute for it inside of the CHFMP2 class attribut list in the initialization also didn't work.
#          It seems that this is an attribute of the gto mole object and not of the neo mole object perhaps?
#          I tried using the mf = RHF(mol) object instead of the con chfmp2 object to see if that would work since it uses the
#          gto mole object, but the problem here is that the gto mole object does not have the attribute nuc or mf.mf_nuc,
#          so this was not able to work either. 
#
#          [October 13th, 2022] The AttributeError for mol.cart has been resolved!
#
#          It turns out that the error was simply caused by one piece where I had forgotten to index the list for the nuclear objects.
#          For future reference though the self.cart attribute gets set in the gto/mole.py file and the default value is false.
#
#
#          [October 14th, 2022] The densities are very, very small, basically zero. I've tried narrowing the start and finish values
#                               for the grid range and have added the grid to all three dimensions, as well as centering the oxygen atom
#                               at the origin and placing the hydrogen atoms in the x-y plane at the appropriate distances from the oxygen
#                               and essentially keeping very close to the experimental angle between them. Nothing has worked so far to 
#                               increase the densities to be significantly non-zero.
#
#                              These coordinates reduced the size of the negative power of ten by almost half.
#
#                              number_of_points = 30 
#                              grid_start_x = -0.25
#                              grid_finish_x = 0.96 
#                              grid_start_y = -0.1 
#                              grid_finish_y = 0.96 
#                              grid_start_z = -0.2 
#                              grid_finish_z = 0.2
#
#          [November 14th, 2022] - Last week I was finally able to resolve most of the problems with the density values.
#                                 
#          One major problem was that it wasn't summing only over the occupied indexes but over all indexes for the NEOHF and CHF
#          cases. Another problem was that for the MP2 density I had been adding the CHF density to the MP2 density, but 
#          I was doing this before the multiplication with the coefficients and ao_set matrices, so it was basically just 
#          giving the same erroneous density values that I was obtaining when I had the HF and CHF sections summing over all
#          of the molecular orbitals. 
#          The only problems remaining are that I need to implement the electronic and electronic-nuclear MP2 density contributions,
#          and figure out why the density values are still not getting computed right when more than one nucleus is treated 
#          quantum mechanically and is not centered at the origin. 
#
#
#          [December 5th, 2022] - Both 1D and 3D densities are working and giving reasonable results!
#               
#          The problem with the 3D densities was that when they were integrated "over all of space," (over a sufficiently large region),
#          the resultant values were around 0.94 - 0.96 depending on the molecule, and were not quite close enough to one. It turns out 
#          that the reason for this is that in the construction of the volume element in the numerical integration sum, for each dimensional
#          piece, I was taking the total distance between the start point and end point in each dimension, and dividing it by the number of 
#          points on the grid in that dimension. What I hadn't been thinking of is that when you take say X number of points, there are only
#          X-1 spaces between the points, so changing the denominators made the integration go to one.  
#
#          [July 7th, 2024] - Turning this into a module to calculate all density related properties we need.
#
#          Now that everything with the constraint has been fixed and implemented I am working on converting this old density code into
#          a module that will take care of calculating all of the required 1D slices of the density, the 3D density, and the expectation 
#          values of the density, both at the cHF and total (cHF + cMP2) levels.



@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
def one_dim_density_on_axis(self, int npoints):

    cdef int nuclei = len(self.con.mol.nuc)
    con = self.con
    #---------------------------------
    # Define Grid
    #---------------------------------
    coords = []
    
    cdef double nuclear_radius = 0.5 
    cdef int i, a, p, q, mi, ni, r, t
    cdef int three = 3

    #-----------------------------------------------------------
    # Append an empty coordinate array for each quantum nucelus.
    #-----------------------------------------------------------
    for i in range(nuclei):
 
        coords.append(numpy.zeros((npoints, three)))
    
    #---------------------------------
    # Build Grid
    #---------------------------------
   
    for i in range(nuclei):
 
        r_mol = self.con.mol.atom_coords(unit='BOHR')
        x_coordinate = r_mol[i,0]
        y_coordinate = r_mol[i,1]
        z_coordinate = r_mol[i,2]

        grid_start = z_coordinate - nuclear_radius
        grid_finish = z_coordinate + nuclear_radius
 

        coords[i][:,0] = x_coordinate
        coords[i][:,1] = y_coordinate
        coords[i][:,2] = numpy.linspace(grid_start, grid_finish, npoints)
        print(coords[i])


    #---------------------------------------------------------------------------
    # Initialize Empty Lists for Density Matrices and Atomic Orbital Integrals  
    #---------------------------------------------------------------------------      
    ao_density_list_chf = []
    ao_density_list_mp2 = []
    ao_values_chf = []
    nuclear_overlap_list = []
    mo_density_list_chf = []
    mo_density_list_mp2 = []
    total_density_list = []

    #-----------------------------------------------------
    # Append Atomic Orbital Integral Arrays to List
    #-----------------------------------------------------
    for i in range(nuclei):

        ao_values_chf.append(numint.eval_ao(self.con.mol.nuc[i], coords[i]))

    #-----------------------------------------------------------------------------------------
    # Construct the AO Density Matrices per Nuclei and Append to List
    #-----------------------------------------------------------------------------------------
    for i in range(nuclei):

        p_nocc = self.con.mf_nuc[i].mo_coeff[:,self.con.mf_nuc[i].mo_occ>0].shape[1]
        p_tot  = self.con.mf_nuc[i].mo_coeff[0,:].shape[0]
        p_nvir = p_tot - p_nocc

        nuclear_overlap_list.append(con.mf_nuc[i].get_ovlp(con.mol.nuc[i]))

        density_matrix_i_chf = numpy.zeros((p_tot, p_tot))

        #--------------------------------------
        # Atomic Orbital Basis Density Matrix
        #--------------------------------------
        for mi in range(p_tot):
            for ni in range(p_tot):
                for m in range(p_tot):
                    density_matrix_i_chf[mi,ni] += (self.con.mf_nuc[i].mo_coeff[mi,m]*self.con.mf_nuc[i].mo_coeff[ni,m])

        ao_density_list_chf.append(density_matrix_i_chf)
        ao_density_list_mp2.append(mp2_density_one(self, i, self.t_nuc, self.l_nuc, self.t_elecnuc, self.l_elecnuc))
    #-------------------------------------------------
    # Transformation to Molecular Orbital Basis       
    #-------------------------------------------------
    for i in range(nuclei):

        mo_density_matrix_i_chf = self.con.mf_nuc[i].mo_coeff.T @ nuclear_overlap_list[i] @ ao_density_list_chf[i] @ nuclear_overlap_list[i] @ self.con.mf_nuc[i].mo_coeff
        mo_density_list_chf.append(mo_density_matrix_i_chf)

        mo_density_matrix_i_mp2 = self.con.mf_nuc[i].mo_coeff.T @ nuclear_overlap_list[i] @ ao_density_list_mp2[i] @ nuclear_overlap_list[i] @ self.con.mf_nuc[i].mo_coeff
        mo_density_list_mp2.append(mo_density_matrix_i_mp2)

    for i in range(nuclei):

        total_density_matrix_chfmp2 = mo_density_list_chf[i] + mo_density_list_mp2[i]  
        total_density_list.append(total_density_matrix_chfmp2)


    #-----------------------------------------------------
    # Initialize Empty List to Store Density Values
    #-----------------------------------------------------
    density_list_chf_master = []
    density_list_mp2_master = []
 
    out_list = []

    for i in range(nuclei):
        output_array = numpy.zeros((npoints, three))
        out_list.append(output_array)
    #------------------------------
    # Print Title
    #------------------------------ 
    print('CHF and Total Densities')

        
    #-----------------------------------------------------
    # Build and Store CHF Denisty for Each Nuclei
    #-----------------------------------------------------
    for i in range(nuclei):

        density_list_chf = numpy.zeros(npoints)
        density_list_mp2 = numpy.zeros(npoints)

        p_nocc = self.con.mf_nuc[i].mo_coeff[:,self.con.mf_nuc[i].mo_occ>0].shape[1]
        p_tot  = self.con.mf_nuc[i].mo_coeff[0,:].shape[0]
        p_nvir = p_tot - p_nocc

        density_chf = 0
        density_mp2 = 0
    
        print(mo_density_list_chf[i])
 
        for a in range(npoints):

            density_chf = 0
            density_mp2 = 0

            ao_set_chf = numpy.outer(ao_values_chf[i][a], ao_values_chf[i][a])

            for p in range(p_nocc):
                for q in range(p_nocc):

                    #-------------------------------------------------------------
                    # Compute the 1D constrained-Hartree Fock density
                    #-------------------------------------------------------------
                    density_chf += mo_density_list_chf[i][p,q] * numpy.matmul(self.con.mf_nuc[i].mo_coeff[:,p].T, numpy.matmul(ao_set_chf, self.con.mf_nuc[i].mo_coeff[:,q]))

            for r in range(p_tot):
                for t in range(p_tot):

                    #-------------------------------------------------------------
                    # Compute the 1D total density
                    #-------------------------------------------------------------
                    density_mp2 += mo_density_list_mp2[i][r,t] * numpy.matmul(self.con.mf_nuc[i].mo_coeff[:,r].T, numpy.matmul(ao_set_chf, self.con.mf_nuc[i].mo_coeff[:,t]))


            print(coords[i][a,2], density_chf)
            print(coords[i][a,2], density_mp2)

        #----------------------------------------------------
        # Append the density for each point to a list
        #----------------------------------------------------
            density_list_chf[a] = density_chf
            density_list_mp2[a] = density_mp2

        density_list_chf_master.append(density_list_chf)
        density_list_mp2_master.append(density_list_mp2)


    for i in range(nuclei):
        for a in range(npoints):

            out_list[i][a,0] = coords[i][a,2]
            out_list[i][a,1] = density_list_chf_master[i][a]
            out_list[i][a,2] = density_list_chf_master[i][a] + density_list_mp2_master[i][a]



    for i in range(nuclei):
        for a in range(npoints):
     
            print(i, out_list[i][a,0], out_list[i][a,1], out_list[i][a,2])



    return out_list



@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
def one_dim_density_off_axis(self, int npoints):

    cdef int nuclei = len(self.con.mol.nuc)
    con = self.con
    #---------------------------------
    # Define Grid
    #---------------------------------
    coords = []
    
    cdef double nuclear_radius = 0.5 
    cdef int i, a, p, q, mi, ni, r, t
    cdef int three = 3

    #-----------------------------------------------------------
    # Append an empty coordinate array for each quantum nucelus.
    #-----------------------------------------------------------
    for i in range(nuclei):
 
        coords.append(numpy.zeros((npoints, three)))
    
    #---------------------------------
    # Build Grid
    #---------------------------------
   
    for i in range(nuclei):
 
        r_mol = self.con.mol.atom_coords(unit='BOHR')
        x_coordinate = r_mol[i,0]
        y_coordinate = r_mol[i,1]
        z_coordinate = r_mol[i,2]

        grid_start = x_coordinate - nuclear_radius
        grid_finish = x_coordinate + nuclear_radius

        #grid_start = y_coordinate - nuclear_radius
        #grid_finish = y_coordinate + nuclear_radius

        #coords[i][:,0] = x_coordinate
        coords[i][:,0] = numpy.linspace(grid_start, grid_finish, npoints)
        coords[i][:,1] = y_coordinate
        #coords[i][:,1] =numpy.linspace(grid_start, grid_finish, npoints)
        coords[i][:,2] = z_coordinate 
        print(coords[i])


    #---------------------------------------------------------------------------
    # Initialize Empty Lists for Density Matrices and Atomic Orbital Integrals  
    #---------------------------------------------------------------------------      
    ao_density_list_chf = []
    ao_density_list_mp2 = []
    ao_values_chf = []
    nuclear_overlap_list = []
    mo_density_list_chf = []
    mo_density_list_mp2 = []
    total_density_list = []

    #-----------------------------------------------------
    # Append Atomic Orbital Integral Arrays to List
    #-----------------------------------------------------
    for i in range(nuclei):

        ao_values_chf.append(numint.eval_ao(self.con.mol.nuc[i], coords[i]))

    #-----------------------------------------------------------------------------------------
    # Construct the AO Density Matrices per Nuclei and Append to List
    #-----------------------------------------------------------------------------------------
    for i in range(nuclei):

        p_nocc = self.con.mf_nuc[i].mo_coeff[:,self.con.mf_nuc[i].mo_occ>0].shape[1]
        p_tot  = self.con.mf_nuc[i].mo_coeff[0,:].shape[0]
        p_nvir = p_tot - p_nocc

        nuclear_overlap_list.append(con.mf_nuc[i].get_ovlp(con.mol.nuc[i]))

        density_matrix_i_chf = numpy.zeros((p_tot, p_tot))

        #--------------------------------------
        # Atomic Orbital Basis Density Matrix
        #--------------------------------------
        for mi in range(p_tot):
            for ni in range(p_tot):
                for m in range(p_tot):
                    density_matrix_i_chf[mi,ni] += (self.con.mf_nuc[i].mo_coeff[mi,m]*self.con.mf_nuc[i].mo_coeff[ni,m])

        ao_density_list_chf.append(density_matrix_i_chf)
        ao_density_list_mp2.append(mp2_density_one(self, i, self.t_nuc, self.l_nuc, self.t_elecnuc, self.l_elecnuc))
    #-------------------------------------------------
    # Transformation to Molecular Orbital Basis       
    #-------------------------------------------------
    for i in range(nuclei):

        mo_density_matrix_i_chf = self.con.mf_nuc[i].mo_coeff.T @ nuclear_overlap_list[i] @ ao_density_list_chf[i] @ nuclear_overlap_list[i] @ self.con.mf_nuc[i].mo_coeff
        mo_density_list_chf.append(mo_density_matrix_i_chf)

        mo_density_matrix_i_mp2 = self.con.mf_nuc[i].mo_coeff.T @ nuclear_overlap_list[i] @ ao_density_list_mp2[i] @ nuclear_overlap_list[i] @ self.con.mf_nuc[i].mo_coeff
        mo_density_list_mp2.append(mo_density_matrix_i_mp2)

    for i in range(nuclei):

        total_density_matrix_chfmp2 = mo_density_list_chf[i] + mo_density_list_mp2[i]  
        total_density_list.append(total_density_matrix_chfmp2)


    #-----------------------------------------------------
    # Initialize Empty List to Store Density Values
    #-----------------------------------------------------
    density_list_chf_master = []
    density_list_mp2_master = []
 
    out_list = []

    for i in range(nuclei):
        output_array = numpy.zeros((npoints, three))
        out_list.append(output_array)
    #------------------------------
    # Print Title
    #------------------------------ 
    print('CHF and Total Densities')

        
    #-----------------------------------------------------
    # Build and Store CHF Denisty for Each Nuclei
    #-----------------------------------------------------
    for i in range(nuclei):

        density_list_chf = numpy.zeros(npoints)
        density_list_mp2 = numpy.zeros(npoints)

        p_nocc = self.con.mf_nuc[i].mo_coeff[:,self.con.mf_nuc[i].mo_occ>0].shape[1]
        p_tot  = self.con.mf_nuc[i].mo_coeff[0,:].shape[0]
        p_nvir = p_tot - p_nocc

        density_chf = 0
        density_mp2 = 0
    
        print(mo_density_list_chf[i])
 
        for a in range(npoints):

            density_chf = 0
            density_mp2 = 0

            ao_set_chf = numpy.outer(ao_values_chf[i][a], ao_values_chf[i][a])

            for p in range(p_nocc):
                for q in range(p_nocc):

                    #-------------------------------------------------------------
                    # Compute the 1D constrained-Hartree Fock density
                    #-------------------------------------------------------------
                    density_chf += mo_density_list_chf[i][p,q] * numpy.matmul(self.con.mf_nuc[i].mo_coeff[:,p].T, numpy.matmul(ao_set_chf, self.con.mf_nuc[i].mo_coeff[:,q]))

            for r in range(p_tot):
                for t in range(p_tot):

                    #-------------------------------------------------------------
                    # Compute the 1D total density
                    #-------------------------------------------------------------
                    density_mp2 += mo_density_list_mp2[i][r,t] * numpy.matmul(self.con.mf_nuc[i].mo_coeff[:,r].T, numpy.matmul(ao_set_chf, self.con.mf_nuc[i].mo_coeff[:,t]))


            print(coords[i][a,0], density_chf)
            print(coords[i][a,0], density_mp2)
#            print(coords[i][a,1], density_chf)
#            print(coords[i][a,1], density_mp2)

        #----------------------------------------------------
        # Append the density for each point to a list
        #----------------------------------------------------
            density_list_chf[a] = density_chf
            density_list_mp2[a] = density_mp2

        density_list_chf_master.append(density_list_chf)
        density_list_mp2_master.append(density_list_mp2)


    for i in range(nuclei):
        for a in range(npoints):

            out_list[i][a,0] = coords[i][a,0]
            #out_list[i][a,0] = coords[i][a,1]
            out_list[i][a,1] = density_list_chf_master[i][a]
            out_list[i][a,2] = density_list_chf_master[i][a] + density_list_mp2_master[i][a]



    for i in range(nuclei):
        for a in range(npoints):
     
            print(i, out_list[i][a,0], out_list[i][a,1], out_list[i][a,2])



    return out_list



#@cython.boundscheck(False)  # Deactivate bounds checking.
#@cython.wraparound(False)   # Deactivate negative indexing.
#def three_dim_density(self, npoints):




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


#        #---------------------------------------------------------
#        # [2.8.3] - Calculating the 3D CHF Densities
#        #---------------------------------------------------------
#
#        coords_chf_3D = [] 
#
#
#        number_of_points_3D = number_of_points**3
#        nuclear_radius_chf_3D = 0.6
#
#        ao_values_chf_3D = []
#        ao_density_array_list_chf_3D = []
#        nuclear_overlap_list_chf_3D = []
#        mo_density_array_list_chf_3D = [] 
#        chfmp2_total_density_list_3D = [] 
#
##
#       
#        #---------------------------------------------
#        # Building the 3D Coordinate Array and Grid
#        #---------------------------------------------
##
#        for i in range(len(self.con.mol.nuc)):
#
#            coords_chf_3D.append(numpy.zeros((number_of_points**3, 3))) 
#
#            R_chf = self.con.mol.atom_coords(unit='ANG')
#
#            x_coordinate_chf = R_chf[i,0]
#            y_coordinate_chf = R_chf[i,1]
#            z_coordinate_chf = R_chf[i,2]
#
#            #---------------------------------------------------------
#            # Grid Ranges
#            #---------------------------------------------------------
#            chf_grid_start_x = x_coordinate_chf - nuclear_radius_chf_3D
#            chf_grid_finish_x = x_coordinate_chf + nuclear_radius_chf_3D
#
#            chf_grid_start_y = y_coordinate_chf - nuclear_radius_chf_3D
#            chf_grid_finish_y = y_coordinate_chf + nuclear_radius_chf_3D
#
#            chf_grid_start_z = z_coordinate_chf - nuclear_radius_chf_3D
#            chf_grid_finish_z = z_coordinate_chf + nuclear_radius_chf_3D
#
#
#            p_nocc = self.con.mf_nuc[i].mo_coeff[:,self.con.mf_nuc[i].mo_occ>0].shape[1]
#            p_tot  = self.con.mf_nuc[i].mo_coeff[0,:].shape[0]
#            p_nvir = p_tot - p_nocc
#
#            #---------------------------------------------------------
#            # Grid Spacing
#            #---------------------------------------------------------
#            linear_spacing_x = numpy.linspace(chf_grid_start_x, chf_grid_finish_x, number_of_points)
#            linear_spacing_y = numpy.linspace(chf_grid_start_y, chf_grid_finish_y, number_of_points)
#            linear_spacing_z = numpy.linspace(chf_grid_start_z, chf_grid_finish_z, number_of_points)
#
#            index = 0 
#
#            for h in range(number_of_points):
#               for j in range(number_of_points):
#                  for k in range(number_of_points):
#
#                      coords_chf_3D[i][index,0] = linear_spacing_x[h]
#                      coords_chf_3D[i][index,1] = linear_spacing_y[j]
#                      coords_chf_3D[i][index,2] = linear_spacing_z[k]
#
#                      index = index+1
#
#
#            ao_values_chf_3D_element = numint.eval_ao(self.con.mol.nuc[i], coords_chf_3D[i])
#
#            ao_values_chf_3D.append(ao_values_chf_3D_element)
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#        #-----------------------------------------------------------------------------------------
#        # Construct the AO Density Matrices per Nuclei and Append to List
#        #-----------------------------------------------------------------------------------------
#        for i in range(len(self.con.mol.nuc)):
#
#            p_nocc = self.con.mf_nuc[i].mo_coeff[:,self.con.mf_nuc[i].mo_occ>0].shape[1]
#
#            p_tot  = self.con.mf_nuc[i].mo_coeff[0,:].shape[0]
#            p_nvir = p_tot - p_nocc
#
#            nuclear_overlap_list_chf_3D.append(con.mf_nuc[i].get_ovlp(con.mol.nuc[i]))
#
#            density_matrix_i_chf_3D = numpy.zeros((p_tot, p_tot))
#     
#            #--------------------------------------
#            # Atomic Orbital Basis Density Matrix
#            #--------------------------------------
#            for mi in range(p_tot):
#                for ni in range(p_tot):
#                    for m in range(p_tot):
#                        density_matrix_i_chf_3D[mi,ni] += (self.con.mf_nuc[i].mo_coeff[mi,m]*self.con.mf_nuc[i].mo_coeff[ni,m])
#
#            ao_density_array_list_chf_3D.append(density_matrix_i_chf_3D)
#
#
#        #-------------------------------------------------
#        # Transformation to Molecular Orbital Basis       
#        #-------------------------------------------------
#        for i in range(len(self.con.mol.nuc)):
#
#            mo_density_matrix_i_chf_3D = self.con.mf_nuc[i].mo_coeff.T @ nuclear_overlap_list_chf_3D[i] @ ao_density_array_list_chf_3D[i] @ nuclear_overlap_list_chf_3D[i] @ self.con.mf_nuc[i].mo_coeff
#            mo_density_array_list_chf_3D.append(mo_density_matrix_i_chf_3D)
#
#
#        for i in range(len(self.con.mol.nuc)):
#
#            total_density_matrix_chfmp2_3D = mo_density_array_list_chf_3D[i] + gamma_ITMP2_total_list[i]
#            chfmp2_total_density_list_3D.append(total_density_matrix_chfmp2_3D)
#
#
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
#        density_list_of_lists_chf_3D = []
#        mp2_density_list_of_lists_3D = []
#
#
#        for i in range(len(self.con.mol.nuc)):
#          
#            density_list_of_lists_chf_3D.append([])
#            mp2_density_list_of_lists_3D.append([])
#
#
#        print ('3D CHF Densities')
#        print(line)
#
#        for i in range(len(self.con.mol.nuc)):
#
#            p_nocc = self.con.mf_nuc[i].mo_coeff[:,self.con.mf_nuc[i].mo_occ>0].shape[1]
#            p_tot  = self.con.mf_nuc[i].mo_coeff[0,:].shape[0]
##            p_nvir = p_tot - p_nocc
#
#            density_list_chf_3D = []
#            mp2_density_list_3D = []
#
#            density_chf_3D = 0
#            density_chfmp2_3D = 0
#
#            density_for_int_over_all_space = 0
#            mp2_density_for_int_over_all_space = 0
#
#
#            for a in range(number_of_points_3D):
#
#                density_chf_3D = 0.0 
#
#                ao_set_chf_3D = numpy.outer(ao_values_chf_3D[i][a], ao_values_chf_3D[i][a])
#
#                for p in range(p_nocc):
#                    for q in range(p_nocc):
#             
#                        density_chf_3D += mo_density_array_list_chf_3D[i][p,q]*numpy.matmul(self.con.mf_nuc[i].mo_coeff[:,p].T, numpy.matmul(ao_set_chf_3D, self.con.mf_nuc[i].mo_coeff[:,q]))
#
#                density_list_of_lists_chf_3D[i].append(density_chf_3D)
#
#                density_list_chf_3D.append(density_chf_3D)
#
#                       density_for_int_over_all_space += (((chf_grid_finish_x - chf_grid_start_x)/number_of_points)*((chf_grid_finish_y - chf_grid_start_y)/number_of_points)*((chf_grid_finish_z - chf_grid_start_z)/number_of_points))*(mo_density_array_list_chf[i][p,q]*numpy.matmul(self.con.mf_nuc[i].mo_coeff[:,p].T, numpy.matmul(ao_set_chf_3D, self.con.mf_nuc[i].mo_coeff[:,q])))
#
#
#                print(coords_chf_3D[i][a][0],coords_chf_3D[i][a][1],coords_chf_3D[i][a][2], density_chf_3D)
#
#            print(density_for_int_over_all_space)
#
#            print(line)
#            print('3D MP2 Densities')
#            print(line)
#
#
#            for a in range(number_of_points_3D):
#
#                density_chfmp2_3D = 0.0
#
#                ao_set_chf_3D = numpy.outer(ao_values_chf_3D[i][a], ao_values_chf_3D[i][a])
#
#                for j in range(p_tot):
#                    for k in range(p_tot):
#
#                        density_chfmp2_3D += gamma_ITMP2_total_list[i][j,k] * numpy.matmul(self.con.mf_nuc[i].mo_coeff[:,j].T , numpy.matmul(ao_set_chf_3D, self.con.mf_nuc[i].mo_coeff[:,k]))
#
#                mp2_density_list_of_lists_3D[i].append(density_chfmp2_3D)
#
#                mp2_density_list_3D.append(density_chfmp2_3D)
#
#                print(coords_chf_3D[i][a][0],coords_chf_3D[i][a][1],coords_chf_3D[i][a][2], density_chfmp2_3D)
#
#            print(line)
#
#            #----------------------------------------------------
#            # [2.8.4] - Integrated Density over all space
#            #----------------------------------------------------
#
#        total_integrated_density_3D = 0
#
#        for i in range(len(self.con.mol.nuc)):
#
#            R_chf = self.con.mol.atom_coords(unit='ANG')
#
#            x_coordinate_chf = R_chf[i,0]
#            y_coordinate_chf = R_chf[i,1]
#            z_coordinate_chf = R_chf[i,2]
#
#            #---------------------------------------------------------
#            # Grid Ranges
#            #---------------------------------------------------------
#            chf_grid_start_x = x_coordinate_chf - nuclear_radius_chf_3D
#            chf_grid_finish_x = x_coordinate_chf + nuclear_radius_chf_3D
#
#            chf_grid_start_y = y_coordinate_chf - nuclear_radius_chf_3D
#            chf_grid_finish_y = y_coordinate_chf + nuclear_radius_chf_3D
#
#            chf_grid_start_z = z_coordinate_chf - nuclear_radius_chf_3D
#            chf_grid_finish_z = z_coordinate_chf + nuclear_radius_chf_3D
#
#
#            length_element_x = ((chf_grid_finish_x - chf_grid_start_x)/(number_of_points-1))
#            length_element_y = ((chf_grid_finish_y - chf_grid_start_y)/(number_of_points-1))
#            length_element_z = ((chf_grid_finish_z - chf_grid_start_z)/(number_of_points-1))
#
#            volume_element = length_element_x*length_element_y*length_element_z
#
#            for a in range(number_of_points_3D):
#         
#                total_integrated_density_3D += volume_element*(density_list_of_lists_chf_3D[i][a]+mp2_density_list_of_lists_3D[i][a])
##
#        print(asterisk)
#        print(asterisk)
#        print(asterisk)
#        print(total_integrated_density_3D)
#        print(asterisk)
#        print(asterisk)
#        print(asterisk)
#
#         #----------------------------------------------------------------------------
#         # [2.8.5] - Expectation Values for the CHF and Total (with MP2) Densities
#         #----------------------------------------------------------------------------
#
#        for i in range(self.con.mol.nuc):
#
#            density_expectation_value_vector_chf = numpy.zeros((1,3))
#            density_expectation_value_vector_total = numpy.zeros((1,3))
#
#            density_ev_chf_x = 0
#            density_ev_chf_y = 0
#            density_ev_chf_z = 0
#
#            density_ev_total_x = 0
#            density_ev_total_y = 0
#            density_ev_total_z = 0
#
#            for a in range(number_of_points_3D):
#           
#                length_element_x = ((chf_grid_finish_x - chf_grid_start_x)/(number_of_points-1))
#                length_element_y = ((chf_grid_finish_y - chf_grid_start_y)/(number_of_points-1))
#                length_element_z = ((chf_grid_finish_z - chf_grid_start_z)/(number_of_points-1))
#               
#                volume_element = length_element_x*length_element_y*length_element_z 
#
#                density_ev_chf_x += coords_chf_3D[i][a][0]*density_list_of_lists_chf_3D[i][a]*volume_element
#                density_ev_chf_y += coords_chf_3D[i][a][1]*density_list_of_lists_chf_3D[i][a]*volume_element
#                density_ev_chf_z += coords_chf_3D[i][a][2]*density_list_of_lists_chf_3D[i][a]*volume_element
#
#                print(a, '*', 'x ', coords_chf_3D[i][a][0], density_list_chf_3D[a], density_ev_chf_x)
#                print(a, '*', 'y ',coords_chf_3D[i][a][1], density_list_chf_3D[a], density_ev_chf_y)
#                print(a, '*', 'z ',coords_chf_3D[i][a][2], density_list_chf_3D[a], density_ev_chf_z)
#
#                density_ev_total_x += coords_chf_3D[i][a][0]*(density_list_of_lists_chf_3D[i][a]+mp2_density_list_of_lists_3D[i][a])*volume_element
#                density_ev_total_y += coords_chf_3D[i][a][1]*(density_list_of_lists_chf_3D[i][a]+mp2_density_list_of_lists_3D[i][a])*volume_element
#                density_ev_total_z += coords_chf_3D[i][a][2]*(density_list_of_lists_chf_3D[i][a]+mp2_density_list_of_lists_3D[i][a])*volume_element
#
#                print(a, '*', 'x ', coords_chf_3D[i][a][0], density_list_chf_3D[a], density_ev_chf_x, ' * ', density_ev_total_x)
#                print(a, '*', 'y ', coords_chf_3D[i][a][1], density_list_chf_3D[a], density_ev_chf_y, ' * ', density_ev_total_y)
#                print(a, '*', 'z: ', coords_chf_3D[i][a][2], 'chf density: ', density_list_chf_3D[a], 'total density: ', (density_list_chf_3D[a]+mp2_density_list_3D[a]), 'sum chf: ', density_ev_chf_z, ' * ', 'sum total: ', density_ev_total_z)
#              
#
#            density_expectation_value_vector_chf[0,0] = density_ev_chf_x
#            density_expectation_value_vector_chf[0,1] = density_ev_chf_y
#            density_expectation_value_vector_chf[0,2] = density_ev_chf_z
#
#            density_expectation_value_vector_total[0,0] = density_ev_total_x
#            density_expectation_value_vector_total[0,1] = density_ev_total_y
##            density_expectation_value_vector_total[0,2] = density_ev_total_z
#
#            print(line)
##            print(i)
#            print(line)
#            print(density_expectation_value_vector_chf)
#            print(density_expectation_value_vector_total)
#           print(coords_chf_3D[i])
#
#
#
#############
