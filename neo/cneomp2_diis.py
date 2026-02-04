#---------------------------------------------
# Importations
#---------------------------------------------
# For troubleshooting
import os
import sys
import inspect

# Math dependencies
import math
import numpy
import sympy
import scipy


class cneomp2_diis:

    def __init__(self, t_amps_inp, part1_occ, part1_tot, part2_occ, part2_tot):

        # The part1/2 or p1/2 abbreviations stand for particle 1 and particle 2, respectively.
        self.p1_tot = part1_tot # Total number of orbitals for particle one.
        self.p1_occ = part1_occ # Number of occupied orbitals for particle one.
        self.p1_vir = part1_tot - part1_occ # Number of virtual orbitals for particle one.
         
        self.p2_tot = part2_tot # Total number of orbitals for particle two.
        self.p2_occ = part2_occ # Number of occupied orbitals for particle two.
        self.p2_vir = part2_tot - part2_occ # Number of virtual orbitals for particle two.
       
        self.max_diis = 8 # Maximum number of DIIS extrapolated t-amplitudes and error vectors to store.
        self.iter = 0 # Iteration counter.
        self.dims_diis = (self.p1_occ*self.p1_vir*self.p2_occ*self.p2_vir) # Total length of ...

        self.error_ex = numpy.zeros((self.max_diis, (self.dims_diis)), dtype=float) # Extrapolated error vectors. ???
        self.t_ex = numpy.zeros((self.max_diis+1, (self.dims_diis)), dtype=float) # Extrapolated t-amplitudes. ???
        self.ts = numpy.zeros((1, (self.dims_diis)), dtype=float) # ???

        self.t_amps_inp = t_amps_inp

    def kernel(self):

        p1_tot = self.p1_tot
        p1_occ = self.p1_occ
        p1_vir = self.p1_vir
 
        p2_tot = self.p2_tot
        p2_occ = self.p2_occ
        p2_vir = self.p2_vir

        max_diis = self.max_diis

        t_amps = numpy.array(self.t_amps_inp)
        t_amps = t_amps.flatten()

        if self.iter == 0:
            self.t_ex[self.iter+1] = t_amps
            self.ts[0] = t_amps
            self.iter += 1
            t_n = t_amps

        elif self.iter < max_diis:
            ei = t_amps - self.ts[0]
            self.t_ex[self.iter+1] = t_amps
            self.error_ex[self.iter] = ei
            n = self.iter + 1
            B = numpy.zeros((n,n))
            b = numpy.zeros((n))
            for i in range(n-1):
                for j in range(i+1):
                    B[i,j] = B[j,i] = numpy.dot(self.e[i+1], self.e[j+1])
            B[n-1,:n-1] = -1
            B[:n-1,n-1] = -1
            B[:-1,:-1] /= numpy.abs(B[:-1,:-1]).max()
            b[n-1] = -1
            c=numpy.linalg.solve(B,b)
            t_n = numpy.zeros((len(t_amps)))
            for i in range(n-1):
                t_n += c[i]*self.t_ex[i+2]
            self.ts[0] = t_n
            self.iter += 1
   
        elif self.iter == max_diis:
            self.error_ex = numpy.roll(self.error_ex, -1, axis=0)
            self.t_ex = numpy.roll(self.t_ex, -1, axis=0)
            ei = t_amps - self.ts[0]
            self.t_ex[max_diis] = t_amps
            self.error_ex[max_diis-1] = ei
            n = max_diis + 1
            B = numpy.zeros((n,n))
            b = numpy.zeros((n))
            for i in rang(n-1):
                for j in range(i+1):
                    B[i,j] = B[j,i] = numpy.dot(self.error_ex[i], self.error_ex[j])

            B[n-1,:n-1] = -1
            B[:n-1,n-1] = -1
            B[:-1,:-1] /= numpy.abs(B[:-1,:-1]).max()
            b[n-1] = -1
            c = numpy.linalg.solve(B,b)
            t_n = numpy.zeros((len(t_amps)))
            for i in range(len(self.error_ex)): # -1 because t is longer than e
                t_n += c[i]*self.t_ex[i+1]
            self.ts[0] = t_n
            self.iter +=1
            
        else:
            self.error_ex = numpy.roll(self.error_ex, -1, axis=0)
            self.t_ex = numpy.roll(self.t_ex, -1, axis=0)
            ei = t_amps - self.ts[0]
            self.t_ex[max_diis] = t_amps
            self.error_ex[max_diis-1] = ei
            n = max_diis + 1
            B = numpy.zeros((n,n))
            b = numpy.zeros((n))
            for i in rang(n-1):
                for j in range(i+1):
                    B[i,j] = B[j,i] = numpy.dot(self.error_ex[i], self.error_ex[j])

            B[n-1,:n-1] = -1
            B[:n-1,n-1] = -1
            B[:-1,:-1] /= numpy.abs(B[:-1,:-1]).max()
            b[n-1] = -1
            c = numpy.linalg.solve(B,b)
            t_n = numpy.zeros((len(t_amps)))
            for i in range(len(self.error_ex)): # -1 because t is longer than e
                t_n += c[i]*self.t_ex[i+1]
            self.ts[0] = t_n
            self.iter +=1

        t_amps_out = t_n
        t_amps_out = t_amps_out.reshape(p1_occ, p1_vir, p2_occ, p2_vir)
        
        print('performing the DIIS steps')
        
        return t_amps_out


if __name__ == '__main__':


    mol = neo.Mole()
    mol.build(atom = '''H1 0 0 0; H2 0 0 0.750''', basis = 'sto-3g', charge = 0, quantum_nuc = [0,1])




