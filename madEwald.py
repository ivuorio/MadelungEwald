#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:09:17 2018

@author: ilkka
"""
import argparse
from math import sqrt, pi
from scipy.special import erfc
import numpy as np
from ase.io import read
import matplotlib.pyplot as plt

#general constant used in the code
sqrtpi=sqrt(pi)
twopi=2*pi


def RealConstant(ions, alpha):
    ##########################
    #REAL SPACE CONSTANT TERM#
    ##########################
    """
    COUNT THE CORRECTION TERM TO CANEL OUT THE INTERACTION OF ARTIFICIAL COUNTER CHARGES
    """
    
    ###VARIABLES###
    Sum = 0
    
    
    ###CALCULATIONS###
    for ion in ions:
         Sum += ion.charge*ion.charge
    
    ###RETURNS###
    return -alpha/sqrtpi*Sum
    
def RealSum(ions, alpha, cutoff):
    ######################
    #REAL SPACE SUMMATION#
    ######################
    """
    COUNT THE SUMMATION OF THE CHARGES IN REAL SPACE INSIDE THE CUTOFF AREA
    THIS USES "SEMI" SPHERICAL SUMMATION. AS IT SUMS SPHERICALLY THE UNIT CELLS INSTEAD SPHERICALLY SUMMING IONS.
    """
    ###CONSTANTS###
    cutoffsq=cutoff*cutoff
    
    ###VARIABLES###
    Sum = 0
    cell=ions.get_cell()
    nOfIons = len(ions)
    
    ###CALCULATIONS###
    #go trough all ion pairs
    for i in range(nOfIons):
        for j in range(nOfIons):
            #And unit cells in the cutoff range
            for n0 in range(-cutoff,cutoff+1):
                for n1 in range(-cutoff,cutoff+1):
                    for n2 in range(-cutoff,cutoff+1):
                        #skip the calculation for ions intercation on its self
                        if i==j and n0==n1==n2==0: 
                            continue
                        else:
                            #location of the unit cell
                            a=n0*cell[0]+n1*cell[1]+n2*cell[2]
                            #location of the particle
                            r=ions[i].position.copy()-(ions[j].position.copy()+a)
                            #check if the particle is in the range of cutoff
                            dsq=np.dot(r,r)
                            if dsq > cutoffsq:
                                continue
                            #distance of the particles i and j
                            d=sqrt(dsq)
                            #addition of charge_i*charge_j*erfc(a*d)/(d)
                            addition = ions[i].charge.copy()*ions[j].charge.copy()*erfc(alpha*d)/(d)
                            #add to the summation
                            Sum += addition
    return Sum
                        
def ResiprocalSum(ions, kmax, alpha):
    ####################################
    #RESIPROCAL SPACE(K-SPACE)SUMMATION#
    ####################################
    """
    USING THE FOURIER SPACE COUNT THE CHARGES THAT ARE SET IN THE CATEGORY OF LONG RANGE EFFECTS
    """
    ###CONSTANTS###
    
    ###VARIABLES###
    #variable for the actuall summation
    Sum = 0
    #reciprocal cell vectors
    Kcell = ions.get_reciprocal_cell()*twopi
    
    ###CALCULATIONS###
    
    for a0 in range(-kmax,kmax+1):
        for a1 in range(-kmax,kmax+1):
            for a2 in range(-kmax,kmax+1):
                if a0==a1==a2==0:
                    continue
                #wave vector 
                k=a0*Kcell[0]+a1*Kcell[1]+a2*Kcell[2]
                #square of the wavevector
                ksq = np.dot(k,k)
                #variable fot the complex exponential term
                exponential = 0.0+0.0j
                for i in ions:
                    #for j in ions:
                    #    if (i.index)==(j.index):
                    #        continue
                    #position of the particle
                    r_i = i.position.copy()  # -j.position.copy()
                    #charge of the particle
                    charge = i.charge.copy()# *j.charge.copy()
                    #charge times the ecponential term
                    exponential += -charge*np.exp(np.dot(1j*k,r_i))
                C=np.exp(-ksq/(4.0*alpha*alpha))/ksq
                addition =C*(np.real(np.conj(exponential)*(exponential)))
                #print(addition)
                Sum +=addition
    
    return twopi*Sum/(ions.get_volume())
def ResSum(ions, kmax, alpha, cutoff):
    kcutsq=kmax
    Kcell = ions.get_reciprocal_cell()*twopi
    cell = ions.get_cell()
    u1=0
    u2=0
    cutoffsq=cutoff*cutoff
    #loop trough the ion pairs.
    for i in ions:
        for j in ions:
            #distance between ions
            r_ij=j.position.copy()-i.position.copy()
            for a0 in range(-kmax,kmax+1):
               for a1 in range(-kmax,kmax+1):
                   for a2 in range(-kmax,kmax+1):
                       #RESIPROCAL SUM
                       if a0==a1==a2==0:
                           continue
                       #wave vector 
                       k=a0*Kcell[0]+a1*Kcell[1]+a2*Kcell[2]
                       #square of the wavevector
                       ksq = np.dot(k,k)
                       if ksq < cutoffsq:
                           u1 += (i.charge*j.charge)/ksq*np.exp(-ksq/(4.0*alpha*alpha))*np.cos(np.dot(k,r_ij))
                           #continue
                       ########################33
                       #REAL SPACE
                       #skip the calculation for ions intercation on its self
                       if i==j and a0==a1==a2==0: 
                           continue
                       else:
                           #location of the unit cell
                           a=a0*cell[0]+a1*cell[1]+a2*cell[2]
                           if np.dot(a,a) < cutoffsq:
                               #location of the particle
                               r=j.position.copy()-(i.position.copy()+a)
                               #check if the particle is in the range of cutoff
                               dsq=np.dot(r,r)
                               #if dsq > cutoffsq:
                               #    continue
                               #distance of the particles i and j
                               d=sqrt(dsq)
                               #addition of charge_i*charge_j*erfc(a*d)/(d)
                               u2 = (i.charge.copy()*j.charge.copy())*erfc(alpha*d)/(d)
                          
    return twopi*u1/ions.get_volume()+u2/2.0
def PointEnergy(ions):
    u=0.0
    for ion in ions:
        u = u - (ion.charge.copy()*ion.charge.copy())
    return u
def nearestNeighbourDistance(atoms):
    ############################
    #NEAREST NEIGHBOUR DISTANCE#
    ############################
    """
    COUNTS THE DISTANCE TO THE NEAREST NEIGHBOUR IN THE CRYSTAL
    """
    
    distances = (atoms.get_all_distances(mic=True))
    #print(distances)
    min = distances[0][1]
    for distance in distances[0]:
        if distance > 0 and distance < min:
            min = distance
    return min
def saveToFile(filename,mc,alphas):
    f = open('output/'+filename, 'w+')
    for i in range(len(alphas)):
        f.write(str(alphas[i])+' & '+str(mc[i]) +'\n')
    f.close()
def madelung(args):
    ###CONSTANTS###
    
    ###VARIABLES###
    minim =0.0
    charge = []
    numberOfMolecules= 1
    #atoms
    positonOfInterest = 0
    alpha = 0.0
    cutoffs=[3] #list of cutoff values to loop 
    alphas= np.linspace(0.01,3,20) #range(20,100,5) #[3,4,6] #list of alpha ewald splitting values to loop
    aCharge = 0 #charge of the anion
    cCharge = 0 #charge of the cation
    mc = 0.0 #for single madelung constant value
    MC = [] #to list all the madelung constant values
    
    #parse the inputs given by user
    print(args.input)
    print('Reading input data...')
    atoms = read(args.input)
    minim = nearestNeighbourDistance(atoms)
    if args.silent:
        print ('Atoms:\n',atoms.get_chemical_symbols())
        print ('Unit cell: \n',atoms.get_cell(),'\n Atom positions: \n',atoms.get_positions())
        print ('PBC: \n', atoms.get_pbc())
        print('Normalizasion parameter: ',minim)
    
    
    
    #handel with charges given by the user
    if args.charges is not '':
        if args.silent:
            print('Loading charges file...')
        #use the charges file given by the user
        with open(args.charges) as r:
            for line in r:
                
                if line[0]=='#':
                    continue
                charge.append(float(line))
    else:
        for x in atoms.get_initial_charges():
            charge.append(x)
    
    #add the charges to the atoms object
    atoms.set_initial_charges(charge)
    if args.silent:
        print('Charges: \n',atoms.get_initial_charges())
    
    #choose the atom you want to count the madelung constant for
    if args.position is not '':
        if args.silent:
            print('Loading the atom of interest...')
        #set positionOfInterest to be the one given by the user
        positonOfInterest = int(args.position)
        if args.silent:
            print('Atom of interest: ', atoms.get_chemical_symbols()[positonOfInterest])
        
    #check the number of molecules
    if args.number is not '1':
        numberOfMolecules = int(args.number)
        if args.silent:
            print('Number of molecules in given file is: ',numberOfMolecules)
    #check ion charges given by user
    #for anions
    if args.anion is not '':
        aCharge = int(args.anion)
        if args.silent:
            print('Anion charge: ',aCharge)
    else:
        aCharge = charge[0]
    #for cations
    if args.cation is not '':
        cCharge = int(args.cation)
        if args.silent:
            print('Cation charge: ',cCharge)
    else:
        cCharge = charge[-1]

    print('Done!')
    
    
    
    for cutoff in cutoffs:
        #charge distribution parameter for ewald splitting
        #alphal=8.0
        for alphal in alphas:
            alpha=alphal/minim
            #caulculate real space contributions qpe and spe
            
            print('Counting real space contribution...')
            spe = RealConstant(atoms, alpha) #selfInteraction
            
            qpe = RealSum(atoms, alpha, cutoff) #realSpaceSum
            
            print('Done!')
            #calculate reciprocal space contribution epe
            print('Counting reciprocal space contribution...')
            epe = ResiprocalSum(atoms, 4, alpha)
            print('Done!')
            #Total Energy
            tpe=(qpe/2.0+epe+spe)*minim #/2  #(-8.0) #(qpe-epe-spe)*minim 
            print('Results for the MC at alpha value: ', alpha)
            print ('Self interaction:', spe)
            print ('ERF: ', qpe)
            print('Fourier: ', epe)
            print('Total energy: ', tpe)
            mc=tpe/numberOfMolecules/aCharge/cCharge
            print('Madelung constant: ', mc)
            
            """
            U1=(ResSum(atoms, 4, alpha,50))
            U2=PointEnergy(atoms)*alpha/sqrtpi
            ressum = (U1+U2)*minim/(aCharge*cCharge*numberOfMolecules)
            print('ressum: ',ressum)
            """
            MC.append(mc)
    plt.plot(alphas, MC)
    plt.savefig('pics/'+args.input)
    saveToFile(args.input,MC,alphas)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='path of the inputfile. Format can be e.g. POSCAR, .cif' )
    parser.add_argument('-c','--charges', help='file containing list of the ion charge values', default='')
    parser.add_argument('-p','--position', help='give the index of the particle you want to count the MC for.', default='')
    parser.add_argument('-n','--number', help='give the number of molecules in the given structure file, default is 1', default='1')
    parser.add_argument('-a','--anion', help='give anion charge, default first number in charge file', default='')
    parser.add_argument('-k','--cation', help='give cation charge, default last number in charge file', default='')
    parser.add_argument('--silent',help='use -s or --silent to not show info of the loaded structure', dest='silent', action='store_false', default=True)
    
    args =parser.parse_args()
    madelung(args)