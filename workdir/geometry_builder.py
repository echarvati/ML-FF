import numpy as np
import math
import matplotlib.pyplot as plt
from math import dist
import pandas as pd

from geometry_calculator import calc_angle, calc_dh


def isMultiple(num,  check_with):
	return num % check_with == 0


def get_atom_bonds(connectivity):

#turn connectivity which show bonds(for example [[1,2],[1,3]]) into atom index(for example {1:[2,3]})
    atom_bonds = {}

    for bond in connectivity:
        atom1, atom2 = bond
        if atom1 not in atom_bonds:
            atom_bonds[atom1] = [atom2]
        else:
            atom_bonds[atom1].append(atom2)
        if atom2 not in atom_bonds:
            atom_bonds[atom2] = [atom1]
        else:
            atom_bonds[atom2].append(atom1)

    return atom_bonds

def checker(contr, Coords_df, contribution = 'contribution'):

    if (isMultiple(len(contr), len(Coords_df)) == True):
        print('All %s counted' %contribution)
    else:
        print('WARNING: %s counting is incorrect!' %contribution)


def find_angle(Connectivity):

    atom_connectivity=get_atom_bonds(Connectivity)
    angle_list=[]

    for i in range(0,len(atom_connectivity)):
        if len(atom_connectivity[i])>1:
            for m in range(0, len(atom_connectivity[i])):
                for n in range(m+1,len(atom_connectivity[i])):
                    angle_list.append([i,atom_connectivity[i][m],atom_connectivity[i][n]])

    return angle_list

def find_dihedral(Connectivity):

    atom_connectivity = get_atom_bonds(Connectivity)
    dihedral_list=[]

    for bond in Connectivity:
        atom1, atom2 = bond
        if len(atom_connectivity[atom1])>1 or len(atom_connectivity[atom2])>1:
            for m in range(len(atom_connectivity[atom1])):
                for n in range(len(atom_connectivity[atom2])):
                    if atom_connectivity[atom1][m]!=atom2 and atom_connectivity[atom2][n]!=atom1:
                        dihedral_list.append([atom1,atom2,atom_connectivity[atom1][m],atom_connectivity[atom2][n]])

    return dihedral_list


def dihedrals_find_calc(Connectivity, Coords_df):

    dihedral_list = find_dihedral(Connectivity)

    dhs = []

    for pt in range(len(Coords_df)):
        for dih in dihedral_list:

            X = Coords_df['X[Angs]'][pt]
            Y = Coords_df['Y[Angs]'][pt]
            Z = Coords_df['Z[Angs]'][pt]

            coords_a= np.array([X[dih[0]],  Y[dih[0]], Z[dih[0]]])
            coords_b = np.array([X[dih[1]], Y[dih[1]], Z[dih[1]]])
            coords_c = np.array([X[dih[2]], Y[dih[2]], Z[dih[2]]])
            coords_d = np.array([X[dih[3]], Y[dih[3]], Z[dih[3]]])
            dihedral = calc_dh(coords_a, coords_b, coords_c, coords_d)
            dhs.append(dihedral)

    checker(dhs, Connectivity, contribution="dihedrals")

    return dihedral_list, dhs


def angles_find_calc(Connectivity, Coords_df):

    angle_list = find_angle(Connectivity)

    angles = []

    #Calculate angles for the dataset
    for pt in range(len(Coords_df)):
        for ang in angle_list:

            X = Coords_df['X[Angs]'][pt]
            Y = Coords_df['Y[Angs]'][pt]
            Z = Coords_df['Z[Angs]'][pt]

            coords_0= np.array([X[ang[0]], Y[ang[0]], Z[ang[0]]])
            coords_1 = np.array([X[ang[1]], Y[ang[1]], Z[ang[1]]])
            coords_2= np.array([X[ang[2]], Y[ang[2]], Z[ang[2]]])
            angle = calc_angle(coords_0, coords_1, coords_2)

            angles.append(angle)

    checker(angles, Connectivity, contribution="bond-angles")

    return angle_list, angles


##TODO: make independent of number of ring atoms (N)
##TODO: more robust connectivity counting
def eval_bonds(Coords_df, N):

    # Bond limits
    r0_CC = 1.54000  # Angstrom
    r0_CC_max = r0_CC + 10 / 100 * r0_CC
    r0_CC_min = r0_CC - 10 / 100 * r0_CC

    r0_OC = 1.43000
    r0_OC_max = r0_OC + 10 / 100 * r0_OC
    r0_OC_min = r0_OC - 10 / 100 * r0_OC

    r0_OC_double = 1.163
    r0_OC_double_max = r0_OC_double + 2 / 100 * r0_OC_double
    r0_OC_double_min = r0_OC_double - 2 / 100 * r0_OC_double

    r0_CN = 1.47500
    r0_CN_max = r0_CN + 10 / 100 * r0_CN
    r0_CN_min = r0_CN - 10 / 100 * r0_CN

    r0_CS = 1.8571
    r0_CS_max = r0_CS + 10 / 100 * r0_CS
    r0_CS_min = r0_CS - 10 / 100 * r0_CS

    r0_OS = 1.72
    r0_OS_max = r0_CS + 10 / 100 * r0_CS
    r0_OS_min = r0_CS - 10 / 100 * r0_CS

    r0_CH = 1.09
    r0_CH_max = r0_CH + 5.0 / 100 * r0_CH
    r0_CH_min = r0_CH - 5.0 / 100 * r0_CH

    r0_NH = 1.00
    r0_NH_max = r0_NH + 10 / 100 * r0_NH
    r0_NH_min = r0_NH - 10 / 100 * r0_NH

    INDX = Coords_df['Atom Indx'][0]
    AtomNumber = Coords_df['Atom No.'][0]

    bonds = []
    Connectivity = []

    #Find coonectivity from one structure. This assumes that the data are clean and without any broken structures
    for i in range(len(INDX)):

        for j in range(int(N)):

            X0 = Coords_df['X[Angs]'][0]
            Y0 = Coords_df['Y[Angs]'][0]
            Z0 = Coords_df['Z[Angs]'][0]

            coords_i = np.array([X0[i], Y0[i], Z0[i]])
            coords_j = np.array([X0[j], Y0[j], Z0[j]])
            r = dist(coords_i, coords_j)


            if AtomNumber[j] == 6 and AtomNumber[i] == 6 and r >= r0_CC_min and r <= r0_CC_max:
                if [i, j] not in Connectivity and [j, i] not in Connectivity:
                    Connectivity.append([i, j])

            if AtomNumber[j] == 6 and AtomNumber[i] == 8 and r >= r0_OC_min and r <= r0_OC_max:
                if [i, j] not in Connectivity and [j, i] not in Connectivity:
                    Connectivity.append([i, j])

            if AtomNumber[j] == 6 and AtomNumber[i] == 8 and r >= r0_OC_double_min and r <= r0_OC_double_max:
                if [i, j] not in Connectivity and [j, i] not in Connectivity:
                    Connectivity.append([i, j])

            if AtomNumber[j] == 6 and AtomNumber[i] == 16 and r >= r0_CS_min and r <= r0_CS_max:
                if [i, j] not in Connectivity and [j, i] not in Connectivity:
                    Connectivity.append([i, j])

            if AtomNumber[j] == 8 and AtomNumber[i] == 16 and r >= r0_OS_min and r <= r0_OS_max:
                if [i, j] not in Connectivity and [j, i] not in Connectivity:
                    Connectivity.append([i, j])

            if AtomNumber[j] == 6 and AtomNumber[i] == 7 and r >= r0_CN_min and r <= r0_CN_max:
                if [i, j] not in Connectivity and [j, i] not in Connectivity:
                    Connectivity.append([i, j])


            if AtomNumber[j] == 6 and AtomNumber[i] == 1 and r >= r0_CH_min and r <= r0_CH_max:
                if [i, j] not in Connectivity and [j, i] not in Connectivity:
                    Connectivity.append([i, j])

            if AtomNumber[j] == 7 and AtomNumber[i] == 1 and r >= r0_NH_min and r <= r0_NH_max:
                if [i, j] not in Connectivity and [j, i] not in Connectivity:
                    Connectivity.append([i, j])


    #Calculate bonds for each connectivity pair
    for pt in range(len(Coords_df)):
        for pair in Connectivity:

            X = Coords_df['X[Angs]'][pt]
            Y = Coords_df['Y[Angs]'][pt]
            Z = Coords_df['Z[Angs]'][pt]

            coords_p1 = np.array([X[int(pair[0])], Y[int(pair[0])], Z[int(pair[0])]])
            coords_p2 = np.array([X[int(pair[1])], Y[int(pair[1])], Z[int(pair[1])]])
            bl = dist(coords_p1, coords_p2)
            bonds.append(bl)


    checker(bonds, Coords_df, contribution="bond-lengths")

    return Connectivity, bonds
