from geometry_builder import angles_find_calc, dihedrals_find_calc, eval_bonds
from read_log import read_logs
import numpy as np
import os
from math import dist
import pandas as pd
from sklearn import preprocessing

def norm(contribution):

    norm_contribution = 2 * (contribution - min(contribution)) / (max(contribution) - min(contribution)) - 1

    return norm_contribution

def analyse_logs(QMdata_file='filepath', work_file='workdir', N='N'):

    #Unit conersions
    kcalmol = float(627.5)  # Hartree to kcal/mol
    angstrom = float(0.529177249)  # Bohr to Angstrom
    f_units = kcalmol / angstrom


    #Fetch equilibrium bond lengths
    os.chdir(work_file)
    eqbonds_df = pd.read_csv('equilibrium_bonds.csv', sep=',')

    #Read all QM data
    os.chdir(QMdata_file)
    NoA, Energies, Forces_X, Forces_Y, Forces_Z, Coords_df = read_logs(QMdata_file)


    #Format energies and forces
    ###Energies
    #####Convert to relative Energy [kcal/mol] and normalize
    RelEnergy = (np.array(Energies) - min(np.array(Energies))) * kcalmol
    NormE = 2 * (RelEnergy - min(RelEnergy)) / (max(RelEnergy) - min(RelEnergy)) - 1

    # print(NormE, NormE.shape,len(NormE))

    ###Forces
    #####Convert to [kcal*mol-1*Angs-1]
    UnitForce_X=[]
    UnitForce_Y = []
    UnitForce_Z = []

    for f in range(len(Forces_X)):
        UnitForce_X.append(np.asarray(Forces_X[f])*f_units)
        UnitForce_Y.append(np.asarray(Forces_Y[f]) * f_units)
        UnitForce_Z.append(np.asarray(Forces_Z[f]) * f_units)

    #Find Connectivity and bond lenghts
    Connectivity, bonds = eval_bonds(Coords_df, N)
    ###Normalize and format bond-lengths
    NormBonds = np.split(norm(np.array(bonds)), len(Coords_df))

    #Find, calculate, and format angles
    angle_list, angles = angles_find_calc(Connectivity, Coords_df)
    ###Normalize and format bond-angles
    NormAngles = np.split(norm(np.array(angles)), len(Coords_df))

    #Find, calculate and format proper dihedrals
    dh_list, dihedrals = dihedrals_find_calc(Connectivity, Coords_df)
    NormDihedrals = np.split(norm(np.array(dihedrals)), len(Coords_df))

    return Connectivity, NormE, UnitForce_X, UnitForce_Y, UnitForce_Z, NormBonds, NormAngles, NormDihedrals