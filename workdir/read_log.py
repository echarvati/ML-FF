import os
import pandas as pd

def get_structure_forces(contents,flg, NoA):

# locate line before and now read information
    contribution=[]
    start = flg+5
    stop = start+NoA
    for i in contents[start:stop]:
        contribution.append(i.split())

    return contribution


def read_logs(datafile):

    os.chdir(datafile)

    #get structure (atom indices, atom numbers, coordinates),energy,force for all QM data
    SCF_list = []
    Forces_X = []
    Forces_Y = []
    Forces_Z = []
    logs = []
    Energies = []
    AtomIndx = []
    AtomNo = []
    X = []
    Y = []
    Z = []

    #log file end signal
    end_signal = 'Normal termination'

    for file in os.listdir(os.getcwd()):
        if file.endswith(".log"):
            with open(file, "r") as f:
                contents = f.read()
                if contents.find(end_signal) != -1:
                    logs.append(file)


    for log in logs:
        with open(log, "r") as l:
            coord_pt = []
            force_pt = []

            contents = l.readlines()
            for i, line in enumerate(contents):

                if line.startswith(' NAtoms='):
                    NoA = int(line.strip().split()[1])

                elif line.startswith(' SCF Done:'):
                    SCF_list.append(float(line.strip().split()[4]))

                elif line.startswith('                         Standard orientation:                     '):
                    coord_pt.append(i)

                elif line.startswith(' ***** Axes restored to original set *****'):
                    force_pt.append(i)


            #Structures
            ### Structure by data point
            structure_pt = get_structure_forces(contents, int(coord_pt[-1]), NoA)
            point_df = pd.DataFrame(structure_pt,
                                        columns=['Atom index', 'Atom label', 'Atom Type', 'X[Angs]', 'Y[Angs]', 'Z[Angs]'])

            ### Gather QM data structures
            AtomIndx.append(point_df['Atom index'].astype(int).to_list())
            AtomNo.append(point_df['Atom label'].astype(int).to_list())
            X.append(point_df['X[Angs]'].astype(float).to_list())
            Y.append(point_df['Y[Angs]'].astype(float).to_list())
            Z.append(point_df['Z[Angs]'].astype(float).to_list())

            #Energies
            Energies.append(SCF_list[-1])

            #Forces
            ### Forces by data point
            Force_pt = get_structure_forces(contents, int(force_pt[-1]), NoA)
            F_df = pd.DataFrame(Force_pt, columns=['Atom index', 'Atom label', 'FX', 'FY', 'FZ'])

            ### Gather QM data forces
            Fx_pt = F_df['FX'].astype(float).to_list()
            Fy_pt = F_df['FY'].astype(float).to_list()
            Fz_pt = F_df['FZ'].astype(float).to_list()
            Forces_X.append(Fx_pt)
            Forces_Y.append(Fy_pt)
            Forces_Z.append(Fz_pt)

    Coords_df = pd.DataFrame(list(zip(AtomIndx, AtomNo, X, Y, Z)), columns=['Atom Indx', 'Atom No.','X[Angs]', 'Y[Angs]', 'Z[Angs]'])


    #Check
    print('Found', len(logs), 'QM data points')

    if len(logs) == len(Energies) and len(logs) == len(Forces_X) and len(logs) == len(Forces_Y) and len(logs) == len(Forces_Z) and len(logs) == len(Coords_df):
        print('All QM points read succesfully')

    elif len(logs) != len(Energies):
        print('Error in Energies:', 'QM data = ', len(logs), 'Energies = ', len(Energies))

    elif len(logs) != len(Forces_X) and len(Forces_X)==len(Forces_Y) and len(Forces_X)==len(Forces_Z):
        print('Error in Forces:', 'QM data = ', len(logs), 'Forces = ', len(Forces_X))

    elif len(logs) != len(Coords_df):
        print('Error in Structures:', 'QM data = ', len(logs), 'Structures = ', len(Coords_df))

    return NoA, Energies, Forces_X, Forces_Y, Forces_Z, Coords_df
