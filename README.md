# ML-FF
Machine learning from Cremer Pople sampling of the Potential Energy Surface of flexible cyclic moieties (https://pubs.acs.org/doi/10.1021/acs.jpca.3c00095). 

Installation:
conda env create -f environment.yaml

Run:
conda activate mlff_things
sbatch run.sh 


Scripts and Functions:

- run.py:	
	* Loads and stacks the formatted input (bond length, bond angle, proper dihedral, improper dihedral) and output (Energy, Force) from the QM folder to the NN
	* Splits and randomizes the datasets
	* Customizes the NN
		- user-defined loss function supported choices: MSE and WMSE
		- user-defined number of epochs
		- user-defined tolerance for the R2 at the training stage and convergence
	* Runs the NN
		- optional plot output of the learning curve and regression
		- on-screen output for R2, MAE evaluation 

- load_data.py: Wraps and formats data from read_log and geometry_builder. This includes:
	
	* Unit conversions from QM to FF units for energy and force
	* Normalizations for energy, bonds, angles, dihedrals

- read_log.py: Reads a series of G16 output files (.log format). This script outputs the following for each data point:
	
	* Energy [Hartree]: list of QM data length.
	* Forces [Hartree/Bohr]: nested lists with QM data length. The inside lists have NoA length and correspond to the forces applied on each atom. Separate nested lists for Fx, Fy, and Fz.     
	* Coordinates [Angs] : pandas dataframe. Each row is a QM data point. Columns include Atom indices, Atom Number, X[Angs]/Y[[Angs]]/Z[Angs] in lists of NoA length


- geometry_builder.py: Evaluates bond connectivity and calls geometry_calculator.py to calculate internal coordinates. It includes:
	
	* eval_bonds: Here connectivity is evaluated from a single QM datapoint. This assumes that the data are clean and contain no broken structures. Two atoms are considered bonded if their distance is 
	within +/-10% of their equilibrium bond. 
	=> input: Coordinate data frame, number of ring atoms (N)
	=> output: Connectivity, bonds for all QM data

	* ()_find_calc: Uses connectivity to find (find function) and calculate (calculate function) angles and dihedrals. Calculations are done with the geometry_calculator script.
	=> input: Connectivity, Coordinate data frame
	=> output: list of atoms with angles/dihedrals, angles/dihedrals for all QM data
 
	* checker: Checks if the number of bonds/angles/dihedrals/etc that were is a multiple of the number of QM data. Indicatevely for cycloalkanes, the total number of internal geometry  
	contributions should be NoA*QMdata (for bonds), 2*NoA*QMdata (for angles), 3*NoA*QMdata (for dihedrals).
	=> input: bonds/angles/dihedrals, coordinate data frame, name of contribution
	=> output: True or False with indicators

	
Tools:

- geometry_calculator : caculates angles and dihedrals
- nn_structure.py : Neural Network architecture
- postprocessing.py : Plots the learning curve and regression

