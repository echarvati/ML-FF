#!/bin/bash
#SBATCH -J check_cpu
#SBATCH -o check_cpu.out
#SBATCH -e check_cpu.err
#SBATCH -p cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8

python run.py

