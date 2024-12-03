#!/bin/sh
#SBATCH --time=3-00:00:00
#SBATCH --job-name=multi_agent_3
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tmxfean@u.nus.edu
#SBATCH --cpus-per-task=100
#SBATCH --mem=100G
#SBATCH --partition=long
#SBATCH --exclude=xcng1

TMPDIR = 'mktemp -d'
cp ~/jobdata/* $TMPDIR

srun multi-agent-1.py $TMPDIR
