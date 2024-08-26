#!/bin/bash
#PBS -q GPU-1A
#PBS -M s2416015@jaist.ac.jp
#PBS -m be

cd $PBS_O_WORKDIR
LOCAL_WORKDIR=/tmp/${USER}/${PBS_JOBID} 
mkdir -p ${LOCAL_WORKDIR} 
cp -r * ${LOCAL_WORKDIR} 
cd ${LOCAL_WORKDIR}
pip install -r requirements.txt
cd TaskA
./test_auto.sh
./eval_auto.sh

# copy back data 
cp -r results/ ${PBS_O_WORKDIR} 
cd ${PBS_O_WORKDIR} 

# cleanup 
rm -rf ${LOCAL_WORKDIR}
