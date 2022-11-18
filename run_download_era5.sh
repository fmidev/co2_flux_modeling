#!/bin/sh
#

# Download ERA-5


code_dir=/users/kamarain/ATMDP-003



#mkdir -p ERA-5_1p0deg/
#cd ERA-5_1p0deg/

#mkdir -p ERA-5_0p25deg/
#cd ERA-5_0p25deg/

mkdir -p /fmi/scratch/project_2002138/ERA-5_0p25deg/
cd /fmi/scratch/project_2002138/ERA-5_0p25deg/



declare -a vars=('pmsl' 'te2m' 'snw' 'prec' 'smo' 'tclc' 'v10m' 'u10m' 'evap' 'sswf' 'slhf' 'sshf' 'ste') 
for var in "${vars[@]}"
do
   echo $var
   python $code_dir/download_era5_sfc_from_ecmwf.py $var &
done



declare -a vars=('rh1000' 'z150')
for var in "${vars[@]}"
do
   echo $var
   python $code_dir/download_era5_pl_from_ecmwf.py $var &
done
wait
