#!/bin/sh
#

# Download ERA-5






mkdir -p ERA-5_1p0deg/
cd ERA-5_1p0deg/





declare -a vars=('pmsl' 'te2m' 'snw' 'prec' 'smo' 'tclc' 'v10m' 'u10m' 'evap' 'sswf' 'slhf' 'sshf' 'ste') 
for var in "${vars[@]}"
do
   echo $var
   python ../download_era5_sfc_from_ecmwf.py $var &
done



declare -a vars=('rh1000' 'z150')
for var in "${vars[@]}"
do
   echo $var
   python ../download_era5_pl_from_ecmwf.py $var &
done
wait
