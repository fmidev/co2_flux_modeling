#!/bin/bash -l
# 
#SBATCH --job-name=BayesOpt
#SBATCH --account=project_2002138 # project_2002138 # project_2001635 # project_2002597
#SBATCH --time=48:00:00 # 100:00:00 # 06:00:00 # 75:00:00
#SBATCH --partition=fmi # large # small # fmi
#SBATCH --mem=185G 
#SBATCH -N 4 
#SBATCH -n 4
#SBATCH -c 40  

# sbatch run_SLURM_optim.sh 

# Prepare the Miniconda environment
export "PATH=/fmi/projappl/project_2002138/miniconda/bin:$PATH"



module load gcc/9.1.0
export OMP_NUM_THREADS=1







code_dir='/users/kamarain/ATMDP-003'



data_dir='/fmi/scratch/project_2002138/ATMDP-003'
rslt_dir='/fmi/scratch/project_2002138/ATMDP-003'

era5_dir='/fmi/scratch/project_2002138/ERA-5_0p25deg'

optimize=True



# Start the stopwatch
start=`date +%s`


echo "Optimizing/fitting!"


srun -n 1 -N 1 -c 40 --exclusive python $code_dir/fit_optim.py $code_dir $data_dir $rslt_dir 'GB' True &
srun -n 1 -N 1 -c 40 --exclusive python $code_dir/fit_optim.py $code_dir $data_dir $rslt_dir 'RF' True &
srun -n 1 -N 1 -c 40 --exclusive python $code_dir/fit_optim.py $code_dir $data_dir $rslt_dir 'GB' False &
srun -n 1 -N 1 -c 40 --exclusive python $code_dir/fit_optim.py $code_dir $data_dir $rslt_dir 'RF' False &
wait

echo "Finished optimizing/fitting!"

end=`date +%s`; runtime=`date -d@$((end-start)) -u +"%H:%M:%S"`
echo "Total run time: "$runtime

echo "Finished training!"
