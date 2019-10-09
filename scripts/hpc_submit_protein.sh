#!/bin/sh

#BSUB -q gpuv100

### -- set the job Name -- 
#BSUB -J XLNET

### -- ask for number of cores (default: 1) -- 
#BSUB -n 2

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=6GB]"

### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 12GB

### -- set walltime limit: hh:mm -- 
#BSUB -W 6:00

#BSUB -u s144471@student.dtu.dk

#BSUB -N

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o hpc_outputs/Output.out
#BSUB -e hpc_outputs/Error.err

module load cuda/9.0
module load cudnn/v7.0.5-prod-cuda-9.0
module load python3/3.6.2
module load tensorflow/1.12-gpu-python-3.6.2

pip3 install --user numpy
pip3 install --user absl-py

python3 run_train_gpu.py --config=param_configs/default-config.json
