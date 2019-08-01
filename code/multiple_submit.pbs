#!/bin/bash
#PBS -N evergreen
#PBS -l nodes=clust7:ppn=39
#PBS -M changqing.fu@dauphine.eu
# bae = begin abort end
#PBS -m bae
#PBS -o log/$PBS_JOBID.o
#PBS -e log/$PBS_JOBID.e
# check output/error in real-time
#PBS -k oe 
# priority
#PBS -p 19
export PBS_NUM_PPN=39
current_load=0
max_load=40
current_s=-1
max_s=1
step_s=0.01
current_d=0
max_d=0.04
step_d=0.001
container_name=1

cd $PBS_O_WORKDIR
# printf "current dir: "
# docker ps -a
echo 0 > num_process.txt
printf "current dir: "
pwd
printf "ppn = "
echo $PBS_NUM_PPN
export OMP_NUM_THREADS=$PBS_NUM_PPN # So that OpenMP uses multithreads
printf "num_threads = "
echo $OMP_NUM_THREADS
# while ((num_process <= max_load))
while [ 1 ]
do
    num_process=`cat num_process.txt`
    printf "num_process = "
    echo $num_process
    if ((num_process > max_load))
    then
        sleep 1
        continue
    fi
    fenicsproject create NS${container_name}
    docker start NS${container_name}
    printf "computing parameter = (${current_s},${current_d})"
    cd ~/GitHub/classification/code
    docker exec -d NS${container_name} bash -c "multi_submit_run.sh ${current_s} ${current_d}"
    docker stop NS${container_name}
    docker rm NS${container_name} 
    # ((num_process+=1))
    ((container_name++))
    current_d=$(echo "$current_d + $step_d" | bc)
    if [ $current_d \> $max_d ]
    then
        current_d=0
        current_s=$(echo "$current_s + $step_s" | bc)
    fi
    if [ $current_s \> $max_s ]
    then
        break
    fi
    sleep 0.01
done

    
