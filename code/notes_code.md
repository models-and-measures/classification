# conda:
conda activate fenicsproject
conda deactivate
#### conda-fenics is said to be incomplete; abandoned

# run from dock:
fenicsproject run

# notebook
##background
docker run --name notebook -w /home/fenics -v $(pwd):/home/fenics/shared -d -p 127.0.0.1:8888:8888 quay.io/fenicsproject/stable 'jupyter-notebook --ip=0.0.0.0'
## foreground
docker run --name notebook -w /home/fenics -v $(pwd):/home/fenics/shared -p 127.0.0.1:8888:8888 quay.io/fenicsproject/stable 'jupyter-notebook --ip=0.0.0.0'
docker start notebook 
docker stop notebook 
docker ps -a
dock rm [id]
#alternative for access to container
jupyter notebook --ip 0.0.0.0

ssh fuch@cluster.ceremade.dauphine.fr

# clone code from git:
git clone https://github.com/models-and-measures/classification.git
## untrack in git
git rm --cached $FILE

## for the second time:
```
cd ~/GitHub
git init
git remote add origin https://github.com/models-and-measures/classification.git
git fetch origin
git checkout -b master # --track origin/master # origin/master is clone's default
```
## for the thrid time:
```
cd ~/GitHub
git init
git fetch origin
git checkout -b master # --track origin/master # origin/master is clone's default
```

# send files from local
## with rsync
./send_from_local
## with scp
```
scp username@remote:/file/to/send /where/to/put
scp fuch@cluster.ceremade.dauphine.fr:~/GitHub/classification/code/temp.png ~/Desktop
```

# on the server
cd GitHub/classification/code
qsub script.pbs

