# conda:
conda activate fenicsproject
conda deactivate
#### conda-fenics is said to be incomplete; abandoned

# GCP machine 
ssh fenics.us-east1-b.deeplearning201906

## configuration
gcloud compute config-ssh

# run from docker (choose one):
fenicsproject start evergreen -i
fenicsproject start notebook
fenicsproject start notebook -i
sudo /home/evergreen/.local/bin/fenicsproject start notebook  -i

## for the first time:
fenicsproject create evergreen
or:
fenicsproject run stable:current
or:
fenicsproject run stable:current python3 file.py (can't output)

# notebook
##background
docker create --name notebook -w /home/fenics -v $(pwd):/home/fenics/shared -d -p 127.0.0.1:8888:8888 quay.io/fenicsproject/stable:current 'jupyter-notebook --ip=0.0.0.0'
## foreground
docker create --name notebook -w /home/fenics -v $(pwd):/home/fenics/shared -p 127.0.0.1:8888:8888 quay.io/fenicsproject/stable:current 'jupyter-notebook --ip=0.0.0.0'
docker start notebook 
docker stop notebook 
docker ps -a
dock rm [id]
# alternative for access to container
jupyter notebook --ip 0.0.0.0

ssh fuch@cluster.ceremade.dauphine.fr
# Password: Fuch19()


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
./send_from_local.sh
## with scp
```
scp username@remote:/file/to/send /where/to/put
scp fuch@cluster.ceremade.dauphine.fr:~/GitHub/classification/code/temp.png ~/Desktop
```

docker start evergreen
docker exec evergreen /bin/sh -c "python3 NS_backflow_server.py"
docker stop evergreen

# on the server
cd GitHub/classification/code
qsub test.pbs
qsub -I
# show jobs
pbstop
qstat
# delete job
qdel #ID

# manage Docker machine with portainer:
docker run -d -p 9000:9000 -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer

## then visit http://0.0.0.0:9000

## note 
ansible clust,erc -b -K -a "pip3 uninstall -y fenics"

# Notebook
sshuttle -r fuch@www.ceremade.dauphine.fr 193.48.71.0/25 -D
## then qsub -I -l nodes=clust6:ppn=2
## then jupyter notebook --ip=0.0.0.0
## then visit clust6

# VPN
sshuttle -r fuch@www.ceremade.dauphine.fr 0.0.0.0/0


# code_saturn by edf (CFD)
docker pull sebastienhoarau/code_saturne

# debug encoding
sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/ /g' run_multiple_submit.sh > run_multiple_submit_bak.sh

# send file from remote to local with scp
scp fuch@cluster.ceremade.dauphine.fr:~/code/*.sh ./download/
*

