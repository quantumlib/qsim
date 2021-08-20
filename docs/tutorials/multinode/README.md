
# Multinode quantum simulation using HTCondor on GCP
This tutorial will take you through the process of running many qsim simulations
on Google Cloud. In some situations, it is required to run many instances of the same 
simulation. This could be used to provide a parameter sweep or to evaluation noise characteristics.

## Objectives

* Use Terraform to deploy a HTCondor cluster
* Run a multinode simulation using HTCondor
* Query cluster information and monitor running jobs in HTCondor


## Create a project
In a seperate tutorial the method to create Google Cloud project is explained in detail. 
[Please refer to this tutorial for guidance](https://quantumai.google/qsim/tutorials/qsimcirq_gcp#gcp_setup).

When you have a project created, move on to the next step.

## Configure your enviroment

In your project using the 
[Cloud Shell](https://console.cloud.google.com/home/dashboard?cloudshell=true), clone this Github repo.
```
git clone https://github.com/jrossthomson/qsim.git
```

Change directory to the tutorial:
```
cd qsim/docs/tutorials/multinode/terraform
```
This is where you will use terraform to create the HTCondor cluster required to run your jobs.

### Edit init file to setup environment

You can now edit `init.sh` to change the name of the project you are using. You can also change the zone and region as you see fit. More information is available [here](https://cloud.google.com/compute/docs/regions-zones).

Change the variables to reflect your project, most important is the first line:
```
export TF_VAR_project=my-quantum-htcondor-project
```
The project id needs to be globally unique and to be the same as the project you just created.

The edited `init.sh` file should be "sourced" in the cloud shell:

```
source init.sh
```
Repsond in the affirmative to any pop-ups that request permissions on the Cloud platform.

The final outcome of this script will include:

* A gcloud config setup correctly
* A service account created
* The appropriate permissions assigned to the service account
* A key file created to enable the use of Google Cloud automation.

This will take about 60 seconds. At the end you will see output about permissions and the configuration of the account.

## Run terraform

When this is complete, you can initialize teraform to begin your cluster creations:
```
terraform init
```
A correct result will contain:
```
Terraform has been successfully initialized!
```
Some terraform commands are wrapped in a makefile, so you can now create your cluster:
```
make apply
```
A successful run will show:
```
Apply complete! Resources: 4 added, 0 changed, 0 destroyed.
```
## Connect to the Submit node for HTCondor
You will now be able list the VMs created:
```
gcloud compute instances list
```
You will log in to the `submit` node:
```
gcloud compute ssh c-submit
```
Now you are logged in to your HTCondor cluster.

### Checking the status 
You can verify if the HTCondor install is completed:
```
condor_q
```
You will see output:
```
-- Schedd: c-submit.c.quantum-htcondor-14.internal : <10.150.0.2:9618?... @ 08/18/21 18:37:50
OWNER BATCH_NAME      SUBMITTED   DONE   RUN    IDLE   HOLD  TOTAL JOB_IDS

Total for query: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended
Total for drj: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended
Total for all users: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended
```
If you get `command not found`, you will need to wait a few minutes for the HTCondor install to complete.

## Clone the repo on your cluster

Again on the `submit` node you you can install the repo to get access to previously created submission files:
```
git clone https://github.com/jrossthomson/qsim.git
```
Then cd to the tutorial directory.
```
cd qsim/docs/tutorials/multinode
```
Now it is possible to submit a job:
```
condor_submit circuit_q24.sub
```
If successful, the output will be:
```
Submitting job(s).
1 job(s) submitted to cluster 1.
```
This may take a few minutes, but when completed the command:
```
ls out
```
will list the files:
```
err.1-0  log.1-0  out.1-0  placeholder
```
You can also see the progress of the job throught the log file:
```
tail -f out/log.1-0
```
After the job s completed, the ouput of the job can be seen:
```
cat out/out.1-0
```
## Running noisy simulations
To run multiple simulations, you can run the submit file `noise.sub`:
```
universe                = docker
docker_image            = gcr.io/quantum-builds/github.com/quantumlib/jupyter_qsim:latest
arguments               = python3 noise3.py
should_transfer_files   = YES
transfer_input_files    = noise3.py
when_to_transfer_output = ON_EXIT
output                  = out/out.$(Cluster)-$(Process)
error                   = out/err.$(Cluster)-$(Process)
log                     = out/log.$(Cluster)-$(Process)
request_memory          = 10GB
queue 50
```
The final line in this submit file has `queue 50`. This means 50 instances of this simulation will be run. The job can be submitted with the `condor_submit` command.
```
condor_submit noise.sub
```
The output will look as follows:
```
Submitting job(s)..................................................
50 job(s) submitted to cluster 2.
```
If this is the second _submit_ you have run, you can see the output of the all the simualtions. The output will be in the `out` directory. 
```
cat out/out.2-*
```
You can see the results of the simulations.
```
Counter({3: 462, 0: 452, 2: 50, 1: 36})
Counter({0: 475, 3: 435, 1: 49, 2: 41})
Counter({0: 450, 3: 440, 1: 59, 2: 51})
Counter({0: 459, 3: 453, 2: 51, 1: 37})
Counter({3: 471, 0: 450, 2: 46, 1: 33})
Counter({3: 467, 0: 441, 1: 54, 2: 38})
Counter({3: 455, 0: 455, 1: 50, 2: 40})
Counter({3: 466, 0: 442, 2: 51, 1: 41})
.
.
.
```
Note that because this is a noise driven circuit, the results of each simulation are different.

To run your own simulations, simply create a noisy circuit in your _qsim_ python file.

There are many more examples of circuits to be run [here](https://quantumai.google/cirq/noise)

## Shutting down
When you are done with this tutorial, it is important to remove the resources. You can do this with _terraform_
```
make destroy
```
> The most effective way to ensure you are not charged is to delete the project. [The instructions are here.](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects)



