# To run a multinode quantum simulation
s
## Objectives

* Use Terraform to deploy a HTCondor cluster
* Run a multinode simulation using HTCondor
* Query cluster information and monitor running jobs in HTCondor
* Autoscale nodes to runt jobs that new more Simulation capacit

## Costs

This tutorial uses the following billable components of Google Cloud:

* [Compute Engine](https://cloud.google.com/compute/all-pricing)

To generate a cost estimate based on your projected usage, use the [pricing calculator](https://cloud.google.com/products/calculator).
When you finish this tutorial, you can avoid continued billing by deleting the resources 
you created. For more information, see Cleaning up.

## Before you begin
1. In the Google Cloud Console, on the project selector page, select or create a Google Cloud project. To make this easy, try to choose a project where you are _Owner_ or _Editor_.

      > Note: If you don't plan to keep the resources that you create in this procedure, create a project instead of selecting an existing project. After you finish these steps, you can delete the project, removing all resources associated with the project.
Go to [project selector](https://console.cloud.google.com/projectselector2/home/dashboard)

1. Make sure that billing is enabled for your Cloud project. Learn how to confirm that billing is enabled for your project.
1. Enable the [Compute Engine and Deployment Manager APIs](https://console.cloud.google.com/flows/enableapi?apiid=compute,deploymentmanager.googleapis.com).
1. In the Cloud Console, [activate Cloud Shell](https://console.cloud.google.com/?cloudshell=true)

At the bottom of the Cloud Console, a Cloud Shell session starts and displays a command-line prompt. Cloud Shell is a shell environment with the Cloud SDK already installed, including the gcloud command-line tool, and with values already set for your current project. It can take a few seconds for the session to initialize.

## Setting up your environment in Cloud Shell
Select a region and zone for your cluster to run. [This guide can help](https://cloud.google.com/compute/docs/regions-zones).

Once selected, define the following environment variables in Cloud Shell.

```
export TF_VAR_project=[[YOUR_PROJECT_ID]]
export TF_VAR_region=[[YOUR_REGION]]
export TF_VAR_zone=[[YOUR_ZONE]]
```
With those variables defined, you can configure `gcloud`. Cut and paste the gcloud config commands in your Cloud Shell.

```
gcloud config set project $TF_VAR_project
gcloud config set compute/region $TF_VAR_region
gcloud config set compute/zone $TF_VAR_zone
```
With that completed, you can create a service account that will run the HTCondor cluster.

```
gcloud iam service-accounts create htcondor
```
Then give the service accounts the permissions that will allow HTCondor to run correctly. 
```
gcloud projects add-iam-policy-binding ${TF_VAR_project} --role roles/iam.serviceAccountUser \
 --member="serviceAccount:htcondor@${TF_VAR_project}.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding ${TF_VAR_project} --role roles/compute.admin \
--member="serviceAccount:htcondor@${TF_VAR_project}.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding ${TF_VAR_project} --role roles/monitoring.admin \
--member="serviceAccount:htcondor@${TF_VAR_project}.iam.gserviceaccount.com"
```

## Clone the repository and build the cluster

The `qsim` repository has all the code required to create the repository and to run the simulations.

1. Clone the repo and change to the working directory.
     ```
     git clone https://github.com/jrossthomson/qsim.git
     cd qsim/docs/tutorials/multinode/terraform
     ```

1. Build the cluster with Terraform
     ```
     terraform init
     make apply
     ```

This process will create 3 VMs instances and an autoscaling Managed Instance Group. To see the instances use the glcoud command in the Cloud Shell.
```
gcloud compute instances list
```
The output should be something like the output here:
```
NAME       ZONE        MACHINE_TYPE    PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP     STATUS
c-8xlg     us-east4-c  n2-standard-16               10.150.0.21  34.86.120.132   RUNNING
c-manager  us-east4-c  n2-standard-4                10.150.0.16  35.188.250.134  RUNNING
c-submit   us-east4-c  n2-standard-4                10.150.0.17  35.245.189.252  RUNNING
```

The Mangaged Instance Group can be seen with gcloud.
```
gcloud compute instance-groups list
```
And the output like the following.
```
NAME  LOCATION    SCOPE  NETWORK  MANAGED  INSTANCES
c     us-east4-c  zone   default  Yes      1
```
## Run a job on the cluster
The next step is to run a job on the HTCondor cluster. Jobs are submitted on the `c-submit` VM instance known as the Submit Node in HTCondor. You connect to the Submit Node using gcloud SSH.
```
gcloud compute ssh c-submit
```
There will now be a prompt with something like `username@c-submit`.

For convenience there is prepared HTCondor job submission files in the Github repo. To get these, 
clone the repository on the Submit Node, and change to the tutorial directory.
```
     git clone https://github.com/jrossthomson/qsim.git
     cd qsim/docs/tutorials/multinode
```

It is very likely that the HTCondor installation is not completely finished at this point: it takes several minutes. The `condor_status` command will tell you if the cluster is ready to run jobs. 

```
condor_status
```
The output from the command should resemble this.
```
Name                                           OpSys      Arch   State     Activity LoadAv Mem    ActvtyTime

slot1@c-6klg.c.test-quantum-multinode.internal LINUX      X86_64 Unclaimed Idle      0.000 64263  0+00:19:33

               Machines Owner Claimed Unclaimed Matched Preempting  Drain

  X86_64/LINUX        1     0       0         1       0          0      0

         Total        1     0       0         1       0          0      0
```

If  the output shows "Unclaimed" machines, you are ready to submit a job with HTCondor.

```
condor_submit circuit_q32.sub
```
There should be a response indicating the job was submitted.
```
Submitting job(s).
1 job(s) submitted to cluster 1.
```
Now you can see if the job is running correctly.
```
condor_q
```
The output will be similar to the snippet below.
```
-- Schedd: c-submit.c.test-quantum-multinode.internal : <10.150.0.6:9618?... @ 05/20/21 15:04:24
OWNER        BATCH_NAME    SUBMITTED   DONE   RUN    IDLE  TOTAL JOB_IDS
jrossthomson ID: 1        5/20 15:04      _      1      _      1 1.0

Total for query: 1 jobs; 0 completed, 0 removed, 0 idle, 1 running, 0 held, 0 suspended
Total for jrossthomson: 1 jobs; 0 completed, 0 removed, 0 idle, 1 running, 0 held, 0 suspended
Total for all users: 1 jobs; 0 completed, 0 removed, 0 idle, 1 running, 0 held, 0 suspended
```
When this is completed you should see output in the `out` directory.
```
ls out/
err.1-0  log.1-0  out.1-0
```
The contents of `out.1-0` will have the content you are expecting from the simulation. This will take about 10 minutes to be complete.
```
cat out/out.1-0
000:   -4.734056e-05   1.2809795e-05   2.4052194e-09
001:  -3.6258607e-06   2.3642724e-07   1.3202764e-11
010:  -2.9523137e-05    2.280164e-05   1.3915304e-09
011:  -1.3954962e-05   9.4717652e-06   2.8445529e-10
100:  -6.8555892e-06  -6.7632163e-07   4.7456514e-11
101:  -2.0390624e-05   3.1813841e-05   1.4278979e-09
110:   1.5711608e-05  -7.5214862e-06   3.0342737e-10
111:   9.2345472e-06  -2.7716227e-05   8.5346613e-10
```
## Sumitting many jobs

The job just submitted only ran a single instance of a qsim simulation. The main purpose of the present study is to
run many (up to thousands of) simulations to provide a broad study with statistical significance.

A simple way to submit many jobs is to use the `queue` command in the HTCondor. For this step edit the file `circuit_q30.sub`
in your favorite editor, such as vim or nano. You will see the full submission file.
```
universe                = docker/jross

docker_image            = gcr.io/quantum-builds/github.com/quantumlib/qsim
arguments               = -c circuit_q30
transfer_input_files    = ../../../circuits/circuit_q30
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
output                  = out/out.$(Cluster)-$(Process)
error                   = out/err.$(Cluster)-$(Process)
log                     = out/log.$(Cluster)-$(Process)
request_memory          = 10GB
queue 1
```
The submission file is running the public qsim `docker image` from [QuantumAI](https://quantumai.google/qsim). 
The command that is running utimately is running the `qsim` base executable.
```
qsim_base.x -c circuit_q30
```
This is achieved by running the _docker_image_ as listed with the `arguments` "-c circuit_q30". The 
circuit file `circuit_q30` is transferred to the compute nodes via the `transfer_input_files` command. 

> If you are interested
> in an introduction to HTCondor in general, there is a 
> [great introduction from CERN](https://indico.cern.ch/event/611296/contributions/2604376/attachments/1471164/2276521/TannenbaumT_UserTutorial.pdf). For details on job submission syntax there is a section of the [HTCondor Manual](https://htcondor.readthedocs.io/en/latest/users-manual/submitting-a-job.html) dedicated to this.

To expand the simulation to run 20 instances of the docker image, the change required is to modify the `queue` command.
```
queue 20
```
Now when you run the `condor_submit` command, 
```
condor_submit circuit_q32.sub
```
20 jobs will be visible from the `condor_q` command.

The really cool part is that now when you look at the list of VM instance you see that the system has 
automatically scaled up the number of VM machines to support the 20 jobs you have requested to run, or as we normally refer to it,
performs __autoscaling__.

```
gcloud compute instances list
NAME       ZONE        MACHINE_TYPE    PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP     STATUS
c-2q8w     us-east4-c  n2-standard-16               10.150.0.15  35.245.204.243  RUNNING
c-40dz     us-east4-c  n2-standard-16               10.150.0.11  34.86.33.206    RUNNING
c-4x4g     us-east4-c  n2-standard-16               10.150.0.14  35.245.201.254  RUNNING
c-6klg     us-east4-c  n2-standard-16               10.150.0.8   34.86.161.51    RUNNING
c-748q     us-east4-c  n2-standard-16               10.150.0.12  35.245.74.106   RUNNING
c-dgt0     us-east4-c  n2-standard-16               10.150.0.13  34.86.102.113   RUNNING
c-manager  us-east4-c  n2-standard-4                10.150.0.7   35.221.59.182   RUNNING
c-submit   us-east4-c  n2-standard-4                10.150.0.6   35.245.203.14   RUNNING
c-vdsz     us-east4-c  n2-standard-16               10.150.0.10  35.236.211.189  RUNNING
c-x4bd     us-east4-c  n2-standard-16               10.150.0.9   34.86.246.195   RUNNING
```

When all these 20 jobs are complete, you can see the output in the `out` directory.
```
ls out/out.*
out/out.2-0  out/out.2-10  out/out.2-12  out/out.2-14  out/out.2-16  out/out.2-18  out/out.2-2  out/out.2-4  out/out.2-6  out/out.2-8
out/out.2-1  out/out.2-11  out/out.2-12  out/out.2-15  out/out.2-17  out/out.2-19  out/out.2-2  out/out.2-5  out/out.2-7  out/out.2-9

```
The output of the simulation is in these files.

Finally, after the system has been idle for several minutes, you will see that the number of VM instances with autoscale 
back to a single Compute Node, the  Manager Node and the Submit node. This helps to control costs by removing VMs when
you do not need them.

## Next steps
Further work here will allow you to run multiple simulations for different work. There are several things to look at.

* Creating and running with your own container
* Running with multiple input files
* Selecting different configurations of the submit file

## Cleaning up
The easiest way to eliminate billing is to delete the Cloud project you created for the tutorial. Alternatively, 
you can delete the individual resources.

### Delete the project
> __Caution:__ Deleting a project has the following effects:
> Everything in the project is deleted. If you used an existing project for this tutorial, when you delete it, you also delete any other work you've done in the project.
> Custom project IDs are lost. When you created this project, you might have created a custom project ID that you want to use in the future. To preserve the URLs that use the project ID, such as an appspot.com URL, delete selected resources inside the project instead of deleting the whole project.

1. In the Cloud Console, go to the Manage resources page.
     > Go to [Manage resources](https://console.cloud.google.com/iam-admin/projects)

1. In the project list, select the project that you want to delete, and then click Delete.
1. In the dialog, type the project ID, and then click Shut down to delete the project.

### Delete the Slurm cluster
The second option is to delete the HTCondor cluster. In the `qsim/docs/tutorials/multinode/terraform` directory, run the make command.
```
     make destroy
```
This will remove the HTCondor cluster.

