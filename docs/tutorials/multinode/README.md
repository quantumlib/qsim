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
Then give the 
```
gcloud projects add-iam-policy-binding ${TF_VAR_project} --role roles/iam.serviceAccountUser \
 --member="serviceAccount:htcondor@${TF_VAR_project}.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding ${TF_VAR_project} --role roles/compute.admin \
--member="serviceAccount:htcondor@${TF_VAR_project}.iam.gserviceaccount.com"
```

## Clone the repository

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
sudo journalctl -f -u google-startup-scripts.service

condor_q
-- Schedd: c-submit.c.quantum-htcondor.internal : <10.150.15.206:9618?... @ 05/14/21 15:09:57
OWNER BATCH_NAME      SUBMITTED   DONE   RUN    IDLE   HOLD  TOTAL JOB_IDS

Total for query: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended 
Total for jrossthomson: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended 
Total for all users: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended


condor_submit .job

condor_q -better-analyze JobId
