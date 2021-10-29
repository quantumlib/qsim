# Multinode quantum simulation using HTCondor on GCP

In this tutorial, you will configure HTCondor to run multiple simulations of a
quantum circuit in parallel across multiple nodes. This method can be used to
accelerate Monte Carlo simulations of noisy quantum circuits.

Objectives of this tutorial:

* Use `terraform` to deploy a HTCondor cluster
* Run a multinode simulation using HTCondor
* Query cluster information and monitor running jobs in HTCondor
* Use `terraform` to destroy the cluster

## 1. Configure your environment

Although this tutorial can be run from your local computer, we recommend the use
of [Google Cloud Shell](https://cloud.google.com/shell). Cloud Shell has many useful tools pre-installed.

Once you have completed the [Before you begin](./gcp_before_you_begin.md)
tutorial, open the [Cloud Shell in the Cloud Console](https://console.cloud.google.com/home/dashboard?cloudshell=true).

### Clone this repo

In your Cloud Shell window, clone this Github repo.

``` bash
git clone https://github.com/quantumlib/qsim.git
```

If you get an error saying something like `qsim already exists`, you may need
to delete the `qsim` directory with `rm -rf qsim` and rerun the clone command.

### Change directory

Change directory to the tutorial:

``` bash
cd qsim/docs/tutorials/multinode/terraform
```

This is where you will use `terraform` to create the HTCondor cluster required to run your jobs.

### Edit `init.sh` file to match your environment

Using your favorite text file editor, open the `init.sh` file. The first few
lines should look like this:

```bash
# ---- Edit below -----#

export TF_VAR_project=[USER_PROJECT]
export TF_VAR_zone=us-east4-c
export TF_VAR_region=us-east4
```

Replace `[USER_PROJECT]` with the project name you chose on the
`Before you begin` page.

The other lines can optionally be modified to adjust your environment.
* The `TF_VAR_zone` and `TF_VAR_region` lines can be modified to select where
your project will create new jobs.

#### Find out more

* [Choosing a zone and region](https://cloud.google.com/compute/docs/regions-zones)

### Source the `init.sh` file

The edited `init.sh` file should be "sourced" in the cloud shell:

``` bash
source init.sh
```

Respond `Agree` to any pop-ups that request permissions on the Google Cloud platform.

The final outcome of this script will include:

* A gcloud config setup correctly
* A service account created
* The appropriate permissions assigned to the service account
* A key file created to enable the use of Google Cloud automation.

This will take up to 60 seconds. At the end you will see output about
permissions and the configuration of the account.

## 2. Run terraform

After the previous steps are completed, you can initialize `terraform` to begin
your cluster creation. The first step is to initialize the `terraform` state.
``` bash
terraform init
```
A successful result will contain the text:
```
Terraform has been successfully initialized!
```

### Run the `make` command

For convenience, some terraform commands are prepared in a `Makefile`. This
means you can now create your cluster, with a simple `make` command.

```bash
make apply
```

A successful run will show:

```
Apply complete! Resources: 4 added, 0 changed, 0 destroyed.
```

## 3. Connect to the submit node for HTCondor

Although there are ways to run HTCondor commands from your local machine, 
the normal path  is to login to the submit node. From there you can run 
commands to submit and monitor jobs on HTCondor.

### List VMs that were created by HTCondor

To see the VMs created by HTCondor, run:

```bash
gcloud compute instances list
```

At this point in the tutorial, you will see two instances listed:

```
NAME: c-manager
ZONE: us-central1-a
MACHINE_TYPE: n1-standard-1
PREEMPTIBLE:
INTERNAL_IP: X.X.X.X
EXTERNAL_IP: X.X.X.X
STATUS: RUNNING

NAME: c-submit
ZONE: us-central1-a
MACHINE_TYPE: n1-standard-1
PREEMPTIBLE:
INTERNAL_IP: X.X.X.X
EXTERNAL_IP: X.X.X.X
STATUS: RUNNING
```

### Connecting to the submit node

To connect to the submit node, click the `Compute Engine` item on the Cloud
dashboard. This will open the VM Instances page, where you should see the two
instances listed above. In the `c-submit` row, click on the `SSH` button to
open a new window connected to the submit node. During this step, you may see a
prompt that reads `Connection via Cloud Identity-Aware Proxy Failed`; simply
click on `Connect without Identity-Aware Proxy` and the connection should
complete.

This new window is logged into your HTCondor cluster. You will see a command
prompt that looks something like this:

```bash
[mylogin@c-submit ~]$
```

The following steps should be performed in this window.

### Checking the status

You can run `condor_q` to verify if the HTCondor install is completed. The output should look something like this:

```
-- Schedd: c-submit.c.quantum-htcondor-14.internal : <10.150.0.2:9618?... @ 08/18/21 18:37:50
OWNER BATCH_NAME      SUBMITTED   DONE   RUN    IDLE   HOLD  TOTAL JOB_IDS

Total for query: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended
Total for drj: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended
Total for all users: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended
```

If you get `command not found`, you will need to wait a few minutes for the HTCondor install to complete.

## 4. Get the sample code and run it

The HTCondor cluster is now ready for your jobs to be run. For this tutorial,
sample jobs have been provided in the Github repo.

###  Clone the repo on your cluster

On the submit node, you can clone the repo to get access to previously
created submission files:

```bash
git clone https://github.com/quantumlib/qsim.git
```

Then cd to the tutorial directory.

```bash
cd qsim/docs/tutorials/multinode
```

### Submit a job

Now it is possible to submit a job:
```
condor_submit noiseless.sub
```
This job will run the code in `noiseless3.py`, which executes a simple circuit and prints the results as a histogram. If successful, the output will be:
```
Submitting job(s).
1 job(s) submitted to cluster 1.
```
You can see the job in queue with the `condor_q` command.

The job will take several minutes to finish. The time includes creating a VM
compute node, installing the HTCondor system and running the job. When complete, the following files will be stored in the `out` directory:

* `out/log.1-0` contains a progress log for the job as it executes.
* `out/out.1-0` contains the final output of the job.
* `out/err.1-0` contains any error reports. This should be empty.

To view one of these files in the shell, you can run `cat out/[FILE]`,
replacing `[FILE]` with the name of the file to be viewed.

## 5. Run multinode noise simulations

Noise simulations make use of a [Monte Carlo
method](https://en.wikipedia.org/wiki/Monte_Carlo_method) for [quantum
trajectories](https://en.wikipedia.org/wiki/Quantum_Trajectory_Theory).

### The noise.sub file

To run multiple simulations, you can define a "submit" file. `noise.sub` is
an example of this file format, and is shown below. Notable features include:

* `universe = docker` means that all jobs will run inside a `docker` container.
* `queue 50` submits 50 separate copies of the job.

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
The job can be submitted with the `condor_submit` command.
```
condor_submit noise.sub
```
The output should look like this:
```
Submitting job(s)..................................................
50 job(s) submitted to cluster 2.
```
To monitor the ongoing process of jobs running, you can take advantage of the
Linux `watch` command to run `condor_q` repeatedly:
```
watch "condor_q; condor_status"
```
The output of this command will show you the jobs in the queue as well as the
VMs being created to run the jobs. There is a limit of 20 VMs for this
configuration of the cluster.

When the queue is empty, the command can be stopped with CTRL-C.

The output from all trajectories will be stored in the `out` directory. To see
the results of all simulations together, you can run:
```
cat out/out.2-*
```
The output should look something like this:
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

## 6. Shutting down

**IMPORTANT**:  To avoid excess billing for this project, it is important to
shut down the cluster. Return to the Cloud dashboard window for the steps below.

If your Cloud Shell is still open, simply run:
```
make destroy
```
If your Cloud Shell closed at any point, you'll need to re-initialize it.
[Open a new shell](https://console.cloud.google.com/home/dashboard?cloudshell=true)
and run:
```
cd qsim/docs/tutorials/multinode/terraform
source init.sh
make destroy
```
After these commands complete, check the Compute Instances dashboard to verify
that all VMs have been shut down. This tutorial makes use of an experimental
[autoscaling script](./terraform/htcondor/autoscaler.py) to bring up and turn
down VMs as needed. If any VMs remain after several minutes, you may need to
shut them down manually, as described in the next section.

## Next steps

The file being run in the previous example was `noise3.py`. To run your own
simulations, simply create a new python file with your circuit and change the
`noise3.py` references in `noise.sub` to point to the new file.

A detailed discussion of how to construct various types of noise in Cirq can be
found [here](https://quantumai.google/cirq/noise).

For more information about managing your VMs, see the following documentation
from Google Cloud:

*   [Stopping and starting a VM](https://cloud.google.com/compute/docs/instances/stop-start-instance)
*   [Suspending and resuming an instance](https://cloud.google.com/compute/docs/instances/suspend-resume-instance)
*   [Deleting a VM instance](https://cloud.google.com/compute/docs/instances/deleting-instance)

As an alternative to Google Cloud, you can download the Docker container or the
qsim source code to run quantum simulations on your own high-performance
computing platform.
