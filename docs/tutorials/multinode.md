# Multinode quantum simulation using HTCondor on GCP

In this tutorial, you will configure HTCondor to run multiple simulations of a
quantum circuit in parallel across multiple nodes. This method can be used to
accelerate Monte Carlo simulations of noisy quantum circuits.

Objectives of this tutorial:

* Use `terraform` to deploy a HTCondor cluster
* Run a multinode simulation using HTCondor
* Query cluster information and monitor running jobs in HTCondor
* Use `terraform` to destroy the cluster

## 1. Deploy your HTCondor cluster

Once you have completed the [Before you begin](./gcp_before_you_begin.md)
tutorial, follow steps 1-6 of the HPC Toolkit
[HTCondor Tutorial](https://github.com/GoogleCloudPlatform/hpc-toolkit/blob/main/docs/tutorials/README.md#htcondor-tutorial){:.external}
to set up a HTCondor cluster in your GCP project. Keep this window open - both
the Cloud Shell and the remaining steps will be used in this tutorial.

### Checking the status

You can run `condor_q` from the access point to verify if the HTCondor install
is completed. The output should look something like this:

```
-- Schedd: access-point-0.c.quantum-htcondor-14.internal : <10.150.0.2:9618?... @ 08/18/21 18:37:50
OWNER BATCH_NAME      SUBMITTED   DONE   RUN    IDLE   HOLD  TOTAL JOB_IDS

Total for query: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended
Total for drj: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended
Total for all users: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended
```

If you get `command not found`, you will need to wait a few minutes for the HTCondor install to complete.

## 2. Get the sample code and run it

The HTCondor cluster is now ready for your jobs to be run. For this tutorial,
sample jobs have been provided in the Github repo.

###  Clone the repo on your cluster

On the access point, you can clone the repo to get access to previously
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
```bash
# create output directory if it doesn't exist
mkdir -p ./out
# submit the job
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

## 3. Run multinode noise simulations

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
The job can be submitted from the access point with the `condor_submit` command.
```bash
# create output directory if it doesn't exist
mkdir -p ./out
# submit the job
condor_submit noise.sub
```
The output should look like this:
```
Submitting job(s)..................................................
50 job(s) submitted to cluster 2.
```
To monitor the ongoing process of jobs running, you can take advantage of the
Linux `watch` command to run `condor_q` repeatedly:
```bash
watch "condor_q; condor_status"
```
The output of this command will show you the jobs in the queue as well as the
VMs being created to run the jobs. There is a limit of 20 VMs for this
configuration of the cluster.

When the queue is empty, the command can be stopped with CTRL-C.

The output from all trajectories will be stored in the `out` directory. To see
the results of all simulations together, you can run:
```bash
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

## 4. Shutting down

**IMPORTANT**:  To avoid excess billing for this project, it is important to
shut down the cluster. Return to steps 7-8 of the HPC Toolkit
[HTCondor Tutorial](https://github.com/GoogleCloudPlatform/hpc-toolkit/blob/main/docs/tutorials/README.md#htcondor-tutorial)
for instructions on how to accomplish this.

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
