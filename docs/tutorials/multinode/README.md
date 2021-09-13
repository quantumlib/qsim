# Multinode quantum simulation using HTCondor on GCP
This tutorial will take you through the process of running multiple simultaneous
`qsim` simulations on Google Cloud. In some situations, it is required to run
many instances of the same simulation. This could be used to provide a parameter
sweep or to evaluation noise characteristics.

One of the key competencies of a quantum computing effort is the ability to run
simulations. While quantum computing hardware is still years from general
availability, quantum computing simulation with `qsim` and `Cirq` is available for
researchers exploring a quantum program. It is expected that simulation will be
a gating requirement to make use of physical hardware, to prove out algorithms
both for computability and stability under noisy conditions. Many thousands of
simulation runs will typically be required to prove out the gating requirements
for each circuit, sometimes requiring years of simulation in advance of any
practical use of the quantum hardware.

## Objectives

* Use `terraform` to deploy a HTCondor cluster
* Run a multinode simulation using HTCondor
* Query cluster information and monitor running jobs in HTCondor
* Use `terraform` to destroy the cluster


## Step 1: Create a project
In a seperate tutorial the method to create Google Cloud project is explained in detail. 
[Please refer to this tutorial for guidance](https://quantumai.google/qsim/tutorials/qsimcirq_gcp#gcp_setup).

When you have a project created, move on to the next step.

## Step 2: Configure your enviroment

Although this tutorial can be run from your local computer, we recommend the use
of [Google Cloud Shell](https://cloud.google.com/shell). Cloud Shell has many useful tools pre-installed.

Once you have created  a project in the previous step, 
the Cloud Console with Cloud Shell activeated can be reached through this link: (https://console.cloud.google.com/home/dashboard?cloudshell=true)

### Clone this repo

In your Cloud Shell window, clone this Github repo.
``` bash
git clone https://github.com/jrossthomson/qsim.git
```

### Change directory
Change directory to the tutorial:
``` bash
cd qsim/docs/tutorials/multinode/terraform
```
This is where you will use `terraform` to create the HTCondor cluster required to run your jobs.

### Edit `init.sh` file to customize your environment

You can now edit `init.sh` to change the name of the project you are using. You
can also change the zone and region as you see fit. More information is
available [here](https://cloud.google.com/compute/docs/regions-zones).

Use your favorite text file editor, either the integrated [Cloud Shell
Editor](https://cloud.google.com/shell/docs/editor-overview), `vim`, `emacs` or `nano`.
For example:
``` bash
vim init.sh
```
The file has many lines, but only edit the first 4.
``` bash
export TF_VAR_project=quantum-htcondor-15
export TF_VAR_project_id=us-east4-c
export TF_VAR_zone=us-east4-c
export TF_VAR_region=us-east4
```

The most important is the first line, indicating the name of the project you created above.
``` bash
export TF_VAR_project=my-quantum-htcondor-project
```
The project id needs to be globally unique and to be the same as the project you just created.

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

##  Step 3: Run terraform

After the previous steps are completed, you can initialize `terraform` to begin your cluster creation.
The first step is to initialize the `terraform` state.
``` bash
terraform init
```
A successful result will contain the text:
```
Terraform has been successfully initialized!
```
### Run the `make` command
For convenience, some terraform commands are prepared in a `Makefile`. This means
you can now create your cluster, with a simple `make` command.
```bash
make apply
```
A successful run will show:
```
Apply complete! Resources: 4 added, 0 changed, 0 destroyed.
```
## Step4: Connect to the _submit_ node for HTCondor
Although there are ways to run `HTCondor` commands from your local machine, 
the normal path  is to login to the _submit_ node. From there you can run 
commands to submit and monitor jobs on HTCondor.

### List VMs that were created by

You can list the VMs created. One of them will be the submit node. It will be the VM with
"submit" in the name.

```
gcloud compute instances list
```
You will see two instances listed:
* c-manager: the core controller node
* c-submit: the submit node where you will run interactive work.

Identify then log in to the `submit` node. If you used the standard install, the
cluster name is "c". In that case you would connect using the `gcloud ssh`
command.
```bash
gcloud compute ssh c-submit
```
Now you are logged in to your HTCondor cluster. You will see a command prompt
something like:
```bash
[mylogin@c-submit ~]$
```

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

## Step 5: Get the sample code and run it
The HTCondor cluster is now ready for your jobs to be run. If you are familiar with HTCondor and you want to run your own jobs, you may do so. 

If you don't have jobs to run, you can get sample jobs from this Github repo. You will clone the repo to 
the submit node and run a job.

###  Clone the repo on your cluster

on the `submit` node you you can install the repo to get access to previously created submission files:
```
git clone https://github.com/jrossthomson/qsim.git
```
Then cd to the tutorial directory.
```
cd qsim/docs/tutorials/multinode
```
### Submit a job
Now it is possible to submit a job:
```
condor_submit circuit_q24.sub
```
If successful, the output will be:
```
Submitting job(s).
1 job(s) submitted to cluster 1.
```
You can see the job in queue by with the `condor_q` command.
```
condor_q
```
The job will take several minutes to finish. The time includes creating a VM
compute node, installing the HTCondor system and running the job. When completed
the files will be in the `out` directory.
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
## Step5: Run multi-node noise simulations
Noise simulations make use of a [Monte Carlo
method](https://en.wikipedia.org/wiki/Monte_Carlo_method) for [quantum
trajectories](https://en.wikipedia.org/wiki/Quantum_Trajectory_Theory). 

The primary goal of these simulations is to understand the behavior of the
quantum circuit taking into account the effects of noise. There are many
potential sources of noise in a quantum circuit, including thermal noise, 1/f
noise or shot noise, as well as perturbations from cosmic rays. Understanding
and calibrating noise in quantum computing is an active field of investigation,
and it will require well characterized simulations.

To run multiple simulations, you can run the submit file `noise.sub`.  The
submit file is shown below. It is key to note that this is running  a `docker`
container. Also, the `queue 50` command at the end of the file submits 50
separate instances of the container. Since this job is noise driven, the output
of each simulation with be different. A typical analysis of the output would be 
a statistical study of the means and variations, perhaps looking for parameter 
ranges where circuits are particularly susceptible to noise.


### The noise.sub file
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
The output will look as follows:
```
Submitting job(s)..................................................
50 job(s) submitted to cluster 2.
```
To monitor the ongoing process of jobs running, you can take advantage of the
Linux `watch` command and run `condor_q` repeatedly. When complete you watch can
be exited with cntrl-c.
```
watch "condor_q; condor_status"
```
The output of this command will show you the jobs in the queue as well as the
VMs being created to run the jobs. There is a limit of 20 VMs for this
configuration of the cluster. This can be modified for future runs.

When the command shows the queue is empty, the command can be stopped with cntl-c.

If this is the second _submit_ you have run, you can see the output of the all
the simualtions. The output will be in the `out` directory. 

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

## Next steps

To run your own simulations, simply create a noisy circuit in your  _qsim_ python file.
The file being run in this was `noise3.py`. You can take this example and expand
to your own circuit. 

```python
import cirq, qsimcirq

# Create a Bell state, |00) + |11)
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='m')
)

# Constructs a noise model that adds depolarizing noise after each gate.
noise = cirq.NoiseModel.from_noise_model_like(cirq.depolarize(p=0.05))

# Use the noise model to create a noisy circuit.
noisy_circuit = cirq.Circuit(noise.noisy_moments(circuit, system_qubits=[q0, q1]))

sim = qsimcirq.QSimSimulator()
result = sim.run(noisy_circuit, repetitions=1000)
# Outputs a histogram dict of result:count pairs.
# Expected result is a bunch of 0s and 3s, with fewer 1s and 2s.
# (For comparison, the noiseless circuit will only have 0s and 3s)
print(result.histogram(key='m'))
```

There are many more examples of circuits to be run [here](https://quantumai.google/cirq/noise)

## FINAL STEP: Shutting down 

> **IMPORTANT**:  To avoid excess billing for this project, it is important to shut down the cluster. This is easily done using the make command built for this purpose.
```
make destroy
```

> The most effective way to ensure you are not charged is to delete the project. [The instructions are here.](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects)



