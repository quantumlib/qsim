# GPU-based quantum simulation on Google Cloud

In this tutorial, you configure and test a virtual machine (VM) to run GPU-based
quantum simulations on Google Cloud.

The later steps in this tutorial require you to enter several commands at the
command line. Some commands might require you to add `sudo` before the command.
For example, if a step asks you to type `icecream -fancy`, you might need to
type `sudo icecream -fancy`.

## 1. Create a virtual machine

Follow the instructions in the
[Quickstart using a Linux VM](https://cloud.google.com/compute/docs/quickstart-linux)
guide to create a VM. In addition to the guidance under the Create a Linux VM
instance heading, ensure that your VM has the following properties:

*   In the Machine Configuration section, in the Machine Family options, click
    on the **GPU** filter.
*   In the Machine Configuration section, in the Machine family options, in the
    GPU type option, choose **NVIDIA Tesla A100**
    *   In the Number of GPUs option, choose **1**.
*   In the Boot disk section, click the **Change** button.
    *   In the Operating System option, choose **Ubuntu**.
    *   In the Version option, choose **20.04 LTS**.
    *   In the Size field, enter **30** (minimum).
    *   This overrides step 3 through 5 in the Quickstart guide.
*   In the Firewall section, ensure that both the **Allow HTTP traffic**
    checkbox and **Allow HTTPS traffic** checkbox have marks.

When Google Cloud finishes creating the VM, you can see your VM listed in the
[Compute Instances dashboard](https://pantheon.corp.google.com/compute/instances)
for your project.

### Find out more

*   [Choosing the right machine family and type](https://cloud.google.com/blog/products/compute/choose-the-right-google-compute-engine-machine-type-for-you)
*   [Creating a VM with attached GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus#create-new-gpu-vm)

## 2. Prepare your computer

Use SSH to create an encrypted tunnel from your computer to your VM and redirect
a local port to your VM over the tunnel.

1.  Follow the instructions in the
    [Installing Cloud SDK](https://cloud.google.com/sdk/docs/install)
    documentation to install the `gcloud` command line tool.
2.  After installation, initialize the Google Cloud environment by using the
    `gcloud init` command. You need to provide details about your VM, such as
    the project name and the region where your VM is located.
    1.  You can verify your environment by using the `gcloud config list`
        command.
3.  Create an SSH tunnel and redirect a local port to use the tunnel by typing
    the following command in a terminal window on your computer. Replace
    `[YOUR_INSTANCE_NAME]` with the name of your VM.

```shell
gcloud compute ssh [YOUR_INSTANCE_NAME]
```

When the command completes successfully, your prompt changes from your local
machine to your virtual machine.

## 3. Enable your virutal machine to use the GPu

1.  Install the GPU driver. In the Google Cloud documentation, in the Installing
    GPU drivers guide, follow the steps provided:
    *   In
        [the Examples section](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#examples),
        under the Ubuntu tab. Only follow the steps for Ubuntu 20.04 (steps 3a
        through 3f).
    *   In the
        [Verifying the GPU driver install](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#verify-driver-install)
        section.
2.  Run `sudo apt install -y nvidia-cuda-toolkit` to install the CUDA toolkit.
3.  Add your CUDA toolkit to the environment search path.
    1.  Run `ls /usr/local` to discover the directory of the CUDA toolkit that
        you installed. The toolkit will be the highest number that looks like
        the pattern `cuda-XX.Y`.  The output of the command should resemble the
        following:

        ```shell
        bin cuda cuda-11 cuda-11.4 etc games include lib man sbin share src
        ```

        In this case, the directory is `cuda-11.4`.
    2.  Add the CUDA toolkit path to your environment by appending the following
        line to your `~/.bashrc` file.  Replace `[DIR]` with the CUDA directory that
        you discovered in the previous step.

        ```shell
        echo "export PATH=/usr/local/[DIR]/bin${PATH:+:${PATH}}" >> ~/.bashrc
        ```

    3.  Run `source ~/.bashrc` to activate the new environment search path

## 4. Install build tools

Run the following command to install the Cmake tool and the pip tool. This step
might take a few minutes to complete.

```shell
sudo apt install cmake && sudo apt install pip
```


## 5. Create a GPU-enabled version of qsim

1.  Run the following command to clone the qsim repository.

    ```shell
    git clone https://github.com/quantumlib/qsim.git
    ```

2.  Run `cd qsim`to change your working directory to qsim.
3.  Run `make`to compile qsim. When CUDA toolkit is installed during a qsim
    compilation, make builds the GPU version automatically.
4.  Run `pip install .` to install your local version of qsimcirq.
5.  Run `python3 -c "import qsimcirq; print(qsimcirq.qsim_gpu)"`to verify your
    installation. If the installation completed successfully, the output from
    the command should resemble the following:

    ```shell
    <module 'qsimcirq.qsim_cuda' from '/home/user_org_com/qsim/qsimcirq/qsim_cuda.cpython-38-x86_64-linux-gnu.so'>
    ```


## 6. Verify your installation

You can use the code below to verify that qsim uses your GPU. You can paste the
code directly into the REPL, or paste the code in a file and ask the Python
interpreter to run it. If you paste the code into a file, use the following
command in the interpreter to run it. Replace `[FILE]` with the name of the file
that you create.

```
>>> exec(open("[FILE]").read())
```

In the `qsim` directory, run `python3` to start the Python REPL. Run the
following code.

```
# Import Cirq and qsim
import cirq
import qsimcirq

# Instantiate qubits and create a circuit
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))

# Instantiate a simulator that uses the GPU
gpu_options = qsimcirq.QSimOptions(use_gpu=True)
qsim_simulator = qsimcirq.QSimSimulator(qsim_options=gpu_options)

# Run the simulation
print("Running simulation for the following circuit:")
print(circuit)

qsim_results = qsim_simulator.compute_amplitudes(
    circuit, bitstrings=[0b00, 0b01])

print("qsim results:")
print(qsim_results)
```

After a moment, you should see a result that looks similar to the following.

```
[(0.7071067690849304+0j), 0j]
```

## Clean up

After you finish either tutorial, you can avoid continued billing by stopping or
deleting the VM instance that you create; visit the
[Compute Instances dashboard](https://pantheon.corp.google.com/compute/instances)
to manage your VM.

For more information about managing your VM, see the following documentation
from Google Cloud:

*   [Stopping and starting a VM](https://cloud.google.com/compute/docs/instances/stop-start-instance)
*   [Suspending and resuming an instance](https://cloud.google.com/compute/docs/instances/suspend-resume-instance)
*   [Deleting a VM instance](https://cloud.google.com/compute/docs/instances/deleting-instance)

## Next steps

After you finish, don't forget to stop or delete your VM on the Compute
Instances dashboard. For more information, see the
[Clean Up section in the Overview](https://quantumai.google/qsim/tutorials/gcp_overview#clean_up).

You are now ready to run your own large simulations on Google Cloud. For sample
code of a large circuit, see the [Simulate a large
circuit](https://quantumai.google/qsim/tutorials/q32d14) example.
