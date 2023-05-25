# GPU-based quantum simulation on Google Cloud

In this tutorial, you configure and test a virtual machine (VM) to run GPU-based
quantum simulations on Google Cloud. The instructions for compiling qsim with GPU
support are also relevant if you are interested in running GPU simulations locally
or on a different cloud platform.

Before starting this tutorial, we recommend reading the [choosing hardware guide](../choose_hw)
in order to decide which type of GPU you would like to use and how many GPUs you
will need. As discussed there, you have a choice among 3 options:
1. using the native qsim GPU backend,
2.  using NVIDIA's cuQuantum as a backend for the latest version of
qsim, or 
3. using cuQuantum Appliance, which runs in a Docker container and has
a modified version of qsim. If you plan to do multi-GPU simulations, then you
will need to pick option 3. The following steps depend on which option you pick.
The headers for each step note which of these three options they apply to.

Note: The later steps in this tutorial require you to enter several commands at the
command line. Some commands might require you to add `sudo` before the command.
For example, if a step asks you to type `icecream -fancy`, you might need to
type `sudo icecream -fancy`.

## 1. Create a virtual machine (Options 1, 2, and 3)

Follow the instructions in the
[Quickstart using a Linux VM](https://cloud.google.com/compute/docs/quickstart-linux)
guide to create a VM. In addition to the guidance specified in the Create a Linux VM
instance section, ensure that your VM has the following properties:

*   In the **Machine Configuration** section:
    1.  Select the tab for the **GPU** machine family.
    2.   Select the **GPU Type** and **Number of GPUs** that you would like to use.
*   In the **Boot disk** section, click the **Change** button:
    1.   In the **Operating System** option, choose **Ubuntu**.
    2.   In the **Version** option, choose **20.04 LTS**.
    3.   In the **Size** field, enter **30** (minimum).
*   The instructions above override steps 3 through 5 in the [Create a Linux VM
    instance](https://cloud.google.com/compute/docs/quickstart-linux)
    Quickstart.
*   In the **Firewall** section, ensure that both the **Allow HTTP traffic**
    checkbox and the **Allow HTTPS traffic** checkboxs are selected.

When Google Cloud finishes creating the VM, you can see your VM listed in the
[Compute Instances dashboard](https://pantheon.corp.google.com/compute/instances)
for your project.

There may be [quotas](https://cloud.google.com/docs/quota) on your
account limiting the number and type of GPUs or the regions in which they can be located.
If necessary, you can request a quota increase, which in some cases is automatically
approved and in others requires communicating with customer support.

### Find out more

*   [Choosing the right machine family and type](https://cloud.google.com/blog/products/compute/choose-the-right-google-compute-engine-machine-type-for-you)
*   [Creating a VM with attached GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus#create-new-gpu-vm)

## 2. Prepare your computer (Options 1, 2, and 3)

Use SSH in the `gcloud` tool to communicate with your VM.

1.  Install the `gcloud` command line tool. Follow the instructions in the
    [Installing Cloud SDK](https://cloud.google.com/sdk/docs/install)
    documentation.
2.  After installation, run the `gcloud init` command to initialize the Google
    Cloud environment. You need to provide the `gcloud` tool with details
    about your VM, such as the project name and the region where your VM is
    located.
    1.  You can verify your environment by using the `gcloud config list`
        command.
3.  Connect to your VM by using SSH.  Replace `[YOUR_INSTANCE_NAME]` with the
    name of your VM.

    ```shell
    gcloud compute ssh [YOUR_INSTANCE_NAME]
    ```

When the command completes successfully, your prompt changes from your local
machine to your virtual machine.


## 3. Install Docker Engine (Option 3 only)
If you are setting up cuQuantum Appliance, follow [these](https://docs.docker.com/engine/install/) instructions
to install Docker Engine.

## 4. Enable your virtual machine to use the GPU (Options 1, 2, and 3)

1.  Install the GPU driver. Complete the steps provided in the following
    sections of the [Installing GPU
    drivers](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu):
    guide:
    *   [Examples](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#examples),
        under the **Ubuntu** tab. For step 3, only perform the steps for
        **Ubuntu 20.04** (steps 3a through 3f).
    *   [Verifying the GPU driver install](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#verify-driver-install)
2.  If missing, install the CUDA toolkit.
    (This may have already been installed along with the driver. You can
    check whether it is installed by checking whether the CUDA toolkit
    directory exists as described in step 3.)

    ```shell
    sudo apt install -y nvidia-cuda-toolkit
    ```

3.  Add your CUDA toolkit to the environment search path (Options 1 and 2 only)
    1.  Discover the directory of the CUDA toolkit that you installed.

        ```shell
        ls /usr/local
        ```

        The toolkit is the highest number that looks like the pattern
        `cuda-XX.Y`.  The output of the command should resemble the
        following:

        ```shell
        bin cuda cuda-11 cuda-11.4 etc games include lib man sbin share src
        ```

        In this case, the directory is `cuda-11.4`.
    2.  Add the CUDA toolkit path to your environment. You can run the following
        command to append the path to your `~/.bashrc` file.  Replace `[DIR]`
        with the CUDA directory that you discovered in the previous step.

        ```shell
        echo "export PATH=/usr/local/[DIR]/bin${PATH:+:${PATH}}" >> ~/.bashrc
        ```

    3.  Run `source ~/.bashrc` to activate the new environment search path

## 5. Install NVIDIA Container Toolkit (Option 3 only)
If you are setting up cuQuantum Appliance, follow the instructions
[here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)
to set up NVIDIA Container Toolkit.

## 6. Install NVIDIA cuQuantum Appliance (Option 3 only)
Follow the instructions [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuquantum-appliance)
to set up cuQuantum Appliance. You may need to use `sudo` for the Docker commands.
You may now skip to step 10.

## 7. Install build tools (Options 1 and 2)

Install the tools required to build qsim. This step might take a few minutes to
complete.

```shell
sudo apt install cmake && sudo apt install pip && pip install pybind11
```

## 8. Install cuQuantum SDK/cuStateVec (Option 2)
Reboot the VM. Then follow the instructions [here](https://docs.nvidia.com/cuda/cuquantum/custatevec/getting_started.html#install-custatevec-from-nvidia-devzone)
to install cuQuantum. Specifically, [this](https://developer.nvidia.com/cuquantum-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network) is the appropriate installer.
Reboot the VM again. Set the `CUQUANTUM_DIR` and `CUQUANTUM_ROOT` environment variables,
```shell
export CUQUANTUM_DIR=/opt/nvidia/cuquantum/
export CUQUANTUM_ROOT=/opt/nvidia/cuquantum/
```
modifying the above if cuQuantum was installed to a different directory.



## 9. Create a GPU-enabled version of qsim (Options 1 and 2)
0.  Reboot the VM (option 1).
1.  Clone the qsim repository.

    ```shell
    git clone https://github.com/quantumlib/qsim.git
    ```

2.  Run `cd qsim` to change your working directory to qsim.
3.  Run `make` to compile qsim. When make detects the CUDA toolkit during
    compilation, make builds the GPU version of qsim automatically.
4.  Run `pip install .` to install your local version of qsimcirq.
5.  Verify your qsim installation.

    ```shell
    python3 -c "import qsimcirq; print(qsimcirq.qsim_gpu)"
    ```

    If the installation completed successfully, the output from the command
    should resemble the following:

    ```none
    <module 'qsimcirq.qsim_cuda' from '/home/user_org_com/qsim/qsimcirq/qsim_cuda.cpython-38-x86_64-linux-gnu.so'>
    ```


## 10. Verify your installation (Options 1, 2, and 3)

You can use the following code to verify that qsim uses your GPU. You can paste
the code directly into the REPL, or paste the code in a file. See the documentation
[here](https://quantumai.google/reference/python/qsimcirq/QSimOptions) for Options 1
and 2 or [here](https://docs.nvidia.com/cuda/cuquantum/appliance/cirq.html) for Option 3.

```
# Import Cirq and qsim
import cirq
import qsimcirq

# Instantiate qubits and create a circuit
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))

# Instantiate a simulator that uses the GPU
# xx = 0 for Option 1, 1 for Option 2, or the number of GPUs for Option 3.
gpu_options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode = xx, max_fused_gate_size=4)
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

```none
[(0.7071067690849304+0j), 0j]
```

## Next steps

After you finish, don't forget to stop or delete your VM on the Compute
Instances dashboard to prevent further billing.

You are now ready to run your own large simulations on Google Cloud. For sample
code of a large circuit, see the [Simulate a large
circuit](https://quantumai.google/qsim/tutorials/q32d14) tutorial.
