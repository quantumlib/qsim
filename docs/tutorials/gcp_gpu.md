# GPU-based quantum simulation on Google Cloud

In this tutorial, you configure and test a virtual machine (VM) to run GPU-based
quantum simulations on Google Cloud.

Note: The later steps in this tutorial require you to enter several commands at the
command line. Some commands might require you to add `sudo` before the command.
For example, if a step asks you to type `icecream -fancy`, you might need to
type `sudo icecream -fancy`.

## 1. Create a virtual machine

Follow the instructions in the
[Quickstart using a Linux VM](https://cloud.google.com/compute/docs/quickstart-linux)
guide to create a VM. In addition to the guidance specified in the Create a Linux VM
instance section, ensure that your VM has the following properties:

*   In the **Machine Configuration** section:
    1.  Select the tab for the **GPU** machine family.
    2.   In the **GPU type** option, choose **NVIDIA Tesla A100**.
    3.   In the **Number of GPUs** option, choose **1**.
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

### Find out more

*   [Choosing hardware for your qsim simulation](/qsim/choose_hw)
*   [Choosing the right machine family and type](https://cloud.google.com/blog/products/compute/choose-the-right-google-compute-engine-machine-type-for-you)
*   [Creating a VM with attached GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus#create-new-gpu-vm)

## 2. Prepare your computer

Use SSH in the `glcoud` tool to communicate with your VM.

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

## 3. Enable your virtual machine to use the GPU

1.  Install the GPU driver. Complete the steps provided in the following
    sections of the [Installing GPU
    drivers](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu):
    guide:
    *   [Examples](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#examples),
        under the **Ubuntu** tab. For step 3, only perform the steps for
        **Ubuntu 20.04** (steps 3a through 3f).
    *   [Verifying the GPU driver install](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#verify-driver-install)
2.  Install the CUDA toolkit.

    ```shell
    sudo apt install -y nvidia-cuda-toolkit
    ```

3.  Add your CUDA toolkit to the environment search path.
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

## 4. Install build tools

Install the tools required to build qsim. This step might take a few minutes to
complete.

```shell
sudo apt install cmake && sudo apt install pip && pip install pybind11
```


## 5. Create a GPU-enabled version of qsim

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


## 6. Verify your installation

You can use the following code to verify that qsim uses your GPU. You can paste
the code directly into the REPL, or paste the code in a file.

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

```none
[(0.7071067690849304+0j), 0j]
```

### Optional: Use the NVIDIA cuQuantum SDK

If you have the [NVIDIA cuQuantum SDK](https://developer.nvidia.com/cuquantum-sdk)
installed (instructions are provided
[here](https://docs.nvidia.com/cuda/cuquantum/custatevec/html/getting_started.html#installation-and-compilation),
cuStateVec v1.0.0 or higher is required),
you can use it with this tutorial. Before building qsim in step 5,
set the `CUQUANTUM_DIR` environment variable from the command line:

```bash
export CUQUANTUM_DIR=[PATH_TO_CUQUANTUM_SDK]
```

Once you have built qsim, modify the `gpu_options` line like so:

```python
gpu_options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=1)
```

This instructs qsim to make use of its cuQuantum integration, which provides
improved performance on NVIDIA GPUs. If you experience issues with this
option, please file an issue on the qsim repository.


## Next steps

After you finish, don't forget to stop or delete your VM on the Compute
Instances dashboard to prevent further billing.

You are now ready to run your own large simulations on Google Cloud. For sample
code of a large circuit, see the [Simulate a large
circuit](https://quantumai.google/qsim/tutorials/q32d14) tutorial.
