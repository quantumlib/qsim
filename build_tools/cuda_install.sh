curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py

# pre-install dependencies to prevent reboot
kernel_version=$(uname -r)
yum clean all
yum install -y kernel
yum install -y kernel-devel-${kernel_version} kernel-headers-${kernel_version} pciutils

# run the installer script
python3 install_gpu_driver.py --force
