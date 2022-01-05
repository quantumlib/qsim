#!/bin/bash -x

SERVER_TYPE="${htserver_type}"

##############################################################
## Install and configure HTCONDOR
##############################################################

if [ "${condorversion}" == "" ]; then
   CONDOR_INSTALL_OPT="condor"
else
   CONDOR_INSTALL_OPT="condor-all-${condorversion}"
  #  email  = "487217491196-compute@developer.gserviceaccount.com"
fi
if [ "${osversion}" == "6" ]; then
   CONDOR_STARTUP_CMD="service condor start"
else
   CONDOR_STARTUP_CMD="systemctl start condor;systemctl enable condor"
fi
CONDOR_REPO_URL=https://research.cs.wisc.edu/htcondor/yum/repo.d/htcondor-stable-rhel${osversion}.repo

sleep 2 #Give it some time to setup yum
cd /tmp
yum update -y
yum install -y wget curl net-tools vim gcc python3 git
wget https://research.cs.wisc.edu/htcondor/yum/RPM-GPG-KEY-HTCondor
rpm --import RPM-GPG-KEY-HTCondor
cd /etc/yum.repos.d && wget $CONDOR_REPO_URL
yum install -y $CONDOR_INSTALL_OPT

##############################################################
# Install Docker on Compute Nodes 
##############################################################
if [ "$SERVER_TYPE" == "compute" ]; then
    yum install -y yum-utils
    yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    yum install -y docker-ce docker-ce-cli containerd.io
    systemctl start docker
    systemctl enable docker
    usermod -aG docker condor
fi

##############################################################
# Configure Condor Daemons 
##############################################################
cd /tmp
cat <<EOF > condor_config.local
DISCARD_SESSION_KEYRING_ON_STARTUP=False
CONDOR_ADMIN=${admin_email}
CONDOR_HOST=${cluster_name}-manager
EOF

# Case for compute
if [ "$SERVER_TYPE" == "compute" ]; then
cat <<EOF1 >> condor_config.local
# Standard Stuff
DAEMON_LIST = MASTER, STARTD
ALLOW_WRITE = \$(ALLOW_WRITE), \$(CONDOR_HOST)
# Run Dynamics Slots
NUM_SLOTS = 1
NUM_SLOTS_TYPE_1 = 1
SLOT_TYPE_1 = 100%
SLOT_TYPE_1_PARTITIONABLE = TRUE
# Allowing Run as Owner
STARTER_ALLOW_RUNAS_OWNER = TRUE
SUBMIT_ATTRS = RunAsOwner
RunAsOwner = True
UID_DOMAIN = c.${project}.internal
TRUST_UID_DOMAIN = True
HasDocker = True
EOF1
fi

# Case for manager
if [ "$SERVER_TYPE" == "manager" ]; then
cat <<EOF2 >> condor_config.local
DAEMON_LIST = MASTER, COLLECTOR, NEGOTIATOR
ALLOW_WRITE = *
EOF2
fi

# Case for submit
if [ "$SERVER_TYPE" == "submit" ]; then
cat <<EOF3 >> condor_config.local
DAEMON_LIST = MASTER, SCHEDD
ALLOW_WRITE = \$(ALLOW_WRITE), \$(CONDOR_HOST)
# Allowing Run as Owner
STARTER_ALLOW_RUNAS_OWNER = TRUE
SUBMIT_ATTRS = RunAsOwner
RunAsOwner = True
UID_DOMAIN = c.${project}.internal
TRUST_UID_DOMAIN = True
EOF3
fi


mkdir -p /etc/condor/config.d
mv condor_config.local /etc/condor/config.d
eval $CONDOR_STARTUP_CMD

##############################################################
# Install and configure logging agent for StackDriver
##############################################################
curl -sSO https://dl.google.com/cloudagents/add-logging-agent-repo.sh
bash add-logging-agent-repo.sh --also-install

# Install Custom Metric Plugin:
google-fluentd-gem install fluent-plugin-stackdriver-monitoring

# Create Fluentd Config

cat <<EOF > condor.conf
<source>
  @type tail
  format none
  path /var/log/condor/*Log
  pos_file /var/lib/google-fluentd/pos/condor.pos
  read_from_head true
  tag condor
</source>
<source>
  @type exec
  command condor_status -direct `hostname` -format "%f " TotalLoadAvg |  cut -d " " -f 1
    <parse>
      keys la0
    </parse>
  tag condor_la0
  run_interval 5s
</source>
<source>
  @type exec
  command condor_status -schedd -format "%d" TotalIdleJobs
    <parse>
      keys q0
    </parse>
  tag condor_q0
  run_interval 5s
</source>
<match condor_la0>
  @type stackdriver_monitoring
  project ${project}
  <custom_metrics>
    key la0
    type custom.googleapis.com/la0
    metric_kind GAUGE
    value_type DOUBLE
  </custom_metrics>
</match>
<match condor_q0>
  @type stackdriver_monitoring
  project ${project}
  <custom_metrics>
    key q0
    type custom.googleapis.com/q0
    metric_kind GAUGE
    value_type INT64
  </custom_metrics>
</match>
EOF
mkdir -p /etc/google-fluentd/config.d/
mv condor.conf /etc/google-fluentd/config.d/

if [ "$SERVER_TYPE" == "submit" ]; then
mkdir -p /var/log/condor/jobs
touch /var/log/condor/jobs/stats.log
chmod 666 /var/log/condor/jobs/stats.log
fi

service google-fluentd restart

# Add Python Libraries and Autoscaler
if [ "$SERVER_TYPE" == "submit" ]; then
  python3 -m pip install --upgrade oauth2client
  python3 -m pip install --upgrade google-api-python-client
  python3 -m pip install --upgrade absl-py

cat <<EOFZ > /opt/autoscaler.py
${autoscaler}
EOFZ

# Create cron entry for autoscaler. Log to /var/log/messages

echo "* * * * * python3 /opt/autoscaler.py --p ${project} --z ${zone} --r ${region} %{ if multizone }--mz %{ endif }--g ${cluster_name} --c ${max_replicas} | logger " |crontab -

fi

# Now we can let everyone know that the setup is complete.

wall "******* HTCondor system configuration complete ********"
