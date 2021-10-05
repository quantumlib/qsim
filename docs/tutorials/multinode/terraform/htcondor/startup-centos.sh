#!/bin/bash -x

SERVER_TYPE="${htserver_type}"

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
systemctl start condor
systemctl enable condor

##############################################################
# Install and configure logging agent for StackDriver
##############################################################
curl -sSO https://dl.google.com/cloudagents/install-logging-agent.sh
bash install-logging-agent.sh

##############################################################
# Add Python Libraries and Autoscaler
##############################################################

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