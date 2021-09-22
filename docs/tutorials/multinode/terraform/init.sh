# This section sets variables used by terraform and gcloud.
# ---- Edit below -----#

export TF_VAR_project=quantum-htcondor-15
export TF_VAR_zone=us-central1-a
export TF_VAR_region=us-central1
export TF_VAR_numzones=4  # for regional/multizone, set to the number of regions in the zone
export TF_VAR_multizone=true # optional 

# ---- Do not edit below -----#

export TF_VAR_project_id=${TF_VAR_project}
export TF_VAR_service_account="htcondor@"${TF_VAR_project}".iam.gserviceaccount.com"

# This section configures gcloud and enables 2 apis

gcloud config set project $TF_VAR_project
gcloud services enable compute.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud config set compute/zone $TF_VAR_zone
gcloud config set compute/region $TF_VAR_region

gcloud config list

# This section creates a service account and enables permissions to run HTCondor

gcloud iam service-accounts create htcondor --display-name="Run HTCondor" 

gcloud projects add-iam-policy-binding ${TF_VAR_project} --member serviceAccount:${TF_VAR_service_account} --role roles/compute.images.get
gcloud projects add-iam-policy-binding ${TF_VAR_project} --member serviceAccount:${TF_VAR_service_account} --role roles/compute.admin
gcloud projects add-iam-policy-binding ${TF_VAR_project} --member serviceAccount:${TF_VAR_service_account} --role roles/iam.serviceAccountUser
gcloud projects add-iam-policy-binding ${TF_VAR_project} --member serviceAccount:${TF_VAR_service_account} --role roles/monitoring.admin
gcloud projects add-iam-policy-binding ${TF_VAR_project} --member serviceAccount:${TF_VAR_service_account} --role roles/logging.admin
gcloud projects add-iam-policy-binding ${TF_VAR_project} --member serviceAccount:${TF_VAR_service_account} --role roles/autoscaling.metricsWriter
