# ---- Edit below -----#

export TF_VAR_project=[USER_PROJECT]
export TF_VAR_zone=us-east4-c
export TF_VAR_region=us-east4

export TF_VAR_multizone=false
# For regional/multizone, set this to the number of regions in the zone.
export TF_VAR_numzones=4

# ---- Do not edit below -----#

export TF_VAR_project_id=${TF_VAR_project}
export TF_VAR_service_account="htcondor@"${TF_VAR_project}".iam.gserviceaccount.com"

gcloud config set project $TF_VAR_project
gcloud services enable compute.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud config set compute/zone $TF_VAR_zone
gcloud config set compute/region $TF_VAR_region

gcloud config list

gcloud iam service-accounts create htcondor --display-name="Run HTCondor" 

# Add roles
gcloud projects add-iam-policy-binding ${TF_VAR_project} --member serviceAccount:${TF_VAR_service_account} --role roles/compute.admin
gcloud projects add-iam-policy-binding ${TF_VAR_project} --member serviceAccount:${TF_VAR_service_account} --role roles/iam.serviceAccountUser
gcloud projects add-iam-policy-binding ${TF_VAR_project} --member serviceAccount:${TF_VAR_service_account} --role roles/monitoring.admin
gcloud projects add-iam-policy-binding ${TF_VAR_project} --member serviceAccount:${TF_VAR_service_account} --role roles/logging.admin
gcloud projects add-iam-policy-binding ${TF_VAR_project} --member serviceAccount:${TF_VAR_service_account} --role roles/autoscaling.metricsWriter
