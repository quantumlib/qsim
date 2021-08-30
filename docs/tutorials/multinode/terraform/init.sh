# ---- Edit below -----#

export TF_VAR_project=quantum-htcondor-15
export TF_VAR_project_id=us-east4-c
export TF_VAR_zone=us-east4-c
export TF_VAR_region=us-east4

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

"""
# Save key file locally.
gcloud iam service-accounts keys create ~/.${TF_VAR_project_id}.json --iam-account=${TF_VAR_service_account}
export GOOGLE_APPLICATION_CREDENTIALS=~/.${TF_VAR_project_id}.json
"""