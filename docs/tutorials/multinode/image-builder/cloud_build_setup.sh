############################################################
# Update cloud build permissions
############################################################
export CLOUD_BUILD_SERVICE_ACCOUNT=$(gcloud projects get-iam-policy $PROJECT --format "value(bindings.members)" | tr ';' '\n' | grep '@cloudbuild' | cut -d"'" -f2)
gcloud projects add-iam-policy-binding ${PROJECT} --member=${CLOUD_BUILD_SERVICE_ACCOUNT} --role=roles/cloudkms.cryptoKeyDecrypter
gcloud projects add-iam-policy-binding ${PROJECT} --member serviceAccount:${IMAGE_BUILDER_SERVICE_ACCOUNT} --role roles/iam.securityAdmin


############################################################
# Create account for Packer to build the image
############################################################
gcloud iam service-accounts create image-builder

export IMAGE_BUILDER_SERVICE_ACCOUNT=$(gcloud iam service-accounts list --filter="EMAIL ~ image-builder" --format="value(EMAIL)")

gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:${IMAGE_BUILDER_SERVICE_ACCOUNT} --role=roles/editor
gcloud projects add-iam-policy-binding ${PROJECT} --member serviceAccount:${IMAGE_BUILDER_SERVICE_ACCOUNT} --role roles/compute.admin
gcloud projects add-iam-policy-binding ${PROJECT} --member serviceAccount:${IMAGE_BUILDER_SERVICE_ACCOUNT} --role roles/iam.serviceAccountUser

############################################################
# Create a key file
############################################################
gcloud iam service-accounts keys create image-builder.key --iam-account=${IMAGE_BUILDER_SERVICE_ACCOUNT}

############################################################
# Create Key ring and encrypt key
############################################################
gcloud kms keyrings create packer --location=global
gcloud kms keys create pk --keyring=packer --location=global --purpose=encryption
gcloud kms encrypt --keyring=packer --key=pk --location=global --plaintext-file=image-builder.key --ciphertext-file=image-builder-enc.key

