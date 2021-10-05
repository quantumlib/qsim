# How to create an image with qsim and htcondor

- enable APIs
    compute
    cloudbuild

```bash
gcloud services enable compute.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

## Packer Custom Build Step
```bash
git clone https://github.com/GoogleCloudPlatform/cloud-builders-community.git
cd cloud-builders-community/packer

export PACKER_VERSION=$(curl -Ls https://releases.hashicorp.com/packer | grep packer | sed 's/.*>packer\(.*\)<.*/packer\1/g' | sort -r | uniq | head -n1 | cut -d'_' -f2)
export PACKER_VERSION_SHA256SUM=$(curl -Ls https://releases.hashicorp.com/packer/${PACKER_VERSION}/packer_${PACKER_VERSION}_SHA256SUMS | grep linux_amd64 | cut -d' ' -f1)

gcloud builds submit --substitutions=_PACKER_VERSION=${PACKER_VERSION},_PACKER_VERSION_SHA256SUM=${PACKER_VERSION_SHA256SUM} .
```

## Cloud Build Setup
```bash
cd ../..
export CLOUD_BUILD_SERVICE_ACCOUNT=$(gcloud projects get-iam-policy $PROJECT_NUMBER --format "value(bindings.members)" | tr ';' '\n' | grep '@cloudbuild' | cut -d"'" -f2)
gcloud projects add-iam-policy-binding $PROJECT_NUMBER --member=${CLOUD_BUILD_SERVICE_ACCOUNT} --role=roles/cloudkms.cryptoKeyDecrypter

gcloud iam service-accounts create image-builder
export IMAGE_BUILDER_SERVICE_ACCOUNT=$(gcloud iam service-accounts list --filter="EMAIL ~ image-builder" --format="value(EMAIL)")
gcloud iam service-accounts keys create image-builder.key --iam-account=${IMAGE_BUILDER_SERVICE_ACCOUNT}
gcloud projects add-iam-policy-binding $PROJECT_NUMBER --member=serviceAccount:${IMAGE_BUILDER_SERVICE_ACCOUNT} --role=roles/editor

gcloud kms keyrings create packer --location=global
gcloud kms keys create pk --keyring=packer --location=global --purpose=encryption
```

## OpenPBS Image Build
```bash
cat <<EOF > openpbs-singlenode-complex.sh
#!/bin/env bash

yum install -y gcc make rpm-build libtool hwloc-devel \
      libX11-devel libXt-devel libedit-devel libical-devel \
      ncurses-devel perl postgresql-devel postgresql-contrib python3-devel tcl-devel \
      tk-devel swig expat-devel openssl-devel libXext libXft \
      autoconf automake gcc-c++ git wget

yum install -y expat libedit postgresql-server postgresql-contrib python3 \
      sendmail sudo tcl tk libical

cd /opt
git clone https://github.com/openpbs/openpbs.git
cd /opt/openpbs
git checkout tags/v20.0.1 -b v20.0.1

./autogen.sh
./configure --prefix=/opt/pbs

make
make install

/opt/pbs/libexec/pbs_postinstall

chmod u+s /opt/pbs/sbin/pbs_iff
EOF

cat <<EOF > packer.pkr.hcl
variable "project_id" {
    type = string
}

source "googlecompute" "openpbs_builder" {
    project_id              = var.project_id
    source_image_family     = "centos-7"
    source_image_project_id = ["centos-cloud"]
    zone                    = "us-central1-b"
    image_description       = "openpbs-singlenode-complex"
    ssh_username            = "admin"
    tags                    = ["packer"]
    account_file            = "image-builder.key"
    startup_script_file     = "openpbs-singlenode-complex.sh"
}

build {
    sources = ["source.googlecompute.openpbs_builder"]
}
EOF

cat <<EOF > cloudbuild.yaml
steps:
- name: 'gcr.io/cloud-builders/gcloud'
  args:
  - kms
  - decrypt
  - --ciphertext-file=image-builder-enc.key
  - --plaintext-file=image-builder.key
  - --location=global
  - --keyring=packer
  - --key=pk
- name: 'gcr.io/$PROJECT_ID/packer'
  args:
  - build
  - -var
  - project_id=$PROJECT_ID
  - packer.pkr.hcl
EOF

gcloud builds submit --config=cloudbuild.yaml . --timeout=15m
```

## Single Node Complex Deployment
```bash
cat <<EOF > complex-startup.sh
#!/bin/env bash

sed -i "s/PBS_START_MOM=0/PBS_START_MOM=1/g" /etc/pbs.conf
sed -i "s/PBS_SERVER=.*/PBS_SERVER=$(hostname)/g" /etc/pbs.conf
sed -i "s/$clienthost .*/$clienthost $(hostname)/g" /var/spool/pbs/mom_priv/config

systemctl restart pbs.service

/opt/pbs/bin/qmgr -c "set server flatuid=true"
/opt/pbs/bin/qmgr -c "set server acl_roots+=root@*"
/opt/pbs/bin/qmgr -c "set server operators+=root@*"
EOF
```

## Using OpenPBS
