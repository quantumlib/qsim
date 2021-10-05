variable "project_id" {
    type = string
}

variable "date" {
    type = string
}

variable "family" {
    type = string
    default="qsim-htcondor"
}

source "googlecompute" "qsim_htcondor_builder" {
    project_id              = var.project_id
    source_image_family     = "hpc-centos-7"
    source_image_project_id = ["cloud-hpc-image-public"]
    zone                    = "us-central1-a"
    image_description       = "qsim on HTCondor"
    ssh_username            = "admin"
    tags                    = ["packer"]
    account_file            = "image-builder.key"
    image_name              = "${var.family}-${var.date}"
    image_family            = var.family
    startup_script_file     = "tmp.sh"
}

build {
    sources = ["source.googlecompute.qsim_htcondor_builder"]
}