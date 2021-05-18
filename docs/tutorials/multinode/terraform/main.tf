variable "cluster_name" {
  type = string
  default = "c"
  description = "Name used to prefix resources in cluster."
  
}

module "htcondor" {
  source = "./htcondor/"
  cluster_name = var.cluster_name
  osversion = "7"
  bucket_name = "quantum-hcondor-save"
  zone="us-east4-c"
  project="quantum-htcondor"
  max_replicas=20
  min_replicas=1
  service_account="htcondor2@quantum-htcondor.iam.gserviceaccount.com"
  use_preemptibles=true
}