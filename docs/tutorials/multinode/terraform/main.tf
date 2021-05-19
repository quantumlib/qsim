variable "project" {
  type=string
}
variable "zone" {
  type=string
}
variable "cluster_name" {
  type = string
  default = "c"
  description = "Name used to prefix resources in cluster."
  
}

module "htcondor" {
  source = "./htcondor/"
  cluster_name = var.cluster_name
  project = var.project
  zone = var.zone
  osversion = "7"
  max_replicas=20
  min_replicas=1
  service_account="htcondor@${var.project}.iam.gserviceaccount.com"
  use_preemptibles=false
}