variable "cluster_name" {
    type = string
    default = "condor"
}
variable "admin_email" {
    type = string
    default = ""
}
variable "osversion" {
    type = string
    default = "7"
}
variable "osimage" {
    type = string
    default = "hpc-centos-7"
}
variable "osproject" {
  type = string
  default = "cloud-hpc-image-public"
}
variable "condorversion" {
    type = string
    default = ""
}
variable "project" {
    type = string
}
variable "zone" {
    type = string
}
variable "region" {
    type = string
}
variable "numzones" {
    type = string
}
variable "multizone" {
    type = bool
}
variable "min_replicas" {
    type = number
    default = 0
}
variable "max_replicas" {
    type = number
    default = 20
}
variable "use_preemptibles" {
    type = bool
    default = true
}
variable "metric_target_loadavg" { 
  type = number 
  default = "1.0"
}
variable "metric_target_queue" { 
  type = number 
  default = 10
}
variable "compute_instance_type" {
  type = string
  default = "n1-standard-1"
}
variable "instance_type" {
  type = string
  default = "n1-standard-1"
}
variable "service_account" {
  type = string
  default = "default"
}
locals{
  autoscaler = file("${path.module}/autoscaler.py")
  compute_startup = templatefile(
    "${path.module}/startup-centos.sh", 
    {
      "project" = var.project,
      "cluster_name" = var.cluster_name,
      "htserver_type" = "compute",
      "osversion" = var.osversion,
      "zone" = var.zone,
      "region" = var.region,
      "multizone" = var.multizone,
      "condorversion" = var.condorversion,
      "max_replicas" = var.max_replicas,
      "autoscaler" = "",
      "admin_email" = var.admin_email
    })
  submit_startup = templatefile(
    "${path.module}/startup-centos.sh", 
    {
      "project" = var.project,
      "cluster_name" = var.cluster_name,
      "htserver_type" = "submit",
      "osversion" = var.osversion,
      "condorversion" = var.condorversion,
      "zone" = var.zone,
      "region" = var.region,
      "multizone" = var.multizone,
      "max_replicas" = var.max_replicas,
      "autoscaler" = local.autoscaler,
      "admin_email" = var.admin_email
    })
  manager_startup = templatefile(
    "${path.module}/startup-centos.sh", 
    {
      "project" = var.project,
      "cluster_name" = var.cluster_name,
      "htserver_type" = "manager",
      "osversion" = var.osversion,
      "zone" = var.zone,
      "region" = var.region,
      "multizone" = var.multizone,
      "max_replicas" = var.max_replicas,
      "condorversion" = var.condorversion,
      "autoscaler" = "",
      "admin_email" = var.admin_email
    })
}
data "google_compute_image" "startimage" {
  family  = var.osimage
  project = var.osproject
}
resource "google_compute_instance" "condor-manager" {
  boot_disk {
    auto_delete = "true"
    device_name = "boot"

    initialize_params {
      image = data.google_compute_image.startimage.self_link
      size  = "200"
      type  = "pd-standard"
    }

    mode   = "READ_WRITE"
  }

  can_ip_forward      = "false"
  deletion_protection = "false"
  enable_display      = "false"

  machine_type            = var.instance_type
  metadata_startup_script = local.manager_startup
  name                    = "${var.cluster_name}-manager"
  network_interface {
    access_config {
      network_tier = "PREMIUM"
    }

    network            = "default"
    #network_ip         = "10.128.0.2"
    subnetwork         = "default"
    subnetwork_project = var.project
  }

  project = var.project

  scheduling {
    automatic_restart   = "true"
    on_host_maintenance = "MIGRATE"
    preemptible         = "false"
  }

  service_account {
    email = var.service_account
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  shielded_instance_config {
    enable_integrity_monitoring = "true"
    enable_secure_boot          = "false"
    enable_vtpm                 = "true"
  }

  tags = ["${var.cluster_name}-manager"]
  zone = var.zone
}

resource "google_compute_instance" "condor-submit" {
  boot_disk {
    auto_delete = "true"
    device_name = "boot"

    initialize_params {
      image = data.google_compute_image.startimage.self_link
      size  = "200"
      type  = "pd-standard"
    }

    mode   = "READ_WRITE"
  }

  can_ip_forward      = "false"
  deletion_protection = "false"
  enable_display      = "false"

  labels = {
    goog-dm = "mycondorcluster"
  }

  machine_type            = var.instance_type
  metadata_startup_script = local.submit_startup
  name                    = "${var.cluster_name}-submit"

  network_interface {
    access_config {
      network_tier = "PREMIUM"
    }

    network            = "default"
    #network_ip         = "10.128.0.3"
    subnetwork         = "default"
    subnetwork_project = var.project
  }

  project = var.project

  scheduling {
    automatic_restart   = "true"
    on_host_maintenance = "MIGRATE"
    preemptible         = "false"
  }

  service_account {
      email = var.service_account
  #  email  = "487217491196-compute@developer.gserviceaccount.com"
    #scopes = ["https://www.googleapis.com/auth/monitoring.write", "https://www.googleapis.com/auth/compute", "https://www.googleapis.com/auth/servicecontrol", "https://www.googleapis.com/auth/devstorage.read_only", "https://www.googleapis.com/auth/logging.write", "https://www.googleapis.com/auth/service.management.readonly", "https://www.googleapis.com/auth/trace.append"]
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  shielded_instance_config {
    enable_integrity_monitoring = "true"
    enable_secure_boot          = "false"
    enable_vtpm                 = "true"
  }

  tags = ["${var.cluster_name}-submit"]
  zone = var.zone
}
resource "google_compute_instance_template" "condor-compute" {
  can_ip_forward = "false"

  disk {
    auto_delete  = "true"
    boot         = "true"
    device_name  = "boot"
    disk_size_gb = "200"
    mode         = "READ_WRITE"
    source_image = data.google_compute_image.startimage.self_link
    type         = "PERSISTENT"
  }

  machine_type = var.compute_instance_type

  metadata = {
    startup-script = local.compute_startup
  }

  name = "${var.cluster_name}-compute"

  network_interface {
    access_config {
      network_tier = "PREMIUM"
    }

    network = "default"
  }

  project = var.project
  region  = var.zone

  scheduling {
    automatic_restart   = "false"
    on_host_maintenance = "TERMINATE"
    preemptible         = var.use_preemptibles
  }

  service_account {
    email = var.service_account
    scopes = ["cloud-platform"]
  }

  tags = ["${var.cluster_name}-compute"]
}
resource "google_compute_instance_group_manager" "condor-compute-igm" {
  count = var.multizone ? 0 : 1
  base_instance_name = var.cluster_name
  name               = var.cluster_name

  project            = var.project
  target_size        = "0"

  update_policy {
    max_surge_fixed         = 2
    minimal_action          = "REPLACE"
    type                    = "OPPORTUNISTIC"
  }

  version {
    instance_template = google_compute_instance_template.condor-compute.self_link
    name              = ""
  }
  timeouts {
    create = "60m"
    delete = "2h"
  }
  # Yup, didn't want to use this, but I was getting create and destroy errors. 
  depends_on = [
   google_compute_instance_template.condor-compute 
  ]
  zone = var.zone
}

resource "google_compute_region_instance_group_manager" "condor-compute-igm" {
  count = var.multizone ? 1 : 0
  base_instance_name = var.cluster_name
  name               = var.cluster_name

  project            = var.project
  target_size        = "0"

  update_policy {
    max_surge_fixed         = var.numzones
    minimal_action          = "REPLACE"
    type                    = "OPPORTUNISTIC"
  }

  version {
    instance_template = google_compute_instance_template.condor-compute.self_link
    name              = ""
  }
  timeouts {
    create = "60m"
    delete = "2h"
  }
  # Yup, didn't want to use this, but I was getting create and destroy errors.
  depends_on = [
   google_compute_instance_template.condor-compute
  ]
  region = var.region
}
/*
resource "google_compute_autoscaler" "condor-compute-as" {
  name    = "${var.cluster_name}-compute-as"
  project = var.project
  target  = google_compute_instance_group_manager.condor-compute-igm.self_link
  zone    = var.zone

  autoscaling_policy {
    cooldown_period = 30
    max_replicas    = var.max_replicas
    min_replicas = var.min_replicas

    cpu_utilization {
      target = 0.2
    }

    metric {
       name   = "custom.googleapis.com/q0"
       target = var.metric_target_queue
       type   = "GAUGE"
    }
    metric {
       name   = "custom.googleapis.com/la0"
       target = var.metric_target_loadavg
       type   = "GAUGE"
    }

  }

  timeouts {
    create = "60m"
    delete = "2h"
  }

  depends_on = [
   google_compute_instance_group_manager.condor-compute-igm
  ]
}
*/

output "startup_script" {
  value = local.submit_startup
}