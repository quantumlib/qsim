gcloud iam service-accounts create "htcondor2"   --display-name="HTCONDOR 2"
gcloud projects add-iam-policy-binding quantum-htcondor --role roles/iam.serviceAccountUser --member="serviceAccount:htcondor2@quantum-htcondor.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding quantum-htcondor --role roles/compute.admin --member="serviceAccount:htcondor2@quantum-htcondor.iam.gserviceaccount.com"


git clone https://github.com/jrossthomson/qsim.git

cd qsim/docs/tutorials/multinode/

sudo journalctl -f -u google-startup-scripts.service

condor_q
-- Schedd: c-submit.c.quantum-htcondor.internal : <10.150.15.206:9618?... @ 05/14/21 15:09:57
OWNER BATCH_NAME      SUBMITTED   DONE   RUN    IDLE   HOLD  TOTAL JOB_IDS

Total for query: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended 
Total for jrossthomson: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended 
Total for all users: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended


condor_submit .job

condor_q -better-analyze JobId
