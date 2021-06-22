Dpendency:
pysam
lsgkm
gnu-parallel

Download *resource* folder from: http://renlab.sdsc.edu/yunjiang/deltaSVM/
Download *obs_threshold* folder from: https://github.com/ren-lab/deltaSVM/tree/master/obs_threshold
Download *pbs_threshold* folder from: https://github.com/ren-lab/deltaSVM/tree/master/pbs_threshold
Download *gkmsvm_models* folder from: https://github.com/ren-lab/deltaSVM/tree/master/gkmsvm_models

Add *"old_batch.csv"*,*"novel_batch.csv"* to the deltasvm dir from http://renlab.sdsc.edu/GVATdb/

To create input_tf for old/new_batch , run the corresponding code in data_pred.ipynb
Then run *bash parallel_run.sh* to calculate deltasvm scores

results will be saved in out folder