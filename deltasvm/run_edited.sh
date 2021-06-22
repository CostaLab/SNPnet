#!/bin/bash
set -e 

echo $1
python scripts/generate_allelic_seqs.py -f resources/hg19.fa -s input_tf/snp_$1.tsv -o data/selex_allelic_oligos_$1 2>log/selex_allelic_oligos_$1.log
scripts/deltasvm_subset_multi data/selex_allelic_oligos_$1.ref.fa data/selex_allelic_oligos_$1.alt.fa resources/models.weights.txt out/pbs_$1.pred.tsv resources/threhsolds.pbs.tsv 2>log/deltasvm_$1.log
sort -k1,1 -k3,3 -o out/pbs_$1.pred.tsv out/pbs_$1.pred.tsv
python scripts/pred_reduce.py -t $1

scripts/gkmpredict -T 1 data/selex_allelic_oligos_$1.ref.fa gkmsvm_models/$3.model.txt tmp/$1.ref.gkm.tsv &>log/$3.ref.gkm.log
sort -k1,1 -o tmp/$1.ref.gkm.tsv tmp/$1.ref.gkm.tsv
scripts/gkmpredict -T 1 data/selex_allelic_oligos_$1.alt.fa gkmsvm_models/$3.model.txt tmp/$1.alt.gkm.tsv &>log/$3.alt.gkm.log
sort -k1,1 -o tmp/$1.alt.gkm.tsv tmp/$1.alt.gkm.tsv
paste tmp/$1.ref.gkm.tsv tmp/$1.alt.gkm.tsv | cut -f1,2,4 > tmp/$1.merge.gkm.tsv	

python scripts/obs_pred_edited.py -s resources/threhsolds.obs.tsv -o out/obs_$1.pred.tsv -t $1
sort -k1,1 -k2,2 -o out/obs_$1.pred.tsv out/obs_$1.pred.tsv
paste out/obs_$1.pred.tsv out/pbs_$1.pred.tsv | cut -f1-6,8,10 | sort -k7,7 -k5,5r | sed '1i snp\ttf\tallele1_bind\tallele2_bind\tref_binding\talt_binding\tdeltaSVM_score\tpreferred_allele' > out/summary_$1.pred.tsv


