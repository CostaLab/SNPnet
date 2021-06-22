Download HT-Selex fastq files from https://www.ebi.ac.uk/ena/browser/view/PRJEB9797 and place in the *fastq* folder
Download corresponding Zero_Cycle from https://www.ebi.ac.uk/ena/browser/view/PRJEB20112 and place in the *fastq* folder

Add HT-Selex files to config.json *"exps": {}*

Run *snakemake --cores all* in Unix terminal to generate seqs

To change subsamplesize change *num* in Snakefile