#!/bin/bash
# requires samtools
# run as bash run_dedup.sh possorted_genome_bam

date;hostname;pwd
FILE=${1}

if [ -z ${SLURM_JOB_CPUS_PER_NODE} ]; then
    N_CORES=$(python -c "import os; print(os.cpu_count())")
else
    N_CORES=${SLURM_JOB_CPUS_PER_NODE}
fi

echo "Starting job with $N_CORES cores"  
echo $FILE
if [ -f $FILE.unique.bam ]; then
    echo "found $FILE.unique.bam";
    if [ ! -f $FILE.unique.bam.bai ]; then
        echo "Indexing file"
        samtools index $FILE.unique.bam
    fi
else
    # remove fake uniquely-mapped reads with modified MAPQ scores from cell ranger bam file
    # see https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam#bam-align-tags
    echo "Extracting unique reads"
    samtools view -H $FILE.bam > header.sam
    samtools view -q 255 -@ $N_CORES $FILE.bam | grep -v "MM:i:1" | cat header.sam - | samtools view -Sb -@ $N_CORES  - > $FILE.unique.bam
    echo "Indexing file"
    samtools index $FILE.unique.bam
fi

if [ -f cell_sorted_$FILE.unique.bam ]; then
    echo "found existing cell_sorted bam"
else 
    echo "sorting by cell barcode"
    samtools sort -@ $N_CORES -t CB $FILE.unique.bam -o cell_sorted_$FILE.unique.bam
fi

echo "extracting unique UMIs"
# also found in scripts folder
python parse_cellsorted_bam.py -i cell_sorted_$FILE.unique.bam

echo "job done."