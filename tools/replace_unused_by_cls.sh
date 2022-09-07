#!/bin/bash
# [unused] ==> [CLS]

CKPT_DIR=$HOME/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts

for file in `grep -l 'unused' ${CKPT_DIR}/*/*/vocab.txt`; do
    echo $file;
    cp $file $file.bak
    sed -i "s/unused/CLS/g" $file
done
