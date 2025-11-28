#!/bin/bash

PATH=$1
MODE=$2
echo "Will loop in ${PATH} looking for dirs with the following pattern: *VAE*latent8_log"

for i in ${PATH}/*VAE*latent8_log; do 
bash reconstruction.sh ${i} ${MODE} 8
; done 

