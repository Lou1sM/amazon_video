#!/bin/sh

dev=$1
echo $dev
for caps in swinbert kosmos nocaptions; do
    CUDA_VISIBLE_DEVICES=${dev} python train.py --caps $caps --n_epochs 10 --expname ${caps} --resumm_scenes --overwrite --bs 4
    #python compute_metrics.py --expname ${caps}
    CUDA_VISIBLE_DEVICES=${dev} python train.py --caps $caps --reorder --n_epochs 10 --expname ${caps}_reorder --resumm_scenes --overwrite --bs 4
    #python compute_metrics.py --expname ${caps}_reorder
    CUDA_VISIBLE_DEVICES=${dev} python train.py --caps $caps --randorder --n_epochs 10 --expname ${caps}_randorder --resumm_scenes --overwrite --bs 4
done
