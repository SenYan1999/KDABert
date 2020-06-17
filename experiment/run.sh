#!/bin/bash
#BSUB -J KDAEvaluate
#BSUB -e /nfsshare/home/dl04/KDABert/experiment/log/kda.err
#BSUB -o /nfsshare/home/dl04/KDABert/experiment/log/kda.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -R "select [ngpus>0] rusage [ngpus_excl_p=1]"

# python -m torch.distributed.launch --nproc_per_node=4 main.py --do_train  --log_file logs/eval.log --fp16 --distributed
python main.py --do_train  --fp16 
