from invoke import task
import os

@task
def demo(c):
    cmd = (
        "python -m torch.distributed.launch --master_port=3000 --nproc_per_node=1 "
        "scripts/demo.py None "
        "--task generation_obj "
        "--path /resources/models/kosmos-2-1.pt  "
        "--model-overrides \"{'visual_pretrained': '', 'dict_path':'kosmos_2/data/dict.txt'}\" "
        "--dict-path 'kosmos_2/data/dict.txt' "
        "--required-batch-size-multiple 1 "
        "--remove-bpe=sentencepiece "
        "--max-len-b 500 "
        "--add-bos-token "
        "--beam 1 "
        "--buffer-size 1 "
        "--image-feature-length 64 "
        "--locate-special-token 1 "
        "--batch-size 1 "
        "--nbest 1 "
        "--no-repeat-ngram-size 3 "
        "--location-bin-size 32"
    )
    with c.prefix('export LOCAL_RANK=0 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0'):
        c.run(cmd, pty=True)
