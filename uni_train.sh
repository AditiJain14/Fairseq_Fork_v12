ROOT="/cs/natlang-expts/aditi/monotonic_multihead_attention_new"

DATA="/cs/natlang-expts/aditi/fairseq_12.0/data-bin/de-orc"

EXPT="${ROOT}/experiments/trial"
mkdir -p ${EXPT}

FAIRSEQ="${ROOT}/fairseq"

USR="${ROOT}/simultaneous_translation"

CKPT="${EXPT}/checkpoints/unidirectional_enc6_dec1_smaller"
    mkdir -p ${CKPT}

export CUDA_VISIBLE_DEVICES=0,1

python3 $FAIRSEQ/train.py \
    $DATA \
    --log-format simple --log-interval 100 \
    --source-lang de --target-lang en \
    --task translation --tensorboard-logdir $CKPT/log \
    --user-dir $USR \
    --criterion label_smoothed_cross_entropy \
    --max-update 50000 \
    --arch transformer_unidirectional_aditi --save-dir $CKPT \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7  --warmup-updates 4000 \
    --lr 5e-4 --stop-min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001 \
    --dropout 0.3 \
    --label-smoothing 0.1\
    --max-tokens 5000 \
    --wandb-project Fairseq_MMA \
    --ddp-backend legacy_ddp --max-epoch 50