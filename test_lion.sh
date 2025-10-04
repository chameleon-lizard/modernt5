export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
export CUDA_VISIBLE_DEVICES="0,1"
export TOKENIZERS_PARALLELISM="false"

#export HF_TOKEN="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"
export HF_TOKEN="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"

#token for WRITE, track-io
#export HF_TOKEN="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"
export HF_HOME="/home/jovyan/shares/SR003.nfs2/.cache/"

#   --weight_decay 0.01 \

accelerate launch --multi_gpu --mixed_precision=bf16 ul2_trainer.py \
    --model_name "mmBERT_base_ul2_e22_d22_v2" \
    --train_dataset "../final_pretrain_mix" \
    --output_dir "./checkpoints_mmBERT_base_ul2_e22_d22_adamw_with_halflife_5e4" \
    --batch_size 32 --eval_batch_size 32 --gradient_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --bf16 \
    --warmup_steps 2000 \
    --optimizer adamw \
    --adam_beta2_halflife_tokens 10000000 \
    --scheduler cosine \
    --max_grad_norm 2.0 \
    --num_epochs 2 \
    --logging_steps 1000 --eval_steps 10000 --save_steps 10000 \
    --use_trackio --trackio_project "mmBERT" --trackio_run_name "mmBERT_base_ul2_e22_d22_adamw_with_halflife_5e4"