export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
export CUDA_VISIBLE_DEVICES="0,1"
export TOKENIZERS_PARALLELISM="false"

accelerate launch --multi_gpu --mixed_precision=bf16 ul2_trainer.py \
    --model_name "mmBERT_base_ul2_e22_d22_v2" \
    --train_dataset "./final_pretrain_mix" \
    --output_dir "./checkpoints_mmBERT_base_ul2_e22_d22_adamw_with_halflife_2e4_v2" \
    --batch_size 32 --eval_batch_size 32 --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --bf16 \
    --warmup_steps 2000 \
    --optimizer adamw \
    --adam_beta2_halflife_tokens 10000000 \
    --scheduler cosine \
    --max_grad_norm 2.0 \
    --num_epochs 2 \
    --logging_steps 1000 --eval_steps 10000 --save_steps 10000 \
    --use_trackio --trackio_project "mmBERT" --trackio_run_name "mmBERT_base_ul2_e22_d22_adamw_with_halflife_2e4"