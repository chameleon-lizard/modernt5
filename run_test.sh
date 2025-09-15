export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
export TOKENIZERS_PARALLELISM="false"

accelerate launch --multi_gpu --mixed_precision=bf16 training_acc_ext.py --model_name "./modernt5_from_USER2-small_e12_d12" --train_dataset "./final_pretrain_mix" \
    --bf16 --use_trackio --trackio_project "my-ul2-training-USER2-small" --output_dir "./checkpoints_USER2-small_e12_d12" --num_epochs 8 --batch_size 86 --eval_batch_size 128 \
    --logging_steps 500 --eval_steps 30000 --save_steps 5000 --learning_rate 1e-3 --optimizer lion8bit --gradient_accumulation_steps 4