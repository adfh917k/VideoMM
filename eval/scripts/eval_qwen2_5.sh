export HF_HOME=./data

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DECORD_EOF_RETRY_MAX=204800

model_name="Qwen2.5-VL-7B-Instruct"
model="/model/path/${model_name}"


pip install transformers==4.49

tkn_budget=0
max_num_frames=512
max_frames_per_group=64
# dataset=videomme
# dataset=lvbench
dataset=longvideobench_val_v

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 12341 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "pretrained=$model,dataset=$dataset,fps=2,use_flash_attention_2=True,max_num_frames=$max_num_frames,tkn_budget=$tkn_budget" \
    --tasks $dataset \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2_5vl_7B_${max_num_frames}_budget_${tkn_budget} \
    --output_path ./logs_qwen2_5vl \
