export HF_HOME=./data

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DECORD_EOF_RETRY_MAX=204800

model_name="Qwen3-VL-8B-Instruct"
model="/model/path/${model_name}"

pip install transformers==4.57.1
tkn_budget=8192
select_head_ratio=0.5
max_num_frames=512
max_frames_per_group=64
# dataset=videomme
# dataset=lvbench
dataset=longvideobench_val_v

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 12341 -m lmms_eval \
    --model qwen3_vl_mm_k3 \
    --model_args "pretrained=$model,dataset=$dataset,load_in_4bit=false,use_chunk=true,fps=2,use_flash_attention_2=true,max_num_frames=$max_num_frames,max_frames_per_group=$max_frames_per_group,tkn_budget=$tkn_budget" \
    --tasks $dataset \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen3vl_8B_mm_k3_${max_num_frames}_budget_${tkn_budget} \
    --output_path ./logs_qwen3vl_mm_k3 \
    # --limit $limit \
