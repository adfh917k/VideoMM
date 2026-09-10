export HF_HOME=./data

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DECORD_EOF_RETRY_MAX=204800

model_name="GLM-4.1V-9B-Thinking"
model="/model/path/${model_name}"


pip install transformers==4.57.1

tkn_budget=8192
max_num_frames=512
max_frames_per_group=64
# dataset=videomme
# dataset=lvbench
dataset=longvideobench_val_v

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 12341 -m lmms_eval \
    --model glm4v_mm \
    --model_args "pretrained=$model,dataset=$dataset,load_in_4bit=false,use_chunk=true,fps=2,use_flash_attention_2=true,max_num_frames=$max_num_frames,max_frames_per_group=$max_frames_per_group,tkn_budget=$tkn_budget" \
    --tasks $dataset \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix glm4v_mm_${max_num_frames}_budget_${tkn_budget} \
    --output_path ./logs_glm4v_mm \
    # --limit $limit \
