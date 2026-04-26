export MODEL_NAME="/home/xzh/xzh/pretrained/sd_xl_base_1.0"

#heatmap parameter
CONTENT_RARE_TOKEN="monadikos"          # 例如：xlo
CLASS_WORD="cat"          # 例如：dog
STYLE_RARE_TOKEN="anime"
SUPER_WORD='dog'

TIMESTEP_MODE='priecewise'

# Hyper parameters
export period_sample_epoch=4
export sampled_column_ratio=0.125

# For weight similarity
export CONTENT_LR=0.00005
export STYLE_LR=0.00005
export weight_lr=0.005
export similarity_lambda=0.5
export RANK=64
export WANDB_NAME="unziplora-改进tlora中的非对称时间步策略"
export INSTANCE_DIR="/home/xzh/xzh/UnZipLoRA/instance_data/anime_cat"
export OUTPUT_DIR="models/anime_cat/anime_cat"
export STEPS=600
# Training prompt 

# both prompt
export PROMPT="A ${CONTENT_RARE_TOKEN} ${CLASS_WORD} in ${STYLE_RARE_TOKEN} style"
# content prompt 
export CONTENT_FORWARD_PROMPT="A ${CONTENT_RARE_TOKEN} ${CLASS_WORD}" 
# style prompt
export STYLE_FORWARD_PROMPT="A ${CLASS_WORD} in ${STYLE_RARE_TOKEN} style" 

# For validation
export VALID_CONTENT="A ${CONTENT_RARE_TOKEN} ${CLASS_WORD} on a table"
export VALID_PROMPT="A ${CONTENT_RARE_TOKEN} ${CLASS_WORD} on a table in ${STYLE_RARE_TOKEN} style"
export VALID_STYLE="A ${CLASS_WORD} in ${STYLE_RARE_TOKEN} style on a table"

# for content validation
export VALID_CONTENT_PROMPT="a photo of a ${CONTENT_RARE_TOKEN} ${CLASS_WORD} on a table"

# for style validation
export VALID_STYLE_PROMPT="A ${SUPER_WORD} in ${STYLE_RARE_TOKEN} style"

export WANDB_PROJECT="unziplora"

accelerate launch train_unziplora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --name="$WANDB_NAME" \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --content_forward_prompt="${CONTENT_FORWARD_PROMPT}" \
  --style_forward_prompt="${STYLE_FORWARD_PROMPT}" \
  --rank="${RANK}" \
  --resolution=1024 \
  --train_batch_size=1 \
  --content_learning_rate="${CONTENT_LR}" \
  --style_learning_rate="${STYLE_LR}" \
  --weight_learning_rate="$weight_lr" \
  --similarity_lambda="$similarity_lambda" \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps="$STEPS" \
  --checkpointing_steps=200 \
  --mixed_precision="fp16" \
  --seed="0" \
  --use_8bit_adam \
  --validation_content="${VALID_CONTENT}" \
  --validation_style="${VALID_STYLE}" \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_prompt_style="${VALID_STYLE_PROMPT}" \
  --validation_prompt_content="${VALID_CONTENT_PROMPT}" \
  --sample_times=$period_sample_epoch \
  --column_ratio=$sampled_column_ratio \
  --entity="elysiaareudi-discord" \
  --content_rare_word="${CONTENT_RARE_TOKEN}" \
  --class_word="${CLASS_WORD}" \
  --style_rare_word="${STYLE_RARE_TOKEN}" \
  --super_word="${SUPER_WORD}" \
  --heatmap_map_size=64 \
  --heatmap_alpha=0.45 \
  --heatmap_steps=100 \
  --focus='true' \
  --with_align_loss="true" \
  --align_loss_weight=1.0 \
  --with_gsa_loss="true" \
  --gsa_loss_weight=2.0 \
  --sig_type="last" \
  --use_base_weight="true" \
  --alpha=1.0 \
  --min_rank_content=32 \
  --min_rank_style=24 \
  --timestep_mode="${TIMESTEP_MODE}" \
  --use_time_control="true"