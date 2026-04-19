export MODEL_NAME="/home/xzh/xzh/pretrained/sd_xl_base_1.0"

#heatmap parameter
RARE_TOKEN="monadikos"          # 例如：xlo
CLASS_WORD="cat"          # 例如：dog

# Hyper parameters
export period_sample_epoch=4
export sampled_column_ratio=0.125

# For weight similarity
export CONTENT_LR=0.00005
export STYLE_LR=0.00005
export weight_lr=0.005
export similarity_lambda=0.5
export RANK=64
export WANDB_NAME="unziplora"
export INSTANCE_DIR="/home/xzh/xzh/UnZipLoRA/instance_data/anime_cat"
export OUTPUT_DIR="models/anime_cat/anime_cat"
export STEPS=2

# Training prompt 

# both prompt
export PROMPT="A ${RARE_TOKEN} ${CLASS_WORD} in anime style"
# content prompt 
export CONTENT_FORWARD_PROMPT="A ${RARE_TOKEN} ${CLASS_WORD}" 
# style prompt
export STYLE_FORWARD_PROMPT="A ${CLASS_WORD} in anime style" 

# For validation
export VALID_CONTENT="A ${RARE_TOKEN} ${CLASS_WORD} on a table"
export VALID_PROMPT="A ${RARE_TOKEN} ${CLASS_WORD} on a table in anime style"
export VALID_STYLE="A ${CLASS_WORD} in anime style on a table"

# for content validation
export VALID_CONTENT_PROMPT="a photo of a ${RARE_TOKEN} ${CLASS_WORD} on a table"

# for style validation
export VALID_STYLE_PROMPT="A dog in anime style"

export WANDB_PROJECT="unziplora"

accelerate launch train_unziplora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --name=$WANDB_NAME \
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
  --rare_word="${RARE_TOKEN}" \
  --class_word="${CLASS_WORD}" \
  --heatmap_map_size=64 \
  --heatmap_alpha=0.45 \
  --heatmap_steps=100 \
  --with_image_per_validation