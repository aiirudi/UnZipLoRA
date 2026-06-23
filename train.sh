export MODEL_NAME="/home/xzh/xzh/pretrained/sd_xl_base_1.0"

#heatmap parameter
CONTENT_RARE_TOKEN="monadikos"          # 例如：xlo
SUPER_WORD='dog'

export content_dir="${content_dir:-origami_sunflower}"
export content_name="${content_name:-sunflower}"
export style_name="${style_name:-origami style}"

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
export WANDB_NAME="unziplora+M1+M2(fused)"
export INSTANCE_DIR="/home/xzh/xzh/UnZipLoRA-fine/instance_data/${content_dir}"
export OUTPUT_DIR="models/${content_dir}/${content_dir}"
export STEPS=600
# Training prom6t 

# both prompt
export PROMPT="A ${CONTENT_RARE_TOKEN} ${content_name} in ${style_name}"
# content prompt 
export CONTENT_FORWARD_PROMPT="A ${CONTENT_RARE_TOKEN} ${content_name}" 
# style prompt
export STYLE_FORWARD_PROMPT="A ${content_name} in ${style_name}" 

# For validation
export VALID_CONTENT="A ${CONTENT_RARE_TOKEN} ${content_name} on a table"
export VALID_PROMPT="A ${CONTENT_RARE_TOKEN} ${content_name} on a table in ${style_name}"
export VALID_STYLE="A ${content_name} in ${style_name} on a table"

# for content validation
export VALID_CONTENT_PROMPT="a photo of a ${CONTENT_RARE_TOKEN} ${content_name} on a table"

# for style validation
export VALID_STYLE_PROMPT="A ${SUPER_WORD} in ${style_name}"

export WANDB_PROJECT="unziplora"

# 模块启动参数配置
#for controlling block effect
# TFM (Token Focus Masking)
export focus_value="false"
# TAL
export align_loss_effect="true"
#gsa loss
export gsa_loss_effect="false"
# random matrix svd init
export with_svd_init_value="true"
# subb init value
export use_base_weight_value="true"
#unsymmetrical time control mask
export use_time_control_value="true"


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
  --class_word="${content_name}" \
  --style_rare_word="${style_name}" \
  --super_word="${SUPER_WORD}" \
  --heatmap_map_size=64 \
  --heatmap_alpha=0.5 \
  --heatmap_steps=100 \
  --focus="${focus_value}" \
  --with_align_loss="${align_loss_effect}" \
  --align_loss_weight=0.2 \
  --with_gsa_loss="${gsa_loss_effect}" \
  --gsa_loss_weight=0.4 \
  --sig_type="last" \
  --with_svd_init="${with_svd_init_value}" \
  --use_base_weight="${use_base_weight_value}" \
  --alpha=1.0 \
  --min_rank_content=32 \
  --min_rank_style=24 \
  --timestep_mode="${TIMESTEP_MODE}" \
  --use_time_control="${use_time_control_value}"
