export MODEL_NAME="/home/xzh/xzh/pretrained/sd_xl_base_1.0"
export RANK=64
export NUM=1
export TIMESTEP_MODE="priecewise"

export content_dir="${content_dir:-tattoo_christmas}"
export content_name="${content_name:-Christmas tree}"
export style_name="${style_name:-tattoo art style}"

# 训练输出前缀
export OUTPUT_DIR="models/${content_dir}/${content_dir}" 
#保存图片路径
export SAVE_DIR="output/${content_dir}/"

# 模块启动参数配置
#for controlling block effect
# TFM (Token Focus Masking)
export focus_value="true"
# random matrix svd init
export with_svd_init_value="true"
# subb init value
export use_base_weight_value="false"
#unsymmetrical time control mask
export use_time_control_value="false"


export VALID_PROMPTS=(
  "a monadikos ${content_name} on a skateboard in ${style_name}"
  "a monadikos ${content_name} in a snowy landscape in ${style_name}"
)
VALID_PROMPT=$(IFS=,; echo "${VALID_PROMPTS[*]}")
export VALID_PROMPT

export VALID_STYLES=(
  "a ${content_name} on a skateboard in ${style_name}"
  "a ${content_name} in a snowy landscape in ${style_name}"
)

VALID_STYLE=$(IFS=,; echo "${VALID_STYLES[*]}")
export VALID_STYLE

export VALID_CONTENTS=(
  "a monadikos ${content_name} on a skateboard"
  "a monadikos ${content_name} in a snowy landscape"
)
VALID_CONTENT=$(IFS=,; echo "${VALID_CONTENTS[*]}")
export VALID_CONTENT

export VALID_CONTENT_RECON_PROMPTS=(
  "A photo of monadikos ${content_name} on a table"
  "A photo of monadikos ${content_name} in a beach"
)
VALID_CONTENT_RECON_PROMPT=$(IFS=,; echo "${VALID_CONTENT_RECON_PROMPTS[*]}")
export VALID_CONTENT_RECON_PROMPT

export VALID_STYLE_PROMPTS=(
  "A dog in ${style_name}"
  "A chair in ${style_name}"
)
VALID_STYLE_PROMPT=$(IFS=,; echo "${VALID_STYLE_PROMPTS[*]}")
export VALID_STYLE_PROMPT

accelerate launch infer.py \
  --output_dir="$OUTPUT_DIR" \
  --rank="${RANK}" \
  --num="${NUM}" \
  --with_unziplora \
  --save_dir="$SAVE_DIR" \
  --validation_prompt_content_recontext="${VALID_CONTENT_RECON_PROMPT}" \
  --validation_prompt_style="${VALID_STYLE_PROMPT}" \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_prompt_style_forward="${VALID_STYLE}" \
  --validation_prompt_content_forward="${VALID_CONTENT}" \
  --with_svd_init="${with_svd_init_value}" \
  --use_base_weight="${use_base_weight_value}" \
  --alpha=1.0 \
  --focus_value="${focus_value}" \
  --content_rare_word="monadikos" \
  --min_rank_content=32 \
  --min_rank_style=24 \
  --timestep_mode="${TIMESTEP_MODE}" \
  --use_time_control="${use_time_control_value}"