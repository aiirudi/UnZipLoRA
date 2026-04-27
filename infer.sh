export MODEL_NAME="/home/xzh/xzh/pretrained/sd_xl_base_1.0"
export RANK=64
export NUM=1
export TIMESTEP_MODE="priecewise"


# 训练输出前缀
export OUTPUT_DIR="models/sketch_cat/sketch_cat" 
#保存图片路径
export SAVE_DIR="output/sketch_rose/"



export VALID_PROMPTS=(
  "a monadikos rose on a skateboard in sketch style"
  "a monadikos rose in a snowy landscape in sketch style"
)
VALID_PROMPT=$(IFS=,; echo "${VALID_PROMPTS[*]}")
export VALID_PROMPT

export VALID_STYLES=(
  "a rose on a skateboard in sketch style"
  "a rose in a snowy landscape in sketch style"
)

VALID_STYLE=$(IFS=,; echo "${VALID_STYLES[*]}")
export VALID_STYLE

export VALID_CONTENTS=(
  "a monadikos rose on a skateboard"
  "a monadikos rose in a snowy landscape"
)
VALID_CONTENT=$(IFS=,; echo "${VALID_CONTENTS[*]}")
export VALID_CONTENT

export VALID_CONTENT_RECON_PROMPTS=(
  "A photo of monadikos rose on a table"
  "A photo of monadikos rose in a beach"
)
VALID_CONTENT_RECON_PROMPT=$(IFS=,; echo "${VALID_CONTENT_RECON_PROMPTS[*]}")
export VALID_CONTENT_RECON_PROMPT

export VALID_STYLE_PROMPTS=(
  "A dog in sketch style"
  "A chair in sketch style"
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
  --with_svd_init="true" \
  --use_base_weight="true" \
  --alpha=1.0 \
  --min_rank_content=32 \
  --min_rank_style=24 \
  --timestep_mode="${TIMESTEP_MODE}" \
  --use_time_control="true"
