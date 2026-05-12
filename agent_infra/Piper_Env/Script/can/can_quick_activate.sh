#!/bin/bash

###############################################################################
# 脚本名称: can_quick_activate.sh
# 脚本功能: 批量快速激活 CAN 口（按列表顺序调用 can_activate.sh）
# 使用方式:
#   1. 在下方配置 CAN_NAME_LIST 和 USB_PORT_LIST
#   2. 执行: sudo bash can_quick_activate.sh
###############################################################################

set -euo pipefail

# 比特率（按你的需求固定为 1000000）
BITRATE=1000000

# 用户自定义: CAN 名称列表（顺序与 USB_PORT_LIST 一一对应）
CAN_NAME_LIST=(
  "can_mr"
  "can_sr"
  "can_ml"
  "can_sl"

)

# 用户自定义: USB 硬件地址列表（通过 `ethtool -i canX | grep bus-info` 获取）
USB_PORT_LIST=(
  
 
  "1-2.3:1.0"
  "1-2.4:1.0"
  "1-3.1:1.0" 
  "1-3.2:1.0"
  

)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVATE_SCRIPT="$SCRIPT_DIR/can_activate.sh"

if [ ! -f "$ACTIVATE_SCRIPT" ]; then
  echo "错误: 未找到脚本 $ACTIVATE_SCRIPT"
  exit 1
fi

if [ "${#CAN_NAME_LIST[@]}" -eq 0 ]; then
  echo "错误: CAN_NAME_LIST 为空，请至少配置 1 项。"
  exit 1
fi

if [ "${#CAN_NAME_LIST[@]}" -ne "${#USB_PORT_LIST[@]}" ]; then
  echo "错误: 列表长度不一致。"
  echo "CAN_NAME_LIST 数量: ${#CAN_NAME_LIST[@]}"
  echo "USB_PORT_LIST 数量: ${#USB_PORT_LIST[@]}"
  exit 1
fi

echo "开始批量激活 CAN 接口，共 ${#CAN_NAME_LIST[@]} 路，bitrate=$BITRATE"

for i in "${!CAN_NAME_LIST[@]}"; do
  can_name="${CAN_NAME_LIST[$i]}"
  usb_port="${USB_PORT_LIST[$i]}"

  echo "[$((i + 1))/${#CAN_NAME_LIST[@]}] 激活: $can_name <- $usb_port"
  bash "$ACTIVATE_SCRIPT" "$can_name" "$BITRATE" "$usb_port"
done

echo "批量激活完成。"
