#!/bin/bash

###############################################################################
# 脚本名称: find_all_can_port.sh
# 脚本功能: 扫描并列出系统中所有物理连接的 CAN 接口及其对应的 USB 物理地址
# 使用场景: 
#   当你有多个相同的 USB-CAN 模块时，用于确定每个 canX 接口对应的具体 USB 插口。
# 使用方法: 
#   bash find_all_can_port.sh
# 输出示例:
#   Interface [can0] is physically connected to USB device: 1-1.2:1.0
###############################################################################

# 检查是否安装了ethtool
if ! dpkg -l | grep -q "ethtool"; then
    echo "Error: ethtool not detected in the system."
    echo "Please install ethtool using the following command:"
    echo "sudo apt update && sudo apt install ethtool"
    exit 1
fi

# Check if can-utils is installed
if ! dpkg -l | grep -q "can-utils"; then
    echo "Error: can-utils not detected in the system."
    echo "Please install can-utils using the following command:"
    echo "sudo apt update && sudo apt install can-utils"
    exit 1
fi

echo "Both ethtool and can-utils are installed."

# 遍历所有 CAN 类型的接口（包括重命名后的）
for iface in $(ip -br link show type can | awk '{print $1}'); do
    # 直接从 sysfs 读取物理设备的符号链接路径
    # 这能穿透重命名的层级，直接找到底层的 USB 设备号
    DEVICE_PATH=$(readlink -f /sys/class/net/"$iface"/device)
    
    if [ -z "$DEVICE_PATH" ]; then
        echo "Error: Unable to find device path for interface $iface."
        continue
    fi

    # 从路径中提取出 USB 总线信息 (通常是路径的最后一段)
    BUS_INFO=$(basename "$DEVICE_PATH")
    
    echo "Interface [$iface] is physically connected to USB device: $BUS_INFO"
done