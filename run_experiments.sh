#!/bin/bash
# 快速运行脚本：自动化AB测试实验

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}DePLNet Volume-Compensated Gating${NC}"
echo -e "${GREEN}AB Testing Automation Script${NC}"
echo -e "${GREEN}========================================${NC}\n"

# 检查数据划分文件是否存在
check_data_splits() {
    local ratio=$1
    local fold=$2
    local file="./data/brats2021_${ratio}percent_fold_${fold}.json"

    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: Data split file not found: $file${NC}"
        echo -e "${YELLOW}Please run: python data/generate_splits.py --ratios $ratio${NC}"
        return 1
    fi
    return 0
}

# 运行单个实验
run_experiment() {
    local net_type=$1
    local labeled_rate=$2
    local gpu=$3

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Running Experiment${NC}"
    echo -e "${GREEN}Network: $net_type${NC}"
    echo -e "${GREEN}Labeled Rate: $labeled_rate%${NC}"
    echo -e "${GREEN}========================================${NC}\n"

    # 检查数据文件
    if ! check_data_splits $labeled_rate 1; then
        return 1
    fi

    # 修改config.py
    python -c "
import re

with open('config.py', 'r') as f:
    content = f.read()

# 更新net配置
content = re.sub(r'config\[\"net\"\] = \".*?\"',
                 f'config[\"net\"] = \"$net_type\"', content)

# 更新labeled_rate配置
content = re.sub(r'config\[\"labeled_rate\"\] = \d+',
                 f'config[\"labeled_rate\"] = $labeled_rate', content)

with open('config.py', 'w') as f:
    f.write(content)

print(f'Config updated: net={net_type}, labeled_rate={labeled_rate}%')
"

    # 运行训练
    echo -e "${YELLOW}Starting training...${NC}"
    python train.py --gpu $gpu

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Training completed successfully!${NC}"
    else
        echo -e "${RED}Training failed!${NC}"
        return 1
    fi

    # 运行测试（对每一折）
    echo -e "\n${YELLOW}Running tests for all folds...${NC}"
    for fold in {1..5}; do
        echo -e "${YELLOW}Testing fold $fold...${NC}"
        python test_improved.py --gpu $gpu --fold $fold --checkpoint best --save_nii
    done

    echo -e "${GREEN}Experiment completed!${NC}\n"
}

# 主函数
main() {
    # 默认参数
    GPU=${1:-0}

    echo -e "${YELLOW}Using GPU: $GPU${NC}\n"

    # 询问用户要运行哪些实验
    echo -e "${YELLOW}Select experiments to run:${NC}"
    echo "1) Full AB Test (4 models × 1 ratio)"
    echo "2) Multi-ratio Test (1 model × 4 ratios)"
    echo "3) Complete Test (4 models × 4 ratios)"
    echo "4) Single Experiment (custom)"
    read -p "Enter choice [1-4]: " choice

    case $choice in
        1)
            # AB测试：4种模型配置，20%标注
            echo -e "\n${GREEN}Running Full AB Test (20% labeled data)${NC}\n"
            MODELS=("MyNet_MoE_VC" "MyNet_MoE" "MyNet_NoMoE" "MyNet_NoDSP")
            RATIO=20

            for model in "${MODELS[@]}"; do
                run_experiment $model $RATIO $GPU
            done
            ;;

        2)
            # 多比例测试：完整模型，4种标注比例
            echo -e "\n${GREEN}Running Multi-ratio Test (MyNet_MoE_VC)${NC}\n"
            MODEL="MyNet_MoE_VC"
            RATIOS=(5 10 20 100)

            for ratio in "${RATIOS[@]}"; do
                run_experiment $MODEL $ratio $GPU
            done
            ;;

        3)
            # 完整测试：4种模型 × 4种比例
            echo -e "\n${GREEN}Running Complete Test (4 models × 4 ratios)${NC}\n"
            echo -e "${RED}Warning: This will take a very long time!${NC}"
            read -p "Are you sure? [y/N]: " confirm

            if [[ $confirm == [yY] ]]; then
                MODELS=("MyNet_MoE_VC" "MyNet_MoE" "MyNet_NoMoE" "MyNet_NoDSP")
                RATIOS=(5 10 20 100)

                for model in "${MODELS[@]}"; do
                    for ratio in "${RATIOS[@]}"; do
                        run_experiment $model $ratio $GPU
                    done
                done
            else
                echo "Cancelled."
                exit 0
            fi
            ;;

        4)
            # 自定义单个实验
            echo -e "\n${YELLOW}Custom Experiment${NC}"
            echo "Available models:"
            echo "  1) MyNet_MoE_VC (Full model)"
            echo "  2) MyNet_MoE (No VC gating)"
            echo "  3) MyNet_NoMoE (No MoE)"
            echo "  4) MyNet_NoDSP (No DSP)"
            read -p "Select model [1-4]: " model_choice

            case $model_choice in
                1) MODEL="MyNet_MoE_VC" ;;
                2) MODEL="MyNet_MoE" ;;
                3) MODEL="MyNet_NoMoE" ;;
                4) MODEL="MyNet_NoDSP" ;;
                *) echo "Invalid choice"; exit 1 ;;
            esac

            read -p "Enter labeled rate [5/10/20/100]: " RATIO

            run_experiment $MODEL $RATIO $GPU
            ;;

        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}All experiments completed!${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

# 运行主函数
main "$@"
