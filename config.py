
config = dict()

# ==================== AB测试配置 ====================
config["net"] = "DePLNet"

# ==================== 数据配置 ====================
config["dataname"] = "brats2021"
config["root"] = "../../data/BraTS_2021"
config['image_shape'] = [128, 128, 128]
config["all_modalities"] = ["flair","t1ce", "t1", "t2"]

# ==================== 训练配置 ====================
config["max_iterations"] = 15000
config["num_class"] = 3
config["num_experts"] = 7
config["test_batch_size"] = 2
config["batch_size"] = 4
config["labeled_bs"] = 2

# ==================== 标注数据比例配置 ====================
# 可选值: 5, 10, 20, 100 (表示百分比)
config["labeled_rate"] = 20

# ==================== Memory Bank 边界特征配置 ====================
config["boundary_ratio"] = 0.3       # boundary features占memory bank的比例 (消融: 0.0~0.5)
config["boundary_kernel_size"] = 5   # 形态学膨胀/腐蚀核大小，5=约2 voxel宽的边界带