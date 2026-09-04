"""
模型工厂：根据config配置创建不同的模型变体
支持AB测试：
1. MyNet_MoE_VC: 完整模型 (MoE + VolumeCompensatedGating + DSP)
2. MyNet_MoE: MoE + LinearGating + DSP
3. MyNet_NoMoE: 标准VNet + DSP
4. MyNet_NoDSP: MoE + VolumeCompensatedGating (无DSP)
"""

import torch
from config import config
from networks.MyNet_MoE.unet_moe_vc import VNetFullMoE
from networks.MyNet_MoE.unet import VNet


def create_model(ema=False, has_dropout=False, dropout_rate=0.5):
    """
    根据config创建模型

    参数:
        ema: 是否为EMA模型（参数detach）
        has_dropout: 是否使用dropout
        dropout_rate: dropout比率

    返回:
        model: 创建的模型（已移至CUDA）
    """
    net_type = config["net"]
    n_channels = len(config["all_modalities"])
    n_classes = config["num_class"]
    num_experts = config["num_experts"]

    # 根据net_type创建不同的模型
    if net_type == "MyNet_MoE_VC":
        # 完整模型: MoE + VolumeCompensatedGating + DSP
        net = VNetFullMoE(
            n_channels=n_channels,
            n_classes=n_classes,
            n_filters=16,
            normalization='batchnorm',
            has_dropout=has_dropout,
            dropout_rate=dropout_rate,
            num_experts=num_experts,
            use_vc_gating=True,
            feature_scaler=4
        )
        print(f"Created model: VNetFullMoE with VolumeCompensatedGating (experts={num_experts})")

    elif net_type == "MyNet_MoE":
        # MoE + LinearGating + DSP (无体积补偿)
        net = VNetFullMoE(
            n_channels=n_channels,
            n_classes=n_classes,
            n_filters=16,
            normalization='batchnorm',
            has_dropout=has_dropout,
            dropout_rate=dropout_rate,
            num_experts=num_experts,
            use_vc_gating=False,
            feature_scaler=4
        )
        print(f"Created model: VNetFullMoE with LinearGating (experts={num_experts})")

    elif net_type == "MyNet_NoMoE":
        # 标准VNet + DSP (无MoE)
        net = VNet(
            n_channels=n_channels,
            n_classes=n_classes,
            n_filters=16,
            normalization='batchnorm',
            has_dropout=has_dropout,
            dropout_rate=dropout_rate
        )
        print("Created model: Standard VNet (no MoE)")

    elif net_type == "MyNet_NoDSP":
        # MoE + VolumeCompensatedGating (无DSP)
        net = VNetFullMoE(
            n_channels=n_channels,
            n_classes=n_classes,
            n_filters=16,
            normalization='batchnorm',
            has_dropout=has_dropout,
            dropout_rate=dropout_rate,
            num_experts=num_experts,
            use_vc_gating=True,
            feature_scaler=4
        )
        print(f"Created model: VNetFullMoE with VolumeCompensatedGating, no DSP (experts={num_experts})")

    else:
        raise ValueError(f"Unknown net type: {net_type}. "
                        f"Supported: MyNet_MoE_VC, MyNet_MoE, MyNet_NoMoE, MyNet_NoDSP")

    model = net.cuda()

    # EMA模型需要detach参数
    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def get_model_info():
    """获取当前配置的模型信息"""
    net_type = config["net"]
    use_moe = config["use_moe"]
    use_vc_gating = config["use_vc_gating"]
    use_dsp = config["use_dsp"]

    info = {
        "net_type": net_type,
        "use_moe": use_moe,
        "use_vc_gating": use_vc_gating,
        "use_dsp": use_dsp,
        "num_experts": config["num_experts"] if use_moe else 0,
        "labeled_rate": config["labeled_rate"]
    }

    return info


if __name__ == "__main__":
    # 测试不同配置
    test_configs = ["MyNet_MoE_VC", "MyNet_MoE", "MyNet_NoMoE", "MyNet_NoDSP"]

    for net_type in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {net_type}")
        print(f"{'='*60}")

        config["net"] = net_type
        config["use_moe"] = "MoE" in net_type
        config["use_vc_gating"] = "VC" in net_type
        config["use_dsp"] = "NoDSP" not in net_type

        try:
            model = create_model()
            info = get_model_info()

            print(f"Model info: {info}")
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

            # 测试前向传播
            x = torch.randn(1, 4, 128, 128, 128).cuda()
            with torch.no_grad():
                out, feat = model(x)
                print(f"Output shape: {out.shape}, Feature shape: {feat.shape}")

            print("✓ Test passed")

        except Exception as e:
            print(f"✗ Test failed: {e}")
