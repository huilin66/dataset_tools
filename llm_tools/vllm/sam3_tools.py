import gc
import os

import cv2
import torch
import torch._dynamo
import torch.nn as nn

# 1. 环境变量级封杀 Triton
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.disable = True

# 🌟🌟🌟 2. 核弹级补丁：劫持并修复 Ultralytics 8.4.37 的传参 Bug 🌟🌟🌟
_original_compile = torch.compile


def _safe_compile(model=None, **kwargs):
    # 如果发现框架愚蠢地把 False 传给了 mode 参数，立刻纠正为合法字符串
    if kwargs.get("mode") is False:
        kwargs["mode"] = "default"

    # 既然拦截了，顺手强行关闭 PyTorch 的编译功能，双重保险
    kwargs["disable"] = True

    # 将修复后的干净参数放行给真正的 PyTorch 底层
    return _original_compile(model, **kwargs)


# 狸猫换太子：全局替换 torch.compile
torch.compile = _safe_compile
from ultralytics.models.sam.predict import SAM3Predictor, SAM3SemanticPredictor


class InteractiveDecoderOnly(SAM3Predictor):
    """
    派生类：纯粹的特征解码器。
    特点：自己不提取特征，只接收外部注入的特征进行毫秒级解码。
    """

    def __init__(self, overrides=None):
        # 🌟 核心防坑：备份并保护 CUDA 环境变量
        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")

        cpu_overrides = overrides.copy() if overrides else {}
        cpu_overrides["device"] = "cpu"
        super().__init__(overrides=cpu_overrides)

        # 强制在 CPU 初始化，此时 Ultralytics 会流氓地把 CUDA_VISIBLE_DEVICES 改为 "-1"
        if self.model is None:
            self.setup_model(None)

        # 🌟 核心防坑：立刻恢复 CUDA 环境变量，防止主引擎找不到显卡！
        if old_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda

        self.model.image_encoder = nn.Identity()
        gc.collect()

        # 将剩下轻量级的“大脑(Decoder)”和预处理参数搬回真实的 GPU 上
        self.target_device = torch.device(overrides.get("device", "cuda:0"))
        self.model.to(self.target_device)
        self.device = self.target_device
        if hasattr(self, "mean"):
            self.mean = self.mean.to(self.target_device)
        if hasattr(self, "std"):
            self.std = self.std.to(self.target_device)

        # 依赖注入的槽位
        self.injected_features = None

    def get_im_features(self, im):
        """
        🌟 拦截底层的特征提取！
        框架流水线走到这里时，直接拿走我们注入好的特征，耗时 0 毫秒。
        """
        if self.injected_features is None:
            raise ValueError("Interactive Predictor: 请先注入特征！")
        return self.injected_features


class UnifiedSAM3Engine:
    """
    前端业务层唯一需要交互的封装器
    """

    def __init__(self, overrides):
        print("====== 正在加载 Semantic 主引擎 ======")
        self.semantic = SAM3SemanticPredictor(overrides=overrides)

        # 强制主引擎优先执行初始化，抢占正确的 GPU Context
        if self.semantic.model is None:
            self.semantic.setup_model(None)

        print("====== 正在加载 Interactive 解码器 (0 显存增加) ======")
        self.interactive = InteractiveDecoderOnly(overrides=overrides)

        # 🌟 内存缓存：存放解码后的图片矩阵，避免重复读取硬盘
        self.current_cv2_img = None

    def set_image(self, img_path):
        print("====== 正在提取并共享图像特征 ======")
        self.current_cv2_img = cv2.imread(img_path)

        # 1. 语义主引擎走完全套流程
        self.semantic.set_image(self.current_cv2_img)

        # 2. 取出专为 SAM2/Interactive 准备的隐藏特征
        semantic_feats = self.semantic.features
        if "sam2_backbone_out" in semantic_feats:
            sam2_feats = semantic_feats["sam2_backbone_out"]
        else:
            sam2_feats = semantic_feats

        # 🌟🌟🌟 补片修复：高分辨率特征通道降维 (256 -> 32/64) 🌟🌟🌟
        interactive_model = self.interactive.model
        if getattr(interactive_model, "use_high_res_features_in_sam", False):
            # 💡 核心修复：关闭梯度计算，完美兼容 Inference Tensor！
            with torch.no_grad():
                fpn_feats = list(sam2_feats["backbone_fpn"])
                fpn_feats[0] = interactive_model.sam_mask_decoder.conv_s0(fpn_feats[0])
                fpn_feats[1] = interactive_model.sam_mask_decoder.conv_s1(fpn_feats[1])
                sam2_feats = {**sam2_feats, "backbone_fpn": fpn_feats}

        # 3. 完美适配维度与格式
        _, vision_feats, _, feat_sizes = interactive_model._prepare_backbone_features(
            sam2_feats
        )
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *f_size)
            for feat, f_size in zip(vision_feats, feat_sizes)
        ]

        # 4. 特征注入
        self.interactive.injected_features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }
        print("====== 依赖注入完毕 ======")

    def __call__(self, **kwargs):
        """
        动态路由分发 (注意这里不再需要传 source=IMG_PATH)
        """
        text = kwargs.get("text")

        # 🌟 极致优化：直接传内存中的 numpy 数组给框架
        # 这样框架底层的 setup_source 会跳过硬盘读取，直接做内存级的 resize！
        kwargs["source"] = self.current_cv2_img
        kwargs["stream"] = False

        # 文本请求 -> 走主引擎
        if text and len(text) > 0:
            return self.semantic(**kwargs)

        # 点/框请求 -> 走解码器
        else:
            kwargs.pop("text", None)
            return self.interactive(**kwargs)


# ==========================================
# 🎯 极其简单的调用测试
# ==========================================
if __name__ == "__main__":
    IMG_PATH = r"E:\data\tp\ShopSign_1265\ShopSign_1265\image_1.jpg"
    SAM3_PATH = r"E:\repository\dataset_tools\llm_tools\vllm\sam3.pt"
    # Initialize predictor with configuration
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=SAM3_PATH,
        half=True,  # Use FP16 for faster inference
        save=True,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    predictor.set_image(IMG_PATH)
    results = predictor(text=["signboard", "cloth"])

    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=SAM3_PATH,
        half=True,  # Use FP16 for faster inference
        save=True,
    )
    predictor = SAM3Predictor(overrides=overrides)

    predictor.set_image(IMG_PATH)
    results = predictor(points=[900, 370], labels=[1])
    results = predictor(bboxes=[[480.0, 290.0, 590.0, 650.0]])

    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=SAM3_PATH,
        half=True,
        compile=False,
        save=False,
        show=False,
    )

    # 1. 实例化
    predictor = UnifiedSAM3Engine(overrides=overrides)

    # 2. 确认图片 (仅发生一次沉重的 ViT 计算)
    predictor.set_image(IMG_PATH)

    # 3. 后续交互 (秒级响应，纯内存运算)
    results = predictor(text=["signboard", "cloth"])
    results = predictor(points=[900, 370], labels=[1])
    results = predictor(bboxes=[[480.0, 290.0, 590.0, 650.0]])
