import torch
import torch.nn.functional as F

# 假设 pred_attributes 和 gt_attributes 是模型的预测值和真实标签
pred_attributes = torch.tensor([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,]).reshape(1,-1).float()
torch.manual_seed(1111)
gt_attributes = torch.randint(0, 2, (1, 14)).float()  # 示例真实标签
print(pred_attributes)
print(gt_attributes)

# 设置正例样本的权重，使得模型更加关注正例样本
pos_weight = torch.tensor([10.0])  # 这里设置了权重为 10，可以根据需要调整

# 计算加权二元交叉熵损失
loss = F.binary_cross_entropy_with_logits(
    input=pred_attributes,
    target=gt_attributes,
    pos_weight=pos_weight
)

print(f"Loss: {loss.item()}")
