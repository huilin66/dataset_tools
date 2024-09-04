import onnx
from onnx import version_converter, helper

# 加载模型
model_path = 'E:/repository/BD-Detection_ZM/Bin/Win64/plugin/pythonv3plugin/script/models/yolo8_leakage.onnx'
original_model = onnx.load(model_path)

# 将模型的 opset 版本从 17 降级到 15
converted_model = version_converter.convert_version(original_model, 15)

# 保存降级后的模型
converted_model_path = 'E:/repository/BD-Detection_ZM/Bin/Win64/plugin/pythonv3plugin/script/models/yolo8_leakage_opset15.onnx'
onnx.save(converted_model, converted_model_path)

print(f'Model converted and saved to {converted_model_path}')
