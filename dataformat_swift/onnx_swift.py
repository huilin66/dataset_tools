import onnx
from onnx import version_converter, helper

# 加载模型
model_path = r'E:\\repository\\BD-Detection_ZM\\Bin\\Win64\\plugin\\pythonv3plugin\\script\\models\\yolo9_ff.onnx'
model = onnx.load(model_path)

# 获取当前 opset 版本
original_opset_version = model.opset_import[0].version
print(f'Original Opset Version: {original_opset_version}')

# 定义目标 opset 版本
target_opset_version = 15

# 转换模型到目标 opset 版本
converted_model = version_converter.convert_version(model, target_opset_version)

# 保存转换后的模型
converted_model_path = r'E:\\repository\\BD-Detection_ZM\\Bin\\Win64\\plugin\\pythonv3plugin\\script\\models\\yolo9_ff_opset15.onnx'
onnx.save(converted_model, converted_model_path)

print(f'Model saved to {converted_model_path} with Opset Version: {target_opset_version}')