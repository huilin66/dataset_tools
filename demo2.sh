#!/bin/bash
set -e

# === 路径配置 ===
IMAGE_DIR="example/Fountain/images"
SPARSE_DIR="example/Fountain/sfm/0"
DENSE_DIR="example/dense"
MODEL_OUTPUT_DIR="$DENSE_DIR/model"

echo "🔧 Step 1: 图像矫正"
colmap image_undistorter \
    --image_path $IMAGE_DIR \
    --input_path $SPARSE_DIR \
    --output_path $DENSE_DIR \
    --output_type COLMAP

echo "🔧 Step 2: 深度图生成"
colmap patch_match_stereo \
    --workspace_path $DENSE_DIR \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

echo "🔧 Step 3: 点云融合"
colmap stereo_fusion \
    --workspace_path $DENSE_DIR \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DENSE_DIR/fused.ply

echo "🔧 Step 4: 网格重建（Poisson）"
colmap poisson_mesher \
    --input_path $DENSE_DIR/fused.ply \
    --output_path $DENSE_DIR/mesh.ply

echo "🔧 Step 5: 转换为 .obj + 贴图"
colmap model_converter \
    --input_path $DENSE_DIR \
    --output_path $MODEL_OUTPUT_DIR \
    --input_type Poisson \
    --output_type OBJ

echo "✅ 模型已生成：$MODEL_OUTPUT_DIR/model.obj"
echo "📦 包括：.obj（网格） .mtl（材质） .jpg（贴图）"
