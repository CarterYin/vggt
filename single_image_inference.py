#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGGT单张图像推断脚本
使用VGGT模型对单张图像进行3D重建和深度估计
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 导入VGGT相关模块
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def setup_model():
    """
    设置和加载VGGT模型
    """
    print("正在初始化VGGT模型...")
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 设置数据类型 - CPU使用float32
    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"使用数据类型: {dtype}")
    
    # 初始化模型
    model = VGGT()
    
    # 加载预训练权重
    local_model_path = "checkpoints/model.pt"
    if os.path.exists(local_model_path):
        print(f"从本地加载模型权重: {local_model_path}")
        model.load_state_dict(torch.load(local_model_path, map_location=device))
    else:
        print("本地模型文件不存在，正在下载...")
        _URL = "https://xget.xi-xu.me/hf/facebook/VGGT-1B/resolve/main/model.pt"
        # 下载到本地checkpoints文件夹
        import urllib.request
        os.makedirs("checkpoints", exist_ok=True)
        print(f"正在下载模型权重到: {local_model_path}")
        urllib.request.urlretrieve(_URL, local_model_path)
        print("下载完成，正在加载...")
        model.load_state_dict(torch.load(local_model_path, map_location=device))
    
    model.eval()
    model = model.to(device)
    print("模型加载完成！")
    
    return model, device, dtype

def single_image_inference(image_path, model, device, dtype):
    """
    对单张图像进行推断
    
    Args:
        image_path: 图像路径
        model: VGGT模型
        device: 计算设备
        dtype: 数据类型
        
    Returns:
        dict: 包含推断结果的字典
    """
    print(f"正在处理图像: {image_path}")
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    # 加载和预处理图像
    images = load_and_preprocess_images([image_path]).to(device)
    print(f"图像预处理完成，形状: {images.shape}")
    
    # 进行推断
    print("正在进行推断...")
    with torch.no_grad():
        if device == "cpu":
            # CPU不支持autocast，直接推断
            predictions = model(images)
        else:
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
    
    # 转换相机参数
    print("正在转换相机参数...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], 
        images.shape[-2:]
    )
    
    # 从深度图生成3D点云
    print("正在生成3D点云...")
    depth_map = predictions["depth"].cpu().numpy().squeeze(0)  # 移除batch维度
    world_points_from_depth = unproject_depth_map_to_point_map(
        depth_map, 
        extrinsic.cpu().numpy().squeeze(0), 
        intrinsic.cpu().numpy().squeeze(0)
    )
    
    # 整理结果
    results = {
        'depth_map': depth_map,
        'depth_confidence': predictions["depth_conf"].cpu().numpy().squeeze(0),
        'world_points': predictions["world_points"].cpu().numpy().squeeze(0),
        'world_points_confidence': predictions["world_points_conf"].cpu().numpy().squeeze(0),
        'world_points_from_depth': world_points_from_depth,
        'extrinsic': extrinsic.cpu().numpy().squeeze(0),
        'intrinsic': intrinsic.cpu().numpy().squeeze(0),
        'original_image': images.cpu().numpy().squeeze(0)
    }
    
    print("推断完成！")
    return results

def visualize_results(results, save_dir="output"):
    """
    可视化推断结果
    
    Args:
        results: 推断结果字典
        save_dir: 保存目录
    """
    print("正在生成可视化结果...")
    
    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 可视化深度图
    plt.figure(figsize=(15, 5))
    
    # 原图
    plt.subplot(1, 3, 1)
    original_img = results['original_image']
    # 处理图像形状: (C, H, W) -> (H, W, C)
    if original_img.ndim == 3 and original_img.shape[0] == 3:
        original_img = original_img.transpose(1, 2, 0)
    # 确保数值在[0,1]范围内
    original_img = np.clip(original_img, 0, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # 深度图
    plt.subplot(1, 3, 2)
    depth_map = results['depth_map']
    # 确保是2D数组
    if depth_map.ndim > 2:
        depth_map = depth_map.squeeze()
    plt.imshow(depth_map, cmap='viridis')
    plt.title('Depth Map')
    plt.colorbar()
    plt.axis('off')
    
    # 深度置信度
    plt.subplot(1, 3, 3)
    depth_conf = results['depth_confidence']
    # 确保是2D数组
    if depth_conf.ndim > 2:
        depth_conf = depth_conf.squeeze()
    plt.imshow(depth_conf, cmap='hot')
    plt.title('Depth Confidence')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'depth_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 保存3D点云数据
    world_points = results['world_points_from_depth']
    valid_mask = ~np.isnan(world_points).any(axis=-1)
    valid_points = world_points[valid_mask]
    
    # 保存为PLY格式（可用于3D软件查看）
    ply_path = os.path.join(save_dir, 'point_cloud.ply')
    save_point_cloud_ply(valid_points, ply_path)
    
    # 3. 保存相机参数
    camera_params = {
        'extrinsic': results['extrinsic'],
        'intrinsic': results['intrinsic']
    }
    np.savez(os.path.join(save_dir, 'camera_params.npz'), **camera_params)
    
    print(f"可视化结果已保存到: {save_dir}")
    print(f"- 深度可视化: {os.path.join(save_dir, 'depth_visualization.png')}")
    print(f"- 3D点云: {os.path.join(save_dir, 'point_cloud.ply')}")
    print(f"- 相机参数: {os.path.join(save_dir, 'camera_params.npz')}")

def save_point_cloud_ply(points, filename):
    """
    保存点云为PLY格式
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

def main():
    """
    主函数
    """
    # 示例图像路径 - 请替换为您的图像路径
    image_path = "examples/single_oil_painting/images/model_was_never_trained_on_single_image_or_oil_painting.png"
    
    # 如果示例图像不存在，提供帮助信息
    if not os.path.exists(image_path):
        print("示例图像不存在，请指定您的图像路径:")
        print("修改 image_path 变量为您的图像文件路径")
        print("支持的格式: .jpg, .jpeg, .png, .bmp 等")
        return
    
    try:
        # 设置模型
        model, device, dtype = setup_model()
        
        # 进行推断
        results = single_image_inference(image_path, model, device, dtype)
        
        # 可视化结果
        visualize_results(results)
        
        # 打印一些统计信息
        print("\n=== 推断结果统计 ===")
        print(f"图像尺寸: {results['original_image'].shape}")
        print(f"深度范围: {results['depth_map'].min():.3f} - {results['depth_map'].max():.3f}")
        print(f"平均深度置信度: {results['depth_confidence'].mean():.3f} (注：VGGT置信度通常>1，数值越高表示预测越可靠)")
        print(f"有效3D点数量: {(~np.isnan(results['world_points_from_depth']).any(axis=-1)).sum()}")
        
    except Exception as e:
        print(f"运行出错: {e}")
        print("请检查:")
        print("1. 图像路径是否正确")
        print("2. 是否有足够的GPU内存")
        print("3. 网络连接是否正常（用于下载模型权重）")

if __name__ == "__main__":
    main()
