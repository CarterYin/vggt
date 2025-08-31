#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGGT多张图像3D重建脚本
使用VGGT模型对多张图像进行3D场景重建，生成相机参数、深度图和3D点云
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import json
from datetime import datetime

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
        raise FileNotFoundError(f"模型文件不存在: {local_model_path}，请先运行 simple_demo.py 下载模型")
    
    model.eval()
    model = model.to(device)
    print("模型加载完成！")
    
    return model, device, dtype

def multi_image_inference(image_folder, model, device, dtype, max_images=None):
    """
    对多张图像进行推断
    
    Args:
        image_folder: 图像文件夹路径
        model: VGGT模型
        device: 计算设备
        dtype: 数据类型
        max_images: 最大处理图像数量，None表示处理所有图像
        
    Returns:
        dict: 包含推断结果的字典
    """
    print(f"正在处理图像文件夹: {image_folder}")
    
    # 获取所有图像文件
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    
    image_paths = sorted(image_paths)
    
    if len(image_paths) == 0:
        raise ValueError(f"在 {image_folder} 中未找到图像文件")
    
    # 限制图像数量
    if max_images is not None and len(image_paths) > max_images:
        image_paths = image_paths[:max_images]
        print(f"限制处理图像数量为: {max_images}")
    
    print(f"找到 {len(image_paths)} 张图像:")
    for i, path in enumerate(image_paths):
        print(f"  {i+1}. {os.path.basename(path)}")
    
    # 加载和预处理图像
    print("正在加载和预处理图像...")
    images = load_and_preprocess_images(image_paths).to(device)
    print(f"图像预处理完成，形状: {images.shape}")
    
    # 进行推断
    print("正在进行多图像推断（CPU模式可能较慢，请耐心等待）...")
    start_time = datetime.now()
    
    with torch.no_grad():
        if device == "cpu":
            # CPU不支持autocast，直接推断
            predictions = model(images)
        else:
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
    
    end_time = datetime.now()
    inference_time = (end_time - start_time).total_seconds()
    print(f"推断完成！用时: {inference_time:.2f} 秒")
    
    # 转换相机参数
    print("正在转换相机参数...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], 
        images.shape[-2:]
    )
    
    # 从深度图生成3D点云
    print("正在生成3D点云...")
    # 获取深度图，保持正确的形状 (S, H, W, 1)
    depth_maps_tensor = predictions["depth"].squeeze(0)  # 移除batch维度: (1, S, H, W, 1) -> (S, H, W, 1)
    depth_maps = depth_maps_tensor.cpu().numpy()
    
    # 确保深度图有正确的形状 (S, H, W, 1)
    if depth_maps.ndim == 3:
        depth_maps = depth_maps[..., np.newaxis]  # (S, H, W) -> (S, H, W, 1)
    
    world_points_from_depth = unproject_depth_map_to_point_map(
        depth_maps, 
        extrinsic.cpu().numpy().squeeze(0), 
        intrinsic.cpu().numpy().squeeze(0)
    )
    
    # 对于可视化，我们需要squeeze掉最后一维
    depth_maps_for_vis = depth_maps.squeeze(-1)  # (S, H, W, 1) -> (S, H, W)
    
    # 整理结果 - 注意多图像的形状处理
    results = {
        'num_images': len(image_paths),
        'image_paths': image_paths,
        'inference_time': inference_time,
        'depth_maps': depth_maps_for_vis,  # 用于可视化的深度图 (S, H, W)
        'depth_confidence': predictions["depth_conf"].cpu().numpy().squeeze(0),
        'world_points': predictions["world_points"].cpu().numpy().squeeze(0),
        'world_points_confidence': predictions["world_points_conf"].cpu().numpy().squeeze(0),
        'world_points_from_depth': world_points_from_depth,
        'extrinsic': extrinsic.cpu().numpy().squeeze(0),
        'intrinsic': intrinsic.cpu().numpy().squeeze(0),
        'original_images': images.cpu().numpy()  # 保持原始形状 (S, C, H, W)
    }
    
    print("多图像推断完成！")
    return results

def visualize_multi_results(results, save_dir="multi_output"):
    """
    可视化多图像推断结果
    
    Args:
        results: 推断结果字典
        save_dir: 保存目录
    """
    print("正在生成多图像可视化结果...")
    
    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)
    
    num_images = results['num_images']
    
    # 1. 创建综合概览图
    fig, axes = plt.subplots(3, min(num_images, 6), figsize=(20, 12))
    if num_images == 1:
        axes = axes.reshape(3, 1)
    elif min(num_images, 6) == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(min(num_images, 6)):
        # 原图 - 现在形状是 (S, C, H, W)
        original_img = results['original_images'][i]  # (C, H, W)
        if original_img.ndim == 3 and original_img.shape[0] == 3:
            original_img = original_img.transpose(1, 2, 0)  # (H, W, C)
        original_img = np.clip(original_img, 0, 1)
        
        axes[0, i].imshow(original_img)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # 深度图
        depth_map = results['depth_maps'][i]
        if depth_map.ndim > 2:
            depth_map = depth_map.squeeze()
        im1 = axes[1, i].imshow(depth_map, cmap='viridis')
        axes[1, i].set_title(f'Depth {i+1}')
        axes[1, i].axis('off')
        
        # 深度置信度
        depth_conf = results['depth_confidence'][i]
        if depth_conf.ndim > 2:
            depth_conf = depth_conf.squeeze()
        im2 = axes[2, i].imshow(depth_conf, cmap='hot')
        axes[2, i].set_title(f'Confidence {i+1}')
        axes[2, i].axis('off')
    
    # 如果图像数量少于6张，隐藏多余的子图
    for i in range(min(num_images, 6), 6):
        if i < axes.shape[1]:
            for j in range(3):
                axes[j, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multi_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 单独保存每张图像的详细结果
    for i in range(num_images):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图 - 现在形状是 (S, C, H, W)
        original_img = results['original_images'][i]  # (C, H, W)
        if original_img.ndim == 3 and original_img.shape[0] == 3:
            original_img = original_img.transpose(1, 2, 0)  # (H, W, C)
        original_img = np.clip(original_img, 0, 1)
        axes[0].imshow(original_img)
        axes[0].set_title(f'Original Image {i+1}')
        axes[0].axis('off')
        
        # 深度图
        depth_map = results['depth_maps'][i]
        if depth_map.ndim > 2:
            depth_map = depth_map.squeeze()
        im1 = axes[1].imshow(depth_map, cmap='viridis')
        axes[1].set_title(f'Depth Map {i+1}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # 深度置信度
        depth_conf = results['depth_confidence'][i]
        if depth_conf.ndim > 2:
            depth_conf = depth_conf.squeeze()
        im2 = axes[2].imshow(depth_conf, cmap='hot')
        axes[2].set_title(f'Depth Confidence {i+1}')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'image_{i+1:02d}_analysis.png'), dpi=200, bbox_inches='tight')
        plt.close()
    
    # 3. 合并所有3D点云
    print("正在合并3D点云...")
    all_points = []
    all_colors = []
    
    for i in range(num_images):
        # 获取该图像的3D点
        world_points = results['world_points_from_depth'][i]
        
        # 获取对应的颜色 - 现在形状是 (S, C, H, W)
        original_img = results['original_images'][i]  # (C, H, W)
        if original_img.ndim == 3 and original_img.shape[0] == 3:
            color_img = original_img.transpose(1, 2, 0)  # (H, W, C)
        else:
            color_img = original_img
        color_img = np.clip(color_img, 0, 1)
        
        # 展平点云和颜色
        points_flat = world_points.reshape(-1, 3)
        colors_flat = color_img.reshape(-1, 3)
        
        # 过滤无效点
        valid_mask = ~np.isnan(points_flat).any(axis=1)
        valid_points = points_flat[valid_mask]
        valid_colors = colors_flat[valid_mask]
        
        all_points.append(valid_points)
        all_colors.append(valid_colors)
        
        print(f"图像 {i+1}: {len(valid_points)} 个有效3D点")
    
    # 合并所有点
    if all_points:
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        # 保存合并的点云
        save_combined_point_cloud(combined_points, combined_colors, 
                                 os.path.join(save_dir, 'combined_point_cloud.ply'))
        
        print(f"合并点云: {len(combined_points)} 个点")
    
    # 4. 保存相机参数
    camera_data = {
        'num_images': num_images,
        'image_paths': [os.path.basename(p) for p in results['image_paths']],
        'extrinsic_matrices': results['extrinsic'].tolist(),
        'intrinsic_matrices': results['intrinsic'].tolist(),
        'inference_time': results['inference_time']
    }
    
    # 保存为JSON格式（便于阅读）
    with open(os.path.join(save_dir, 'camera_parameters.json'), 'w', encoding='utf-8') as f:
        json.dump(camera_data, f, indent=2, ensure_ascii=False)
    
    # 保存为NPZ格式（便于加载）
    np.savez(os.path.join(save_dir, 'reconstruction_data.npz'), 
             extrinsic=results['extrinsic'],
             intrinsic=results['intrinsic'],
             depth_maps=results['depth_maps'],
             depth_confidence=results['depth_confidence'],
             world_points=results['world_points'],
             world_points_from_depth=results['world_points_from_depth'])
    
    print(f"多图像可视化结果已保存到: {save_dir}")
    print(f"- 综合概览: {os.path.join(save_dir, 'multi_overview.png')}")
    print(f"- 单张分析: {save_dir}/image_XX_analysis.png")
    print(f"- 合并点云: {os.path.join(save_dir, 'combined_point_cloud.ply')}")
    print(f"- 相机参数: {os.path.join(save_dir, 'camera_parameters.json')}")
    print(f"- 重建数据: {os.path.join(save_dir, 'reconstruction_data.npz')}")

def save_combined_point_cloud(points, colors, filename):
    """
    保存合并的点云为PLY格式
    
    Args:
        points: 3D点坐标 (N, 3)
        colors: RGB颜色 (N, 3), 范围[0,1]
        filename: 输出文件名
    """
    # 确保颜色在0-255范围内
    colors_255 = (colors * 255).astype(np.uint8)
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for point, color in zip(points, colors_255):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {color[0]} {color[1]} {color[2]}\n")

def analyze_camera_poses(results, save_dir="multi_output"):
    """
    分析和可视化相机位姿
    
    Args:
        results: 推断结果字典
        save_dir: 保存目录
    """
    print("正在分析相机位姿...")
    
    extrinsic = results['extrinsic']  # (N, 3, 4)
    num_cameras = extrinsic.shape[0]
    
    # 提取相机位置（世界坐标系中的位置）
    camera_positions = []
    camera_orientations = []
    
    for i in range(num_cameras):
        # 外参矩阵是 camera-from-world，所以需要求逆得到 world-from-camera
        R = extrinsic[i, :3, :3]  # 旋转矩阵
        t = extrinsic[i, :3, 3]   # 平移向量
        
        # 相机在世界坐标系中的位置
        camera_pos = -R.T @ t
        camera_positions.append(camera_pos)
        
        # 相机的朝向（光轴方向）
        camera_forward = R[2, :]  # 相机的z轴（朝向）
        camera_orientations.append(camera_forward)
    
    camera_positions = np.array(camera_positions)
    camera_orientations = np.array(camera_orientations)
    
    # 创建相机位姿可视化
    fig = plt.figure(figsize=(15, 5))
    
    # 3D相机位置图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
               c=range(num_cameras), cmap='viridis', s=100)
    
    # 添加相机朝向箭头
    for i, (pos, orient) in enumerate(zip(camera_positions, camera_orientations)):
        ax1.quiver(pos[0], pos[1], pos[2], 
                  orient[0], orient[1], orient[2], 
                  length=0.3, color='red', alpha=0.7)
        ax1.text(pos[0], pos[1], pos[2], f'  {i+1}', fontsize=8)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Camera Positions & Orientations')
    
    # XY平面投影
    ax2 = fig.add_subplot(132)
    ax2.scatter(camera_positions[:, 0], camera_positions[:, 1], 
               c=range(num_cameras), cmap='viridis', s=100)
    for i, pos in enumerate(camera_positions):
        ax2.annotate(f'{i+1}', (pos[0], pos[1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Camera Positions (Top View)')
    ax2.grid(True, alpha=0.3)
    
    # 相机间距分析
    ax3 = fig.add_subplot(133)
    distances = []
    for i in range(num_cameras):
        for j in range(i+1, num_cameras):
            dist = np.linalg.norm(camera_positions[i] - camera_positions[j])
            distances.append(dist)
    
    ax3.hist(distances, bins=min(20, len(distances)), alpha=0.7)
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Inter-camera Distances')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'camera_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 打印统计信息
    print(f"\n=== 相机位姿分析 ===")
    print(f"相机数量: {num_cameras}")
    print(f"相机位置范围:")
    print(f"  X: {camera_positions[:, 0].min():.3f} ~ {camera_positions[:, 0].max():.3f}")
    print(f"  Y: {camera_positions[:, 1].min():.3f} ~ {camera_positions[:, 1].max():.3f}")
    print(f"  Z: {camera_positions[:, 2].min():.3f} ~ {camera_positions[:, 2].max():.3f}")
    print(f"平均相机间距: {np.mean(distances):.3f}")
    print(f"相机间距范围: {np.min(distances):.3f} ~ {np.max(distances):.3f}")

def main():
    """
    主函数
    """
    print("VGGT 多图像3D重建脚本")
    print("=" * 60)
    
    # 图像文件夹路径
    image_folder = "examples/room/images"
    
    if not os.path.exists(image_folder):
        print(f"❌ 图像文件夹不存在: {image_folder}")
        print("请确保图像文件夹路径正确")
        return
    
    try:
        # 设置模型
        model, device, dtype = setup_model()
        
        # 进行多图像推断
        # 注意：处理所有8张图像，如果内存不足可以设置max_images=4
        results = multi_image_inference(image_folder, model, device, dtype, max_images=None)
        
        # 可视化结果
        output_dir = f"multi_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        visualize_multi_results(results, output_dir)
        
        # 分析相机位姿
        analyze_camera_poses(results, output_dir)
        
        # 打印统计信息
        print(f"\n=== 多图像重建统计 ===")
        print(f"处理图像数量: {results['num_images']}")
        print(f"推断耗时: {results['inference_time']:.2f} 秒")
        print(f"每张图像平均耗时: {results['inference_time']/results['num_images']:.2f} 秒")
        print(f"图像分辨率: {results['original_images'].shape[-2:]} (H×W)")
        
        total_valid_points = 0
        for i in range(results['num_images']):
            world_points = results['world_points_from_depth'][i]
            valid_points = (~np.isnan(world_points).any(axis=-1)).sum()
            total_valid_points += valid_points
            print(f"图像 {i+1} 有效3D点数: {valid_points:,}")
        
        print(f"总计有效3D点数: {total_valid_points:,}")
        print(f"平均每张图像3D点数: {total_valid_points//results['num_images']:,}")
        
        print(f"\n✅ 多图像3D重建完成！")
        print(f"结果保存在: {output_dir}/")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请检查:")
        print("1. 模型文件是否存在 (checkpoints/model.pt)")
        print("2. 图像文件夹是否存在且包含图像")
        print("3. 是否有足够的内存")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
