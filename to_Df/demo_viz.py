import os
import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def main():
    # 1. 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "model/sam3.pt"
    
    print("=" * 50)
    print(f"正在加载 SAM3 模型... (设备: {device})")
    print(f"权重路径: {checkpoint_path}")
    print("=" * 50)
    
    # 建立模型
    model = build_sam3_image_model(checkpoint_path=checkpoint_path, device=device)
    processor = Sam3Processor(model=model, device=device)
    print("=> 模型加载完成！\n")

    # 2. 准备图片和提示词
    # 我们使用 datasets 目录下的第一张测试图
    image_path = "datasets/data1/000000000009.jpg"
    prompt = "bowl" # 你可以把这里换成 "broccoli" (西兰花) 或 "orange" (橙子) 来测试其他目标
    
    print(f"正在读取图片: {image_path}")
    print(f"搜索目标: '{prompt}'")
    
    if not os.path.exists(image_path):
        print(f"错误: 找不到图片 {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    
    # 3. 图像编码 (最耗时的一步，但一张图只需要做一次)
    print("正在对图像进行编码提取特征...")
    state = processor.set_image(image)

    # 4. 文本提示推理 (文本分支，速度很快)
    print("正在根据文本提示生成分割掩码...")
    state = processor.set_text_prompt(prompt=prompt, state=state)

    # 获取结果
    masks = state.get("masks")
    scores = state.get("scores")

    if masks is None or len(masks) == 0:
        print("=> 没有检测到任何目标。")
        return
        
    print(f"=> 找到了 {len(masks)} 个可能的目标实例。")
    
    # 5. 合并掩码并可视化
    # 比赛要求: 同一张图中同一类别的多个目标需要合并 (Union)
    scores_np = scores.detach().cpu().numpy().reshape(-1)
    merged_mask = None
    
    for index, score in enumerate(scores_np):
        if score >= 0.01: # 置信度阈值
            mask_np = masks[index].detach().cpu().numpy()
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            mask_np = (mask_np > 0).astype(np.uint8)
            
            if merged_mask is None:
                merged_mask = mask_np
            else:
                # 逻辑或 (Union) 运算合并掩码
                merged_mask = np.logical_or(merged_mask, mask_np).astype(np.uint8)
    
    if merged_mask is not None:
        print("=> 正在生成可视化图片...")
        # 将二值掩码转换为用于合成的灰度图
        mask_image = Image.fromarray(merged_mask * 255).convert("L")
        
        # 创建一个红色的遮罩层
        color_image = Image.new("RGB", image.size, (255, 0, 0))
        
        # 将红色遮罩和原图按照掩码形状进行合成
        result_image = Image.composite(color_image, image, mask_image)
        
        # 加一点透明度，让底部的图像能透出来，更好看
        final_image = Image.blend(image, result_image, alpha=0.6)
        
        output_path = f"demo_output_{prompt}.jpg"
        final_image.save(output_path)
        print(f"\n✅ 成功！请在资源管理器中打开查看结果图: {os.path.abspath(output_path)}")
    else:
        print("=> 所有检测结果的置信度都太低，未生成可视化图片。")

if __name__ == "__main__":
    main()
