"""
YOLOE-26 模型验证脚本（优化版）
数据集配置文件: data.yaml
参考: https://docs.ultralytics.com/zh/modes/val/#带参数的验证示例.
"""

from ultralytics import YOLO

# 加载训练好的模型（train.py 输出的 best.pt）
model = YOLO("./runs/detect/runs/train_model/weights/best.pt")

# 数据集
data_source = "data.yaml"

# 验证
metrics = model.val(
    # ========== 基础配置 ==========
    data=data_source,
    imgsz=640,  # 输入图片尺寸
    batch=16,  # batch size，根据显存调整
    device=0,  # GPU 设备号；CPU 填 "cpu"
    workers=8,  # 数据加载线程数
    split="test",  # 数据集划分: "val" / "test" / "train"
    # ========== 评估设置 ==========
    conf=0.001,  # 目标置信度阈值
    iou=0.6,  # NMS IoU 阈值
    max_det=300,  # 每张图片最大检测数
    half=True,  # FP16 半精度推理（省显存加速）
    # ========== 保存与可视化 ==========
    save_json=False,  # 保存预测结果为 JSON
    save_hybrid=False,  # 保存混合标签（标注 + 预测）
    plots=True,  # 绘制混淆矩阵、P-R 曲线等分析图表
    # ========== 数据集加载 ==========
    rect=True,  # 矩形推理：按宽高比分组批处理，减少 padding 开销
    # ========== 输出 ==========
    project="./runs",  # 输出根目录
    name="val_model",  # 实验名称（子目录）
    exist_ok=True,  # 覆盖已有输出目录
)

# 打印各项指标
print(f"mAP50-95: {metrics.box.map:.4f}")  # 平均精度 (IoU 0.5~0.95)
print(f"mAP50:    {metrics.box.map50:.4f}")  # 平均精度 (IoU 0.5)
print(f"mAP75:    {metrics.box.map75:.4f}")  # 平均精度 (IoU 0.75)
print(f"Precision: {metrics.box.mp:.4f}")  # 精确率
print(f"Recall:    {metrics.box.mr:.4f}")  # 召回率
