"""
YOLOE-26 训练脚本（优化版）
数据集配置文件: data.yaml
参考: https://docs.ultralytics.com/zh/modes/train/.
"""

from ultralytics import YOLO

# 从 YAML 构建新模型 + 加载预训练权重
model = YOLO("yolo11n.pt")

# 数据集
data_source = "data.yaml"

# 训练
results = model.train(
    # ========== 基础配置 ==========
    data=data_source,
    val=False,  # ← 加上这行
    epochs=300,  # 训练总轮数
    imgsz=640,  # 输入图片尺寸
    batch=16,  # batch size，根据显存调整
    device=0,  # GPU 设备号；CPU 填 "cpu"；多 GPU 填 [0,1,2,3]
    workers=8,  # 数据加载线程数
    patience=50,  # 早停轮数（默认50，之前100等于禁用）
    save=True,  # 保存训练检查点
    save_period=10,  # 每N轮保存一次（减少磁盘占用）
    # ========== 优化器与学习率 ==========
    optimizer="auto",  # 优化器: SGD / Adam / AdamW / NAdam / auto
    lr0=0.01,  # 初始学习率 (SGD=0.01, Adam/AdamW=0.001)
    lrf=0.01,  # 最终学习率因子 (final_lr = lr0 × lrf)
    momentum=0.937,  # SGD动量 / Adam beta1
    weight_decay=0.0005,  # 权重衰减 (L2正则化)
    warmup_epochs=3.0,  # 学习率预热轮数（小数支持渐进式预热）
    warmup_momentum=0.8,  # 预热阶段初始动量
    warmup_bias_lr=0.1,  # 预热阶段偏置学习率
    cos_lr=True,  # 余弦退火LR调度（比线性衰减收敛更平滑）
    # ========== 损失函数增益 ==========
    box=7.5,  # 边界框回归损失增益
    cls=1.0,  # 分类损失增益
    dfl=1.5,  # Distribution Focal Loss 增益
    cls_pw=0.0,  # 类别权重幂 (0=禁用; 1=完全逆频率; 0.5=平方根)
    nbs=64,  # 名义批大小（损失归一化基准）
    # ========== 数据增强 - 颜色空间 ==========
    hsv_h=0.015,  # HSV 色调偏移幅度
    hsv_s=0.7,  # HSV 饱和度偏移幅度
    hsv_v=0.4,  # HSV 亮度偏移幅度
    # ========== 数据增强 - 几何变换 ==========
    degrees=0.0,  # 随机旋转角度 (±degrees)
    translate=0.1,  # 随机平移比例
    scale=0.5,  # 缩放增益 (范围: 1-scale ~ 1+scale)
    fliplr=0.5,  # 水平翻转概率
    flipud=0.0,  # 垂直翻转概率（通常不启用）
    # ========== 数据增强 - 混合增强 ==========
    mosaic=1.0,  # Mosaic 拼接增强（4图拼1图）
    close_mosaic=10,  # 最后N轮关闭Mosaic（消除mosaic带来的分布偏移）
    mixup=0.1,  # MixUp 混合增强（小比例提升泛化）
    cutmix=0.0,  # CutMix 增强（与mixup同时开启可能过度）
    # ========== 数据增强 - 多尺度 ==========
    multi_scale=0.3,  # 多尺度训练范围（±30% imgsz），提升尺度鲁棒性
    # ========== 性能 ==========
    amp=True,  # 自动混合精度（FP16），省显存加速
    cache=False,  # True=缓存数据集到内存（RAM充足时开启）
    # ========== 输出 ==========
    project="./runs",  # 输出根目录
    name="train_model",  # 实验名称（子目录）
    exist_ok=True,  # 覆盖已有输出目录
)
