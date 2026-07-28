"""
YOLOE-26 模型推理预测脚本
参考: https://docs.ultralytics.com/zh/modes/predict/#推理参数
"""

from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("./runs/detect/runs/train_model/weights/best.pt")

# 推理数据源（支持：图片路径 / 目录 / 视频 / RTSP流 / YouTube / webcam=0）
source = "dataset_predict/QQ2026727-212554.mp4"  # TODO: 替换为实际图片/视频路径

# 推理
results = model.predict(
    source=source,

    # ========== 基础配置 ==========
    imgsz=640,                     # 推理图片尺寸
    conf=0.25,                     # 置信度阈值（低于此值的框不输出）
    iou=0.7,                       # NMS IoU 阈值
    max_det=300,                   # 每张图片最大检测数
    device=0,                      # GPU 设备号；CPU 填 "cpu"
    batch=1,                       # 推理 batch size（通常为 1）
    half=True,                     # FP16 半精度推理（省显存加速）

    # ========== 数据增强推理 ==========
    augment=False,                 # 测试时增强（TTA），提升精度但降低速度
    agnostic_nms=False,            # 类别无关 NMS（跨类别去重）

    # ========== 类别过滤 ==========
    classes=None,                  # 只检测指定类别；None=全部；示例: [0, 2, 5]

    # ========== 保存与可视化 ==========
    save=True,                     # 保存标注结果图片/视频
    save_txt=False,                # 保存检测结果为 TXT（YOLO 格式）
    save_conf=False,               # TXT 中附带置信度分数
    save_crop=False,               # 保存每个检测目标的裁剪图
    show=False,                    # 实时窗口显示结果
    show_labels=True,              # 结果图显示类别标签
    show_conf=True,                # 结果图显示置信度分数
    show_boxes=True,               # 结果图显示检测框
    line_width=None,               # 边框线宽；None 自动根据 imgsz 计算

    # ========== 视频/流专用 ==========
    vid_stride=1,                  # 视频帧间隔（1=每帧都处理，N=每隔N帧）
    stream=False,                  # True 时返回生成器，避免视频/流场景内存溢出
    stream_buffer=False,           # 流缓冲区（所有帧或混合所有线程帧）

    # ========== 输出 ==========
    project="./runs",              # 输出根目录
    name="predict_model",          # 实验名称（子目录）
    exist_ok=True,                 # 覆盖已有输出目录
    verbose=True,                  # 打印详细信息
)

# 遍历结果
for i, r in enumerate(results):
    if r.boxes is not None and len(r.boxes):
        # 打印检测信息
        boxes = r.boxes.xyxy.cpu().numpy()       # [x1, y1, x2, y2]
        confs = r.boxes.conf.cpu().numpy()        # 置信度
        clss  = r.boxes.cls.cpu().numpy().astype(int)  # 类别 ID
        names = model.names                        # 类别 ID → 名称映射

        print(f"\n图片 {i}: 检测到 {len(boxes)} 个目标")
        for box, conf, cls_id in zip(boxes, confs, clss):
            print(f"  {names.get(cls_id, cls_id):<15s}  conf={conf:.3f}  box={box}")
    else:
        print(f"\n图片 {i}: 未检测到目标")
