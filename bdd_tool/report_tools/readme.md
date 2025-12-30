report_tools/
├── main.py                # 程序入口：负责配置加载、多线程调度、连续任务管理
├── config.py              # 配置中心：包含传感器参数、阈值、路径
├── core/                  # 【核心逻辑】
│   ├── engine.py          # 任务调度器：负责 run 流程
│   └── processor.py       # 图像处理器：单张图像/图像对的处理逻辑 (GSD, 物理尺寸)
├── loaders/               # 【输入层】
│   ├── base_loader.py     # 抽象基类
│   ├── yolo_loader.py     # 现有的 YOLO 格式
│   └── pair_loader.py     # 未来：处理 RGB+T 图像对路径匹配的 Loader
├── utils/                 # 【工具层】
│   ├── metadata.py        # 核心：整合 pyexif, fallback 的元数据管理器
│   ├── geo_utils.py       # 地理/物理计算
│   ├── visualization.py   # 绘图逻辑
│   └── analysis.py        # 判定与统计
└── exporters/             # 【输出层】
    ├── base_exporter.py
    └── pdf_styles.py