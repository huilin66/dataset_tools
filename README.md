# dataset_tools

## YOLO Instance Segmentation Risk Visualizer

A PyQt5-based application for visualizing YOLO instance segmentation masks with risk attributes and editing risk values.

### Features
- Load and display images with instance segmentation masks
- View and edit risk attributes for each segmentation mask
- Intuitive interface with four functional areas
- Real-time updates to label files when risk values change
- Support for custom YOLO format with risk attributes

### Installation
```bash
pip install -r isds_tool/PS_data/qt_win/yolo_risk_annotator/requirements.txt
```

### Running the Application
```bash
python isds_tool/PS_data/qt_win/yolo_risk_annotator/main.py
```

### Label File Format
Modified YOLO instance segmentation format with risk attributes:
```
<cat_id> <risk_num> <risk1> <risk2> <risk3> <risk4> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
Where:
- `<cat_id>`: Category ID of the object
- `<risk_num>`: Number of risk attributes (fixed at 4)
- `<risk1>` to `<risk4>`: Risk levels (0=No, 1=Medium, 2=High)
- `<x1> <y1> ... <xn> <yn>`: Polygon coordinates for the instance mask