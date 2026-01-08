# utils/visualization.py
from PIL import Image, ImageDraw, ImageFont

def draw_box(img_pil, bboxes, labels, colors):
    """
    在 PIL 图片上绘制边界框
    bboxes: [[cls, score, x1, y1, x2, y2], ...] (Pixel coordinates)
    """
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(15, int(img_pil.width/50)))
    except IOError:
        font = ImageFont.load_default()

    for box in bboxes:
        cls_id = int(box[0])
        score = float(box[1])
        x1, y1, x2, y2 = box[2:6]
        uid = box[6]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_pil.width, x2), min(img_pil.height, y2)

        label_text = labels[cls_id] if cls_id < len(labels) else str(cls_id)
        color = colors[cls_id % len(colors)]
        
        if isinstance(color, str):
            fill_color = color
        else:
            fill_color = tuple(color[::-1]) 

        line_width = max(2, int(img_pil.width/300))
        draw.rectangle([x1, y1, x2, y2], outline=fill_color, width=line_width)
        
        text_content = f"id:{uid} - {label_text} {score:.2f}"
        left, top, right, bottom = draw.textbbox((x1, y1), text_content, font=font)
        draw.rectangle((left, top, right, bottom), fill=fill_color)
        draw.text((x1, y1), text_content, fill="white", font=font)
    
    return img_pil

def crop_box(img_pil, bboxes):
    """
    裁剪出检测框对应的图片
    """
    crops = []
    for box in bboxes:
        x1, y1, x2, y2 = box[2:6]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img_pil.width, int(x2)), min(img_pil.height, int(y2))
        
        if x2 > x1 and y2 > y1:
            crop = img_pil.crop((x1, y1, x2, y2))
            crops.append(crop)
        else:
            crops.append(Image.new('RGB', (50, 50), color='black'))
    return crops