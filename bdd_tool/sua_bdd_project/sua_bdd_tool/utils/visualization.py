from PIL import Image, ImageDraw, ImageFont

def hex_to_rgb(hex_color: str):
    return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*map(int, rgb))


def blend_colors(c1_hex, c2_hex, ratio):
    c1, c2 = hex_to_rgb(c1_hex), hex_to_rgb(c2_hex)
    return rgb_to_hex([c1[i]*(1-ratio) + c2[i]*ratio for i in range(3)])


def get_dynamic_bearing_color(bearing_deg: float) -> str:
    COLORS = {0: "#FFD700", 90: "#FF0000", 180: "#008000", 270: "#0000FF", 360: "#FFD700"}
    b = bearing_deg % 360.0
    quadrant = int(b // 90)
    start_a, end_a = quadrant * 90, (quadrant + 1) * 90
    return blend_colors(COLORS[start_a], COLORS[end_a], (b - start_a) / 90.0)


def get_class_color(cls_idx, color_palette):
    """根据类别索引返回颜色"""
    if cls_idx < 0: return (255, 255, 255)
    return color_palette[cls_idx % len(color_palette)]

def get_contrasting_text_color(bg_color):
    """根据背景色亮度决定文字是黑还是白"""
    luminance = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]) / 255
    return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)


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