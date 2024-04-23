import gradio as gr
from transformers import pipeline
import time, os

def greet(name):
    return "Hello " + name + "!"

def process_data(text, image):
    # 假设这里有数据处理逻辑
    processed_text = text.upper()
    return processed_text, image


def classify_image(model, img):
    return {i['label']: i['score'] for i in model(img)}

def process_image(img, filter_type):
    if filter_type == "Black and White":
        img = img.convert("L")
    return img

def slow_echo(message, history):
    for i in range(len(message)):
        time.sleep(0.05)
        yield "机器人回复: " + message[: i+1]

def function1(input1):
    return f"处理结果: {input1}"

def function2(input2):
    return f"分析结果: {input2}"

# iface = gr.Interface(
#     fn=greet,
#     inputs=gr.Textbox(),
#     outputs=gr.Textbox(),
#     title="简单问候",
#     description="输入你的名字，获得个性化问候。"
# )

# iface = gr.Interface(
#     fn=process_data,
#     inputs=[gr.Textbox(label="get text"), gr.Image(label="upload image")],
#     outputs=[gr.Textbox(label="output text"), gr.Image(label="processed image")],
#     title="get input",
#     description="get input of different types"
# )


# model = pipeline('image-classification')
# iface = gr.Interface(
#     fn=classify_image,
#     inputs=[model, gr.Image(type="pil")],
#     outputs=gr.Label(num_top_classes=5))

# iface = gr.Interface(
#     fn=process_image,
#     inputs=[gr.Image(type="pil"), gr.Radio(["None", "Black and White"])],
#     outputs="image"
# )

# iface = gr.ChatInterface(slow_echo).queue()

#
# iface1 = gr.Interface(function1, "text", "text")
# iface2 = gr.Interface(function2, "text", "text")
# iface= gr.TabbedInterface([iface1, iface2], ["界面1", "界面2"])
# iface.launch()


# import os
#
# import gradio as gr
# import tempfile
# import shutil
# def generate_file(file_obj):
#     global tmpdir
#     print('临时文件夹地址：{}'.format(tmpdir))
#     print('上传文件的地址：{}'.format(file_obj.name)) # 输出上传后的文件在gradio中保存的绝对地址
#
#     #获取到上传后的文件的绝对路径后，其余的操作就和平常一致了
#
#     # 将文件复制到临时目录中
#     shutil.copy(file_obj.name, tmpdir)
#
#     # 获取上传Gradio的文件名称
#     FileName=os.path.basename(file_obj.name)
#
#     # 获取拷贝在临时目录的新的文件地址
#     NewfilePath=os.path.join(tmpdir,FileName)
#     print(NewfilePath)
#
#     # 打开复制到新路径后的文件
#     with open(NewfilePath, 'rb') as file_obj:
#
#         #在本地电脑打开一个新的文件，并且将上传文件内容写入到新文件
#         outputPath=os.path.join(tmpdir,"New"+FileName)
#         with open(outputPath,'wb') as w:
#             w.write(file_obj.read())
#
#     # 返回新文件的的地址（注意这里）
#     return outputPath
# def main():
#     global tmpdir
#     with tempfile.TemporaryDirectory(dir='.') as tmpdir:
#         # 定义输入和输出
#         inputs = gr.components.File(label="上传文件")
#         outputs = gr.components.File(label="下载文件")
#
#         # 创建 Gradio 应用程序g
#         app = gr.Interface(fn=generate_file, inputs=inputs, outputs=outputs,   title="文件上传、并生成可下载文件demo",
#                       description="上传任何文件都可以，只要大小别超过你电脑的内存即可"
#       )
#
#         # 启动应用程序
#         app.launch(share=True)
# if __name__=="__main__":
#     main()


import gradio as gr
import pandas as pd

def save_chat(user_input, assistant_output, timestamp):
    global df
    # 将对话添加到DataFrame
    df = df.append({"User": user_input, "Assistant": assistant_output, "Time": timestamp}, ignore_index=True)
    return assistant_output

def get_data():
    # 返回对话的表格形式
    return df

def download_data():
    # 将对话保存为csv文件
    if not os.path.exists('chat.csv'):
        df.to_csv('chat.csv')
    return 'chat.csv'

# iface = gr.ChatInterface(slow_echo).queue()
#
# iface.add_button("Get Data", get_data)
# iface.add_button("Download Data", download_data)
#
# iface.launch()

# # 将聊天历史记录转换为pandas DataFrame
# df = pd.DataFrame(chat_history)
#
# # 将DataFrame保存为CSV文件
# df.to_csv('chat_history.csv', index=False)
#
# # 使用Gradio创建一个下载链接
# def download_link():
#     return 'chat_history.csv'

# iface = gr.Interface(fn=download_link, inputs=[], outputs='file')


from pathlib import Path
import gradio as gr


def upload_file(filepath):
    name = Path(filepath).name
    return [gr.UploadButton(visible=True), gr.DownloadButton(label=f"Download {name}", value=filepath, visible=True)]


def download_file():
    # gr.DownloadButton(label=f"Download", value=r'E:\record.xlsx', visible=True)
    return [gr.UploadButton(visible=True), gr.DownloadButton(label=f"Download", value=r'E:\record.xlsx', visible=True)]


with gr.Blocks() as demo:
    gr.Markdown("First upload a file and and then you'll be able download it (but only once!)")
    with gr.Row():
        u = gr.UploadButton("Upload a file", file_count="single")
        d = gr.DownloadButton("Download the file", visible=True)

    u.upload(upload_file, u, [u, d])
    d.click(download_file, None, [u, d])

if __name__ == "__main__":
    demo.launch()
