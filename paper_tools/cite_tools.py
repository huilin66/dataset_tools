import difflib
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import requests


def clean_title_for_compare(title):
    """清洗标题：转小写，去除所有标点符号和多余空格，用于高精度对比"""
    if not title:
        return ""
    title = re.sub(r"[^\w\s]", "", title).lower()
    return " ".join(title.split())


def is_title_match(target_title, retrieved_title, threshold=0.85):
    """计算两个标题的相似度，大于阈值则认为匹配"""
    t1 = clean_title_for_compare(target_title)
    t2 = clean_title_for_compare(retrieved_title)
    if not t1 or not t2:
        return False
    return difflib.SequenceMatcher(None, t1, t2).ratio() >= threshold


def format_bibtex(bib_string):
    """轻量级解析器：将单行 BibTeX 转换为标准的多行缩进格式"""
    bib_string = bib_string.strip()
    start_idx = bib_string.find("{")
    if start_idx == -1:
        return bib_string
    first_comma = bib_string.find(",", start_idx)
    if first_comma == -1:
        return bib_string

    formatted = bib_string[: first_comma + 1] + "\n"
    rest = bib_string[first_comma + 1 : -1].strip()

    fields, current_field, bracket_count = [], "", 0
    for char in rest:
        if char == "{":
            bracket_count += 1
        elif char == "}":
            bracket_count -= 1

        if char == "," and bracket_count == 0:
            fields.append(current_field.strip())
            current_field = ""
        else:
            current_field += char

    if current_field.strip():
        fields.append(current_field.strip())

    for i, field in enumerate(fields):
        if field:
            formatted += "  " + field
            if i < len(fields) - 1:
                formatted += ",\n"
            else:
                formatted += "\n"

    formatted += "}"
    return formatted


def search_crossref(title):
    print(f"  🔍 尝试 Crossref (正式发表数据库)...")
    query_url = (
        f"https://api.crossref.org/works?query.title={urllib.parse.quote(title)}&rows=3"
    )

    try:
        response = requests.get(query_url, timeout=10)
        items = response.json().get("message", {}).get("items", [])

        for item in items:
            retrieved_titles = item.get("title", [])
            if not retrieved_titles:
                continue
            retrieved_title = retrieved_titles[0]

            if is_title_match(title, retrieved_title):
                doi = item.get("DOI")
                if not doi:
                    continue

                bib_url = f"https://doi.org/{doi}"
                headers = {"Accept": "application/x-bibtex"}
                bib_response = requests.get(bib_url, headers=headers, timeout=10)

                if bib_response.status_code == 200:
                    raw_bib = bib_response.content.decode("utf-8")
                    return format_bibtex(raw_bib)
    except Exception:
        pass
    return None


def search_arxiv(title):
    print(f"  🔄 尝试 arXiv (预印本数据库)...")
    OAI = "{http://www.w3.org/2005/Atom}"
    ARXIV = "{http://arxiv.org/schemas/atom}"

    query = f'ti:"{title}"'
    url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&max_results=1"

    try:
        response = urllib.request.urlopen(url).read()
        root = ET.fromstring(response)
        entry = root.find(f"{OAI}entry")
        if entry is None:
            return None

        retrieved_title = entry.find(f"{OAI}title").text.replace("\n", " ").strip()

        if not is_title_match(title, retrieved_title):
            return None

        authors = [
            author.find(f"{OAI}name").text for author in entry.findall(f"{OAI}author")
        ]
        published = entry.find(f"{OAI}published").text
        year = published[:4]

        paper_id_url = entry.find(f"{OAI}id").text
        paper_id = paper_id_url.split("/abs/")[-1]

        primary_category = entry.find(f"{ARXIV}primary_category")
        category = (
            primary_category.attrib.get("term", "")
            if primary_category is not None
            else ""
        )

        first_author_last = authors[0].split()[-1] if authors else "Unknown"
        cite_key = f"{first_author_last}{year}{paper_id.split('v')[0].replace('.', '')}"
        authors_str = " and ".join(authors)

        bibtex = f"""@misc{{{cite_key},
  title={{{retrieved_title}}},
  author={{{authors_str}}},
  year={{{year}}},
  eprint={{{paper_id}}},
  archivePrefix={{arXiv}},
  primaryClass={{{category}}}
}}"""
        return bibtex
    except Exception:
        pass
    return None


def get_smart_bibtex(titles):
    bibtex_results = []
    failed_titles = []  # ✨ 新增：用于专门记录失败的论文名字

    for title in titles:
        title = title.strip()
        if not title:
            continue

        print(f"\n▶ 正在处理: {title}")
        bibtex = search_crossref(title)

        if not bibtex:
            bibtex = search_arxiv(title)

        if bibtex:
            bibtex_results.append(bibtex)
            print("  ✔️ 成功获取并生成 BibTeX")
        else:
            failed_titles.append(title)  # ✨ 记录失败名单
            print(f"  ❌ 检索失败: 两个数据库均未找到匹配该标题的论文")

        time.sleep(1.5)

    return bibtex_results, failed_titles  # ✨ 同时返回成功数据和失败名单


if __name__ == "__main__":
    input_file = "papers.txt"
    output_file = "smart_references.bib"

    paper_titles = []

    if os.path.exists(input_file):
        print(f"📄 发现 {input_file}，正在读取...")
        with open(input_file, "r", encoding="utf-8") as f:
            paper_titles = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # 默认测试数据：包含必定成功的、预印本的，以及一个故意写错用来测试失败捕获的
        paper_titles = [
            "Single Image Deraining: A Comprehensive Benchmark Analysis",
            "Image Deraining: A Survey",
            "ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding",
            "Vision for Autonomous Vehicles: 2020 State of the Art",
            "Multi-Stage Progressive Image Restoration",
            "Progressive Image Deraining Networks: A Better and Simpler Baseline",
            "Restormer: Efficient Transformer for High-Resolution Image Restoration",
            "Uformer: A General U-Shaped Transformer for Image Restoration",
            "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
            "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
            "SwinIR: Image Restoration Using Swin Transformer",
            "Activating More Pixels in Image Super-Resolution Transformer",
            "Back to basics: Let denoising generative models denoise",
            "A convnet for the 2020s",
        ]
        with open(input_file, "w", encoding="utf-8") as f:
            for t in paper_titles:
                f.write(t + "\n")

    print(f"🚀 共读取到 {len(paper_titles)} 篇论文，开始批量提取...")

    # 解包获取两个列表
    results, failed_list = get_smart_bibtex(paper_titles)

    # 写入文件
    if results:
        with open(output_file, "w", encoding="utf-8") as f:
            for bib in results:
                f.write(bib + "\n\n")

    # ================= ✨ 最终 Summary 输出 ✨ =================
    print("\n" + "=" * 40)
    print("📊 检索任务总结报告")
    print("=" * 40)
    print(f"总计处理论文 : {len(paper_titles)} 篇")
    print(f"✅ 成功获取   : {len(results)} 篇 (已保存至 {output_file})")
    print(f"❌ 检索失败   : {len(failed_list)} 篇")

    if failed_list:
        print("\n⚠️ 以下论文未能自动匹配，请手动前往学术网站搜索：")
        for i, failed_title in enumerate(failed_list, 1):
            print(f"  {i}. {failed_title}")
    print("=" * 40 + "\n")
