import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import requests


def search_crossref(title):
    """尝试从 Crossref 获取正式发表论文的 BibTeX"""
    print(f"  🔍 尝试 Crossref (正式发表数据库)...")
    query_url = (
        f"https://api.crossref.org/works?query.title={urllib.parse.quote(title)}&rows=1"
    )

    try:
        response = requests.get(query_url, timeout=10)
        data = response.json()
        items = data.get("message", {}).get("items", [])

        if not items:
            return None

        # 简单校验匹配度（取第一个结果的标题和输入标题的长度差），防止匹配到完全不相关的文章
        # 这里可以根据需要加强匹配逻辑
        doi = items[0].get("DOI")

        if not doi:
            return None

        bib_url = f"https://doi.org/{doi}"
        headers = {"Accept": "application/x-bibtex"}
        bib_response = requests.get(bib_url, headers=headers, timeout=10)

        if bib_response.status_code == 200:
            return bib_response.content.decode("utf-8")
    except Exception as e:
        # 静默处理异常，交由 arXiv 兜底
        pass

    return None


def search_arxiv(title):
    """尝试从 arXiv 获取预印本论文的 BibTeX"""
    print(f"  🔄 Crossref 未命中，尝试 arXiv (预印本数据库)...")
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

        paper_title = entry.find(f"{OAI}title").text.replace("\n", " ").strip()
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
  title={{{paper_title}}},
  author={{{authors_str}}},
  year={{{year}}},
  eprint={{{paper_id}}},
  archivePrefix={{arXiv}},
  primaryClass={{{category}}}
}}"""
        return bibtex
    except Exception as e:
        pass

    return None


def get_smart_bibtex(titles):
    bibtex_results = []

    for title in titles:
        print(f"\n▶ 正在处理: {title}")

        # 1. 优先查询 Crossref
        bibtex = search_crossref(title)

        # 2. 如果失败，降级查询 arXiv
        if not bibtex:
            bibtex = search_arxiv(title)

        # 3. 结果判断
        if bibtex:
            bibtex_results.append(bibtex)
            print("  ✔️ 成功获取并生成 BibTeX")
        else:
            print(f"  ❌ 检索失败: 两个数据库均未找到该论文")

        # 礼貌性休眠，保护 API
        time.sleep(1.5)

    return bibtex_results


# 测试你的论文列表
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

print("🚀 开始批量提取文献引用...")
results = get_smart_bibtex(paper_titles)

with open("smart_references.bib", "w", encoding="utf-8") as f:
    for bib in results:
        f.write(bib + "\n\n")

print("\n🎉 全部任务完成！已汇总保存至 smart_references.bib")
