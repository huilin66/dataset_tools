import json


class Cap123RAG:
    def __init__(self, json_path):
        # 加载本地的条例知识库
        with open(json_path, "r", encoding="utf-8") as f:
            self.knowledge_base = json.load(f)

    def retrieve_ordinance(self, defect_type, defect_level, is_old_building=True):
        """
        基于规则匹配对应的法例 Section。
        加入了 Section 25, 30B, 30C 的逻辑。
        """
        matched_sections = []

        # 1. 强制验楼/验窗计划 (最优先的常规维护法例)
        if "window" in defect_type or defect_type in [
            "frame_deformed",
            "frame_corroded",
        ]:
            matched_sections.append(
                self._get_section("Cap. 123 Section 30C")
            )  # 窗户缺陷 -> MWIS

        elif is_old_building and defect_type in [
            "spalling",
            "crack",
            "finishes_peeling",
        ]:
            matched_sections.append(
                self._get_section("Cap. 123 Section 30B")
            )  # 外墙/结构缺陷 -> MBIS

        # 2. 危险/欠妥建筑 (根据严重程度)
        if defect_level == "serious" or defect_type == "spalling":
            matched_sections.append(
                self._get_section("Cap. 123 Section 26")
            )  # 危险建筑物

        elif defect_level in ["moderate", "minor"]:
            matched_sections.append(
                self._get_section("Cap. 123 Section 26A")
            )  # 欠妥建筑物

        # 3. 未经授权的更改 / 违建招牌
        if defect_type in ["added_billboard", "unauthorized"]:
            matched_sections.append(
                self._get_section("Cap. 123 Section 25")
            )  # 用途更改/违建

        # 4. 排水/渗水
        if defect_type in ["water_seepage", "moisture", "mold"]:
            matched_sections.append(self._get_section("Cap. 123 Section 28"))  # 排水渠

        # 去重并组装文本
        unique_sections = {sec["section"]: sec for sec in matched_sections if sec}

        context = ""
        for sec_id, sec_data in unique_sections.items():
            context += f"【{sec_data['section']} - {sec_data['title']}】\n{sec_data['content']}\n\n"

        return context.strip()

    def _get_section(self, section_id):
        for item in self.knowledge_base:
            if item["section"] == section_id:
                return item
        return None


def generate_llava_prompt_with_cap123(defect_type, defect_level):
    # 1. 初始化 RAG 检索器
    rag = Cap123RAG("cap123_knowledge.json")

    # 2. 检索相关法例
    ordinance_context = rag.retrieve_ordinance(defect_type, defect_level)

    # 3. 构造增强版的 Prompt
    prompt = (
        f"<image>\n"
        f"You are a professional building inspector in Hong Kong. "
        f"A bounding box indicates a '{defect_level}' '{defect_type}' on the building facade.\n\n"
        f"### Reference Ordinances (Cap. 123) ###\n"
        f"{ordinance_context}\n"
        f"########################################\n\n"
        f"Please output a detailed technical description of this defect in a Markdown table. "
        f"You MUST include a column named 'Reference Ordinance' where you explicitly cite the provided Cap. 123 Section and briefly explain how it applies to this specific defect. "
        f"Also include 'Defect Type', 'Severity Level', and 'Recommended Action'."
    )
    return prompt


# 模拟测试：假设 YOLO 在航拍图中检测到了严重的混凝土剥落
prompt = generate_llava_prompt_with_cap123(
    defect_type="spalling", defect_level="serious"
)
print(prompt)
