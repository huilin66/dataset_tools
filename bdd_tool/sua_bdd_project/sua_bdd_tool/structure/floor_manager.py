import os
import json

class FloorManager:
    def __init__(self, floor_params=None, cache_file=None):
        """
        :param floor_params: 用于构建的参数字典 (构建模式必填)
        :param cache_file: 用于保存或加载的 JSON 文件路径 (加载模式必填)
        """
        self.floor_map = {} 
        self.is_valid = False
        self.base_height = 0.0 # 用于绘图
        self.final_calc_height = 0.0 # 用于绘图

        # === 逻辑分支 ===
        # 情况 1: 提供了缓存文件，且文件存在 -> 直接加载 (Load Mode)
        if cache_file and os.path.exists(cache_file):
            print(f"📂 发现缓存文件 '{cache_file}'，正在加载...")
            self._load_from_file(cache_file)
        
        # 情况 2: 提供了参数 -> 重新构建 (Build Mode)
        elif floor_params:
            print("⚙️ 未找到缓存或强制构建，正在计算楼层...")
            self.params = floor_params
            self._parse_and_build()
            # 如果构建成功且指定了缓存路径，自动保存
            if self.is_valid and cache_file:
                self.write_floor_map(cache_file)
        
        # 情况 3: 既没文件也没参数 -> 报错
        else:
            raise ValueError("❌ 必须提供 floor_params 进行构建，或提供有效的 cache_file 进行加载。")

        # 打印图表 (可选，只有在数据完整时打印)
        if self.is_valid:
            self.print_floor_chart()

    def _parse_and_build(self):
        # 补丁：为了让 print_chart 不报错，这里模拟原逻辑的 scale 计算来获取 base_height
        p = self.params
        scale = 0.001 if p['normal floor height'] > 100 else 1.0
        self.base_height = p['base_height'] * scale
        
        
        base_h = p['base_height'] * scale
        final_h = p['final height'] * scale
        norm_h = p['normal floor height'] * scale
        
        # 转换列表和字典中的高度
        podium_hs = [h * scale for h in p['podium heights']]
        top_hs = [h * scale for h in p['top heights']]
        special_hs = {str(k): v * scale for k, v in p['special heights'].items()}
        
        # 2. 构建楼层序列 (Name, Height)
        floor_sequence = []
        
        # A. Podium (裙楼/底层)
        if len(p['podium names']) != len(podium_hs):
            print(f"❌ 楼层参数错误: Podium 名字数量 ({len(p['podium names'])}) 与 高度数量 ({len(podium_hs)}) 不一致")
            return
            
        for name, h in zip(p['podium names'], podium_hs):
            floor_sequence.append((str(name), h))
            
        # B. Normal (标准层 + 特殊层)
        # range 是左闭右闭，所以 end + 1
        start_idx, end_idx = p['normal height number list']
        expected_norm_count = p['normal height numbers']
        
        # 校验数量
        real_norm_count = end_idx - start_idx + 1
        if real_norm_count != expected_norm_count:
             print(f"⚠️ 警告: Normal floor 数量定义不一致 (Number: {expected_norm_count} vs List range: {real_norm_count})，以 List 为准")

        for i in range(start_idx, end_idx + 1):
            name = str(i)
            # 检查是否是特殊层
            h = special_hs.get(name, norm_h)
            floor_sequence.append((name, h))
            
        # C. Top (顶层)
        if len(p['top names']) != len(top_hs):
            print(f"❌ 楼层参数错误: Top 名字数量 ({len(p['top names'])}) 与 高度数量 ({len(top_hs)}) 不一致")
            return

        for name, h in zip(p['top names'], top_hs):
            floor_sequence.append((str(name), h))

        # 3. 生成高度分布字典 & 校验总高度
        current_z = base_h
        
        for name, h in floor_sequence:
            # 格式化 Key: "楼层编号/F"
            key = f"{name}/F"
            self.floor_map[key] = (current_z, current_z + h)
            current_z += h
            
        # 4. 校验高度闭环
        # 理论总高度 = final - base
        # 累加总高度 = current_z - base_h
        self.final_calc_height = current_z
        diff = abs(current_z - final_h)
        
        print(f"🏢 楼层构建完成: 起始 {base_h:.2f}m -> 计算结束 {current_z:.2f}m (定义结束 {final_h:.2f}m)")
        
        if diff > 0.1: # 允许 10cm 误差
            print(f"⚠️ 警告: 建筑高度校验失败! 偏差 {diff:.4f}m")
            print("   请检查: base_height, final height 或 各层高度之和是否匹配")
        else:
            print("✅ 建筑高度校验通过")
            self.is_valid = True

    def _load_from_file(self, path):
        """从 JSON 加载数据，恢复状态"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # JSON 读取后的格式通常是:
            # { "meta": {"base": 10.0, "top": 100.0}, "map": {"1/F": [10, 13], ...} }
            # 为了简单，如果你只存了 map，就只读 map。
            # 但为了 print_chart 能用，建议保存时多存一点元数据。
            
            if "floor_map" in data:
                self.floor_map = data["floor_map"]
                self.base_height = data.get("base_height", 0.0)
                self.final_calc_height = data.get("final_calc_height", 0.0)
            else:
                # 兼容旧版本只存了 map 的情况
                self.floor_map = data
            
            self.is_valid = True
            print("✅ 楼层数据加载成功")
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            self.is_valid = False

    def write_floor_map(self, output_path):
        """将楼层映射及元数据写入 JSON 文件"""
        # 建议把元数据一起存了，这样下次加载后还能画图
        save_data = {
            "base_height": self.base_height,
            "final_calc_height": self.final_calc_height,
            "floor_map": self.floor_map
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
        print(f"💾 楼层数据已保存至: {output_path}")

    def get_floor(self, z_value):
        # 原逻辑保持不变
        # 注意：JSON 加载回来的 value 是 List [start, end]
        # 但 Python 的解包赋值 (start, end) = [10, 13] 对 List 和 Tuple 都适用
        # 所以这里的代码完全不需要改动
        epsilon = 0.01 
        for name, (start, end) in self.floor_map.items():
            if start - epsilon <= z_value < end + epsilon:
                return name

        # 如果找不到
        sorted_floors = sorted(self.floor_map.values(), key=lambda x: x[0])
        if not sorted_floors: return "Unknown"
        
        min_h = sorted_floors[0][0]
        max_h = sorted_floors[-1][1]
        
        if z_value < min_h:
            return "Below Base"
        elif z_value >= max_h:
            return "Above Top"
        
        return "Unknown"

    def print_floor_chart(self):
        # 稍微修改，不再依赖 self.params，而是依赖 self.base_height
        if not self.floor_map: return

        print("\n🏢 Building Elevation Chart (Top-Down)")
        print("=" * 40)
        print(f"{'[TOP]':<10} ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅  {self.final_calc_height:7.2f}m")

        sorted_floors = sorted(self.floor_map.items(), key=lambda item: item[1][0], reverse=True)
        for name, (start_z, end_z) in sorted_floors:
            print(f"{name:<10} ______  {start_z:7.2f}m")
            
        # 这里改用 self.base_height
        print(f"{'[BASE]':<10} ______  {self.base_height:7.2f}m")
        print("=" * 40 + "\n")
    
    @property
    def floors_heights(self):
        """
        [新增] 适配 FacadeVisualizer 的接口
        返回一个字典 { '楼层名': 高度值(通常是地板高度) }
        """
        simple_map = {}
        # 遍历 floor_map, 提取每层的起始高度作为画线的依据
        for name, (start_z, end_z) in self.floor_map.items():
            # 这里的 name 可能是 "1/F", "2/F"，在图上显示不需要改
            simple_map[name] = start_z
        
        # 也可以把顶层的封顶线加进去，但这可能导致图表最上面太挤
        # 如果需要，可以解开下面这行
        # simple_map['[TOP]'] = self.final_calc_height
        
        return simple_map
