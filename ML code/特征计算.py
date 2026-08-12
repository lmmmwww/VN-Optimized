import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import ProtParamData

# Boman指数常用的结合自由能表（单位 kcal/mol）
boman_scale = {
    'A': 0.17, 'C': -0.24, 'D': -0.72, 'E': -0.62, 'F': 1.13,
    'G': 0.01, 'H': -0.40, 'I': 0.99, 'K': -0.99, 'L': 0.98,
    'M': 0.44, 'N': -0.64, 'P': -0.32, 'Q': -0.69, 'R': -1.01,
    'S': -0.04, 'T': 0.14, 'V': 0.86, 'W': 1.85, 'Y': 0.33
}

# 极性氨基酸集合和极性计算函数
polar_aa = {'R', 'N', 'D', 'Q', 'E', 'K', 'H', 'S', 'T', 'Y', 'C'}

def polarity_ratio(seq):
    polar_count = sum(1 for aa in seq if aa in polar_aa)
    return polar_count / len(seq)

# 读取 CSV 文件
input_file = "G:\\VX\\特征筛选\\all_mmgbsa_avb3.csv"  # 输入文件名
df_peptides = pd.read_csv(input_file)

# 存放所有特征和计算结果
features = []
names = []

for index, row in df_peptides.iterrows():
    name = row["Name"]
    seq = row["Sequence"]
    seq = seq.upper()

    # 跳过非法序列
    if not set(seq).issubset(set("ACDEFGHIKLMNPQRSTVWY")):
        print(f"非法序列跳过: {name}")
        continue

    analysis = ProteinAnalysis(seq)

    # 提取理化特征、AAC 和二级结构倾向
    aac_dict = analysis.get_amino_acids_percent()
    aac_features = [aac_dict.get(aa, 0) for aa in 'ACDEFGHIKLMNPQRSTVWY']  # 20种氨基酸比例
    helix, turn, sheet = analysis.secondary_structure_fraction()

    # 基本特征
    mol_weight = analysis.molecular_weight()
    gravy = analysis.gravy()
    charge = analysis.charge_at_pH(7.0)
    net_charge_density = charge / mol_weight if mol_weight > 0 else 0

    # Aliphatic Index
    aa_percent = analysis.get_amino_acids_percent()
    aliphatic_index = 100 * (
        aa_percent.get("A", 0) +
        aa_percent.get("V", 0) * 2.9 +
        (aa_percent.get("I", 0) + aa_percent.get("L", 0)) * 3.9
    )

    # Boman指数
    boman_score = sum([boman_scale.get(aa, 0) for aa in seq]) / len(seq)

    # 添加特征
    features.append([
        name, seq,
        analysis.aromaticity(),  # 芳香性
        analysis.instability_index(),  # 不稳定性
        analysis.isoelectric_point(),  # 等电点
        analysis.gravy(),  # 疏水性
        mol_weight,  # 分子量
        charge,  # 电荷
        polarity_ratio(seq),  # 极性比例
    ] + aac_features + [helix, turn, sheet] + [
        mol_weight, gravy, charge,
        net_charge_density, aliphatic_index, boman_score
    ])

# 创建特征表
feature_columns = [
    "Name", "Sequence", "Aromaticity", "Instability", "pI", "Hydrophobicity",
    "MolWeight", "Charge", "Polarity"
] + [f"AAC_{aa}" for aa in 'ACDEFGHIKLMNPQRSTVWY'] + [
    "Helix", "Turn", "Sheet", "MolWeight", "GRAVY", "Charge",
    "NetChargeDensity", "AliphaticIndex", "BomanIndex"
]

df_features = pd.DataFrame(features, columns=feature_columns)

# 保存特征为一个 CSV 文件
df_features.to_csv("G:\\VX\\特征筛选\\avb3_peptide_features.csv", index=False)
print("✅ 所有特征已保存")
