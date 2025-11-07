# from pathlib import Path
#
# img_name = r'/localnvme/data/billboard/all_data/mseg_c6/data7961_mseg_c6_1030/images/頭頸肩200.png'
# if Path(img_name).suffix.lower() in ['.jpg', 'jpeg', 'png']:
#     print('true')
# else:
#     print(Path(img_name).suffix.lower())

import os


print(len(os.listdir(r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/val694/risk_abandonment_pred_no_label_background')))
print(len(os.listdir(r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/val694/risk_broken_pred_no_label_background')))
print(len(os.listdir(r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/val694/risk_corrosion_pred_no_label_background')))
print(len(os.listdir(r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/val694/risk_deformation_pred_no_label_background')))