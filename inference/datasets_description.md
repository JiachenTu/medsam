# 医学图像分割数据集说明

## 概览

| 数据集 | 模态 | 数量 | 分辨率 | 任务 | SAM3 Prompt |
|--------|------|------|--------|------|-------------|
| CHASE_DB1 | 眼底 | 28 | 1280×960 | 血管分割 | Retinal Blood Vessel |
| STARE | 眼底 | 20 | 700×605 | 血管分割 | Retinal Blood Vessel |
| CVC-ClinicDB | 内窥镜 | 612 | 384×288 | 息肉分割 | Polyp |
| ETIS-Larib | 内窥镜 | 196 | 1225×966 | 息肉分割 | Polyp |
| PH2 | 皮肤镜 | 200 | 768×560 | 病变分割 | Skin Lesion |

**数据路径**: `/srv/local/shared/medsam_data/`

---

## 目录结构

### CHASE_DB1
```
CHASE_DB1/
├── Image_XXY.jpg        # 原图 (XX=01-14, Y=L/R)
├── Image_XXY_1stHO.png  # 专家1标注
└── Image_XXY_2ndHO.png  # 专家2标注
```

### STARE
```
STARE/
├── imXXXX.ppm.gz     # 原图 (gzip压缩)
├── imXXXX.ah.ppm.gz  # AH专家标注
└── imXXXX.vk.ppm.gz  # VK专家标注
```
**注意**: 需先执行 `gunzip *.gz` 解压

### CVC-ClinicDB
```
CVC-ClinicDB/PNG/
├── Original/N.png      # 原图 (N=1-612)
└── Ground Truth/N.png  # mask
```

### ETIS-Larib
```
ETIS-Larib/
├── images/N.png  # 原图 (N=1-196)
└── masks/N.png   # mask
```

### PH2
```
PH2/PH2Dataset/PH2 Dataset images/IMDXXX/
├── IMDXXX_Dermoscopic_Image/IMDXXX.bmp  # 原图
└── IMDXXX_lesion/IMDXXX_lesion.bmp      # mask
```

---

## SAM3 Prompt配置

需在 `all_dataset_prompts_simple(1).json` 中添加:
```json
{
  "CHASE_DB1": "Retinal Blood Vessel",
  "STARE": "Retinal Blood Vessel",
  "CVC-ClinicDB": "Polyp",
  "ETIS-Larib": "Polyp",
  "PH2": "Skin Lesion"
}
```
