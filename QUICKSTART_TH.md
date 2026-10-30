> Historical document. Use START_HERE_TH.md for the updated workflow; old scores are exploratory.

# TI-Artery — คู่มือรันบนเครื่อง GPU (ภาษาไทย)

แพ็กเกจนี้คือโค้ดทั้งหมดของระบบ TI-Artery พร้อมข้อมูลและโมเดลที่ดีที่สุด
แตกไฟล์แล้วรันได้เลย ไม่ต้องโหลดอะไรเพิ่ม

## ในแพ็กเกจมีอะไรบ้าง

| ไฟล์/โฟลเดอร์ | คืออะไร |
|---|---|
| `tiartery/` | โค้ดหลัก (preprocessing, segmentation, IMT, plaque, risk, pipeline) |
| `train_plaque_unet.py` | สคริปต์ฝึกโมเดล |
| `eval_plaque.py` | สคริปต์ประเมินผล (ได้ตัวเลข F1/Dice + ภาพ overlay) |
| `eval_ensemble.py` | สคริปต์ประเมินแบบ ensemble 3 โมเดล |
| `app.py` | เดโม Gradio (อัปโหลดภาพ ultrasound แล้วดูผล) |
| `data/ar-plaquesegm1/` | ข้อมูลฝึก (541 คู่ภาพ+mask, พร้อมใช้) |
| `models/plaque_unet.pt` | โมเดลที่ดีที่สุด (F1 0.65) — ใช้ต่อยอดหรือเดโมได้เลย |
| `requirements.txt` | รายการ library ที่ต้องติดตั้ง |

## ขั้นตอนติดตั้ง (Windows)

1. ติดตั้ง **Python 3.10+** จาก python.org (ติ๊ก "Add python.exe to PATH")
2. ติดตั้ง **PyTorch แบบ GPU** (เลือกให้ตรง CUDA ของเครื่อง):
   - เปิด https://pytorch.org/get-started/locally/ เลือก Stable, Windows, Pip, Python, CUDA (เช่น 12.1)
   - จะได้คำสั่งประมาณนี้ รันใน PowerShell:
     ```
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     ```
3. เข้าโฟลเดอร์ที่แตกไฟล์ แล้วติดตั้งที่เหลือ:
   ```
   pip install -r requirements.txt
   ```
4. เช็คว่า GPU พร้อมใช้:
   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   ต้องขึ้น `True`

## ฝึกโมเดลผนังหลอดเลือด (wall segmentation → IMT)

ข้อมูล CUBS พร้อมใช้ใน `data/cubs/` (295 train / 103 val / 102 test)
คำสั่งฝึกบน GPU:
```
python train_wall_unet.py --data data/cubs --epochs 50 --batch 16 --lr 3e-4 --out models/my_wall.pt
```
ประเมินผล (ได้ Dice + คลาดเคลื่อน IMT เป็นไมครอน):
```
python eval_wall.py --model models/my_wall.pt
```
ค่า IMT error ควรต่ำกว่า ~160 um (คือความต่างระหว่างผู้เชี่ยวชาญด้วยกันเอง)

## ฝึกโมเดล plaque ต่อบน GPU (เร็วกว่า CPU หลายสิบเท่า)

**ฝึกต่อจากโมเดลที่ดีที่สุด** (แนะนำ — เริ่มจาก F1 0.65):
```
python train_plaque_unet.py --arch unet --init-from models/plaque_unet.pt --epochs 50 --size 320 --batch 16 --lr 1e-4 --out models/my_model.pt
```

**ลอง Attention U-Net** (มี attention gate ช่วยตัด false positive):
```
python train_plaque_unet.py --arch attnunet --init-from models/plaque_unet.pt --epochs 30 --size 320 --batch 16 --lr 1e-4 --out models/my_attn.pt
```

**ฝึก U-Net ใหญ่ตั้งแต่ต้น** (17M params, ช้ากว่าแต่ความจุเยอะ):
```
python train_plaque_unet.py --arch unet --base 48 --epochs 50 --size 320 --batch 8 --lr 3e-4 --out models/my_unet48.pt
```

> บน GPU ปรับ `--batch` ได้ใหญ่ขึ้น (16–32) จะยิ่งเร็ว
> ถ้าไฟดับ/หยุดกลางคัน เติม `--resume` ฝึกต่อจากจุดเดิมได้เลย

## ประเมินผล

```
python eval_plaque.py --model models/my_model.pt --min-area 1500
```
จะได้ค่า F1/Dice + ภาพ overlay (สีเขียว=ทายถูก, แดง=false positive, น้ำเงิน=false negative) ในโฟลเดอร์ `eval_out/`

`--min-area 1500` คือ post-processing ตัดจุดรบกวนเล็กๆ (เทคนิคที่ดันคะแนน 0.47 → 0.73)

## รันเดโม

```
python app.py
```
เปิดเบราว์เซอร์ไปที่ http://localhost:7860 อัปโหลดภาพ ultrasound แล้วดูผลวิเคราะห์ได้เลย

## ผลลัพธ์ที่ทำไว้แล้ว (บน CPU, test set 48 ภาพ)

| วิธี | F1 |
|---|---|
| U-Net เดี่ยว + post-processing | 0.73 |
| Ensemble 3 โมเดล + post-processing | **0.75** |

เป้าหมาย 0.90 ยังไม่ถึง — บน GPU ลองฝึกนานขึ้น / โมเดลใหญ่ขึ้น / ข้อมูลเพิ่ม อาจขยับเข้าใกล้ได้

## โครงสร้าง pipeline (ตรงตาม proposal 9 ขั้น)

ดู `tiartery/pipeline.py` — `run_pipeline()` เรียกครบทุกขั้น:
preprocessing → QC → denoise → wall seg → IMT → plaque detect → plaque seg → quantify → calibrate → risk
