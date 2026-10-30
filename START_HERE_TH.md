# TI-Artery: โค้ดปรับปรุงสำหรับฝึกและประเมินใหม่

ฝึกอัตโนมัติจน validation F1 ถึงเป้า: ดู AUTO_TRAIN_TH.md และ train_until_target.py

แพ็กเกจนี้มีเฉพาะโค้ด ให้นำไฟล์ไปวางทับในสำเนาโฟลเดอร์ ti-artery-gpu เดิม
ใช้ data/ และ models/ เดิมได้ แต่เริ่มการทดลอง plaque ใหม่จากศูนย์ตามคำสั่งด้านล่าง
ไม่มีผลฝึกใหม่หรือการรับประกัน F1 0.90 รวมอยู่ในแพ็กเกจนี้
README และ QUICKSTART เดิมเป็นบันทึกประวัติ ให้ใช้เอกสารนี้สำหรับขั้นตอนปัจจุบัน

## 1. แบ่ง validation ใหม่

ต้องมี CSV ระบุภาพกับผู้ป่วยจริง (อย่าเดาจากชื่อไฟล์):

```csv
filename,patient_id
train/001.png,P001
train/002.png,P002
test/500.png,P500
```

CSV ต้องครอบคลุมทุกภาพ ชื่อไฟล์ต้องตรงรวมช่องว่างถ้ามี

```bash
python prepare_plaque_validation.py --groups patient_groups.csv
```

หากยังไม่มีข้อมูลผู้ป่วย ทดลองแบบแบ่งรายภาพได้ด้วย:

```bash
python prepare_plaque_validation.py --allow-image-split
```

แบบรายภาพยังยืนยันความเป็นอิสระระหว่างผู้ป่วยไม่ได้ สคริปต์ตรวจภาพซ้ำแบบไบต์
แต่ยังตรวจภาพต่างเฟรมที่คล้ายกันไม่ได้ และไม่ได้ stratify สัดส่วน plaque-positive อัตโนมัติ
ตรวจจำนวนภาพบวก/ลบในแต่ละ split ก่อนฝึก
test เดิมถูกใช้เลือกโมเดลแล้ว ต้องเรียกผลจากชุดนี้ว่า exploratory
ห้ามเรียกเป็น untouched test; การยืนยันสุดท้ายต้องมีผู้ป่วยใหม่ที่กันไว้จริง
อย่าฝึกต่อจาก plaque checkpoint เดิมเพื่อรายงานผล validation ชุดใหม่ เพราะโมเดลเดิมฝึกบนภาพเหล่านี้แล้ว

## 2. ฝึก plaque ใหม่

เริ่มที่ 320 เพื่อให้มี baseline ก่อน แล้วเทียบ 512 โดยใช้ seed/split เดียวกัน

```bash
python train_plaque_unet.py --data data/plaque_v2 --epochs 100 --size 320 --batch 8 --resize-mode letterbox --seed 42 --out models/plaque_v2.pt --last models/plaque_v2_last.pt
```

```bash
python train_plaque_unet.py --data data/plaque_v2 --epochs 100 --size 512 --batch 4 --resize-mode letterbox --seed 42 --out models/plaque_v2_512.pt --last models/plaque_v2_512_last.pt
```

batch/epochs เป็นค่าเริ่มทดลอง ปรับตาม VRAM และกราฟ validation ไม่ใช่เงื่อนไขว่าจะถึง 0.90
เลือก checkpoint จาก validation เท่านั้น; val_dice ระหว่างฝึกคือ global pixel Dice ที่ความละเอียดฝึก
ตัวประเมินรายงานทั้ง mean per-image Dice และ global pixel F1 ที่ความละเอียดภาพต้นฉบับ
จึงไม่ควรนำตัวเลขจากสองความละเอียดมาเทียบตรง ๆ

## 3. เลือก threshold และขนาดก้อนจาก validation เท่านั้น

```bash
python eval_plaque.py --model models/plaque_v2.pt --data data/plaque_v2/val --tune --min-area 0 --out eval_v2_val
```

ทดลอง min-area เพิ่มบน val ได้ แต่ก้อนเล็กอาจเป็น plaque จริง
หลังเลือกค่าแล้ว ล็อกตัวเลขในการประเมินและ pipeline เหมือนกัน ตัวอย่างต่อไปใช้ 0.5/0 เท่านั้น
แทนด้วยค่าที่เลือกจาก validation ของคุณ:

```bash
python eval_plaque.py --model models/plaque_v2.pt --data data/plaque_v2/test --threshold 0.5 --min-area 0 --out eval_v2_test
```

รายงาน mean Dice, plaque-positive Dice, global pixel F1 และ image detection F1 แยกกัน
ภาพ GT ว่างและทำนายว่างให้ per-image Dice=1 ตาม convention ที่ระบุ
precision/recall แบบเฉลี่ยรายภาพเดิมให้ 0 สำหรับตัวหารเป็นศูนย์;
ใช้ image detection precision/recall และ global pixel F1 ประกอบ ห้ามตีความเป็น metric เดียวกัน
ไม่ใช้ eval_ensemble.py เดิมสำหรับการทดลองใหม่นี้ เพราะยังเป็นวิธีประเมินรุ่นเก่า

## 4. ฝึก wall

```bash
python train_wall_unet.py --data data/cubs --epochs 100 --size 512 --batch 4 --resize-mode letterbox --out models/wall_v2.pt --last models/wall_v2_last.pt
python eval_wall.py --model models/wall_v2.pt --out eval_wall_v2
```

CUBS ยังใช้ fold ที่เตรียมไว้เดิม ต้องตรวจต้นทางว่าผู้ป่วยไม่ซ้ำข้าม split
การวัด mask ใช้จำนวนพิกเซลในคอลัมน์ลบหนึ่งเพื่อให้ตรงกับการสร้าง mask แบบรวมปลายทั้งสองด้าน
ยังเป็น vertical thickness ไม่ใช่การยืนยัน subpixel/perpendicular IMT ตามมาตรฐานคลินิก
รายงาน MAE พร้อมจำนวนเคสวัดไม่สำเร็จ; MAE คิดเฉพาะเคสวัดสำเร็จ

## 5. ใช้โมเดลใหม่กับ pipeline

```python
from tiartery.pipeline import run_pipeline
result = run_pipeline(
    'sample.png', model_path='models/plaque_v2.pt',
    wall_model_path='models/wall_v2.pt',
    pixel_spacing_mm=0.05,  # ต้องแทนด้วยสเกลจริงของภาพต้นฉบับ ห้ามใช้ค่าตัวอย่างกับผู้ป่วย
    plaque_threshold=0.5, plaque_min_area=0)
```

ไม่มี wall model หรือสเกลจริงจะไม่รายงาน IMT เป็น mm และไม่มี plaque model จะคืน plaque_present=None
เดโม app.py อ่านค่า environment ได้:
TIARTERY_PLAQUE_MODEL, TIARTERY_WALL_MODEL, TIARTERY_THRESHOLD, TIARTERY_MIN_AREA
ค่า threshold/min-area ต้องตรงกับค่าที่เลือกบน validation
Risk เป็น heuristic เดิม ยังไม่ผ่าน validation; เมื่อข้อมูลภาพไม่ครบหรือ QC ไม่ผ่านจะไม่ให้ tier
วิดีโอและ ROI detector ไม่ได้เพิ่มในรอบนี้

## การตรวจสอบ

ใช้ python -m unittest discover -s tests -v
ต้องติดตั้ง dependencies ตาม requirements.txt และ PyTorch ที่เหมาะกับเครื่องก่อน
ไม่มีการรันฝึกเต็มหรือยืนยันคะแนนใหม่ในแพ็กเกจนี้
