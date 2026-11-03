# TI-Artery: สรุปโครงการจากหลักฐานใน snapshot ปัจจุบัน

ตรวจจากโค้ด checkpoint ข้อมูล และผลประเมินที่มีอยู่ในโครงการ ณ 30 ตุลาคม 2026 เอกสารนี้อธิบายสิ่งที่ทำงานอยู่จริง ไม่ใช่แผนงาน ค่า Dice/IoU บอกความตรงกันของ mask กับ label ไม่ได้บอกความแม่นยำในการทำนาย stroke

## 1. เป้าหมายและขอบเขต

แอป Gradio รับภาพอัลตราซาวนด์ carotid แบบ B-mode ภาพนิ่ง แล้วแสดงภาพซ้อน mask plaque และผนังหลอดเลือด พร้อมค่าที่คำนวณได้ เช่น การมี plaque ขนาด plaque และ IMT รวมทั้งคะแนนคัดกรองแบบกฎ (rule-based) หากข้อมูลที่จำเป็นครบ (`app.py:42-70,125-166`; `tiartery/pipeline.py:57-67,199-214`). ช่องอายุ ความดันตัวบน LDL การสูบบุหรี่ เบาหวาน และ pixel spacing เป็นข้อมูลที่ผู้ใช้กรอก ไม่ได้อ่านจากภาพ (`app.py:133-147`).

ปัจจุบันหน้าแอปรับภาพนิ่งผ่าน `gr.Image` ไม่ได้รับวิดีโอโดยตรง (`app.py:125-129`). ฟังก์ชัน pipeline รองรับการอ่านวิดีโอบางนามสกุลและเลือกเฟรมได้ แต่เป็นความสามารถของโค้ด ไม่ใช่ช่องทางที่ UI ปัจจุบันเปิดให้ใช้ (`tiartery/pipeline.py:72-76`; `tiartery/preprocessing.py`). ระบบไม่ได้ประเมิน Doppler/hemodynamic stenosis, intraplaque hemorrhage, plaque 3 มิติ หรือหลอดเลือดนอกขอบเขตที่ระบุในหน้าแอป (`app.py:168-173`).

## 2. Dataset และความน่าเชื่อถือของการแบ่งชุด

| ชุดข้อมูล | แหล่ง/จำนวนและ label | การใช้และ split ที่พบ | ข้อจำกัด |
|---|---|---|---|
| Ar-PlaqSegm1 | [Mendeley Data v2](https://data.mendeley.com/datasets/8srkpz52dy/2), DOI `10.17632/8srkpz52dy.2`; 541 ภาพ B-mode 800×800 คู่กับ binary mask plaque ที่ผ่านการตรวจโดยผู้เชี่ยวชาญ; คำอธิบายชุดข้อมูลระบุ 340 ภาพมี plaque และ 201 ไม่มี plaque | ข้อมูลต้นทางแบ่ง train 493 / test 48. โปรเจกต์เตรียม `plaque_v3`: train 391 (245 positive/146 negative), validation 98 (67/31), test 52 (28/24); checkpoint plaque ฝึก/ปรับต่อจากชุดนี้ | manifest ระบุ `patient_grouped=false` และ patient ID หายทั้ง 541 ภาพ แม้ hash ภาพไม่ซ้ำข้าม split ก็พิสูจน์ความเป็นอิสระระดับผู้ป่วยไม่ได้; test เดิมถูกใช้เลือกโมเดล จึงเป็น exploratory ไม่ใช่ untouched test |
| CUBS technical | [Mendeley Data v1](https://data.mendeley.com/datasets/m7ndn58sv6/1), DOI `10.17632/m7ndn58sv6.1`; ชุด technical มี 500 ภาพ (โปรเจกต์เลือก `tech_*`); มี calibration factor, manual LI/MA profiles และ IMT อ้างอิงในข้อมูลที่เตรียม | ใช้ฝึก/ตรวจ segmentation ผนังและเทียบ IMT; split ใน manifest: train 295 / validation 103 / test 102. mask ผนังสร้างจากบริเวณระหว่าง LI/MA profile (`prepare_cubs.py`) | พบภาพซ้ำแบบ pixel-exact ระหว่าง train/test สองคู่: `train/169.png` กับ `test/049.png`, และ `train/123.png` กับ `test/048.png`; label mask ของภาพซ้ำไม่เหมือนกัน. ประเมินหลักจึงตัด test สองภาพนี้ออก เหลือ 100. source ID ของแต่ละ split แยกกัน แต่ไม่มี patient ID จึงยืนยัน patient independence ไม่ได้ |

ข้อมูลและ manifest อยู่ใน data root ที่อยู่ข้าง checkout (`../data` จากโฟลเดอร์โครงการ): `ar-plaquesegm1/manifest.json`, `plaque_v3/split_manifest.json`, `cubs/manifest.json`. วิธีสร้างชุดอยู่ใน `../data/prepare_plaque_grouped.py` และ `../data/prepare_cubs.py` หากข้อมูลเหล่านี้หายจากการแจกจ่าย จะไม่สามารถตรวจ split/label เดิมจาก checkpoint อย่างเดียวได้. ไม่พบหลักฐานว่ามี label ผล stroke หรือ label การตัดสินใจทางคลินิก

## 3. เส้นทางประมวลผลจริง

1. **รับภาพ:** UI ได้ภาพเป็น NumPy RGB แล้วแปลงเป็น grayscale และบันทึกไฟล์ชั่วคราว (`app.py:42-52`). ถ้าเรียก pipeline ด้วยวิดีโอโดยตรง จะสุ่มเลือกเฟรมอย่างสม่ำเสมอได้สูงสุด 60 เฟรม แล้วเลือกเฟรมที่ Laplacian variance สูงสุด; UI ไม่ได้เปิดรับวิดีโอ (`tiartery/preprocessing.py`; `pipeline.py:72-80`).
2. **ตรวจคุณภาพ:** คำนวณ blur/ความสว่าง/contrast; เกณฑ์ที่โค้ดใช้คือ blur ต่ำกว่า 30, mean intensity ต่ำกว่า .05 หรือสูงกว่า .95, contrast ต่ำกว่า .03 ถือเป็นปัญหา. pipeline ยังทำ segmentation ได้ แต่ไม่แสดง risk assessment ที่สมบูรณ์เมื่อภาพไม่ผ่าน QC (`preprocessing.py`; `pipeline.py:78-80,23-42`).
3. **ลด speckle เพื่อ preview:** UI เรียก `denoise="both"` คือ Lee ตามด้วย SRAD; ผลเก็บเป็น `denoised_preview`. โมเดล segmentation รับ grayscale ต้นฉบับ `gray` ไม่ใช่ภาพ denoise เพื่อให้สอดคล้องกับข้อมูลฝึก (`app.py:54-60`; `pipeline.py:82-90`).
4. **Plaque:** โหลด U-Net plaque, resize ตาม checkpoint, ได้ probability map, threshold ที่ .5 และสร้าง mask. จาก mask ตัดสินว่าพบ plaque หรือไม่ และคำนวณจำนวน pixel/ขนาด (`pipeline.py:91-115`; `segmentation.py`; `plaque.py`).
5. **ผนังและ IMT:** โหลด wall U-Net แยกต่างหาก ได้ mask/probability แล้วหา LI/MA boundary ตามแนวคอลัมน์และคำนวณระยะ; ถ้าไม่มีโมเดลหรือวัดไม่ได้ ใช้ classical edge demo fallback (`pipeline.py:117-170`; `imt.py`).
6. **Risk และภาพผลลัพธ์:** ส่งค่าที่ผ่าน calibration และข้อมูลผู้ป่วยไปกฎคะแนน; ถ้าขาดข้อมูลจำเป็น ให้ tier เป็น unavailable และระบุสาเหตุ. วาด plaque สีเขียวและ wall สีส้มลงภาพ (`pipeline.py:172-214`; `risk.py`).

## 4. เทคนิคที่ใช้จริงและช่วงที่ใช้

| เทคนิค | ใช้จริงอย่างไร |
|---|---|
| Grayscale, normalize | ภาพเข้า segmentation เป็น grayscale; dataset/model ใช้ช่วง intensity ที่ normalize แล้ว. ใช้ทั้งเตรียมข้อมูลและ inference |
| เลือกเฟรม | ถ้าเรียก video loader จะเลือกเฟรมคมที่สุดจากชุดเฟรมที่สุ่มสม่ำเสมอ; ไม่ได้ใช้กับ UI ปัจจุบันที่รับภาพนิ่ง |
| Lee/SRAD | ในแอปสร้างภาพ denoised preview; ไม่ได้ป้อนเข้า segmentation. ไม่พบว่าใช้เป็น augmentation/training input |
| Resize/letterbox | โมเดลรับขนาด 320×320. Letterbox รักษาสัดส่วนและเติมขอบ; plaque ระบุ letterbox. Wall checkpoint ไม่มี metadata resize mode; เมื่อเทียบ validation พบ stretch ทำซ้ำ Dice ใน checkpoint ได้ ส่วน letterbox ได้ Dice ต่ำกว่า จึงคง stretch สำหรับ checkpoint นี้และใช้ mode เดียวกันทั้ง resize/restore |
| Augmentation | ระหว่างฝึก plaque: flip, หมุน/scale/translate, gamma/brightness/contrast, CLAHE และ noise; wall: flip, หมุนและปรับ intensity. ไม่ได้ใช้ใน inference (`train_plaque_unet.py`, `train_wall_unet.py`) |
| Threshold | probability ≥ .5 เป็น foreground; plaque min component area ในแอป default 0 จึงไม่มี component cleanup (`app.py:30-31`; `evaluation/accuracy_20261030/REPORT.md:3,7`) |
| Postprocessing | IMT เก็บองค์ประกอบ wall ที่เหมาะสมและใช้ขอบ sub-pixel จาก probability; plaque presence ตรวจจากพื้นที่ mask อย่างน้อย 50 px. การวัด geometry ทำหลัง segmentation ไม่ใช่โมเดล classification แยก |

## 5. โมเดล checkpoint และการฝึก

ทั้งสองโมเดลเป็น custom U-Net แยกหน้าที่ ไม่ใช่ classifier stroke. encoder/decoder ใช้ base channels 32,64,128,256 และ bottleneck 512; รับภาพ grayscale ขนาด 320 และคืน probability mask หนึ่งช่อง. โค้ดโหลด state dict ของโปรเจกต์เอง ไม่พบหลักฐาน pretrained น้ำหนักจากภายนอก (`tiartery/segmentation.py`).

| งาน | checkpoint ที่แอปเลือกจริง | สิ่งที่ยืนยันได้ |
|---|---|---|
| Plaque | `models/plaque_finetune.pt` (ลำดับ fallback ใน `app.py:26-35`) | metadata: U-Net base32, input 320, letterbox, threshold .5, BCE+Dice loss, seed 42, validation global Dice .621561. Log `runs/plaque_finetune_from_0015/console.log` แสดงฝึกจาก checkpoint ของโปรเจกต์ `trial_0015`, best epoch 25, best val Dice .621561; early stop ที่ epoch 43. ไม่มีหลักฐาน external pretrained. รายละเอียด batch size และทุก CLI override ของ fine-tune รอบนี้บันทึกไม่ครบ จึงระบุค่าจริงทั้งหมดไม่ได้ |
| Wall | `models/wall_unet.pt` (fallback `my_wall.pt`) | metadata: U-Net base32, size 320, val Dice .762632; ไม่มี `resize_mode`, seed, epoch, loss หรือ optimizer ใน checkpoint. log `../restore_original/ti-artery-gpu/logs/wall_auto.log` ที่สอดคล้องกับ checkpoint val .7626 แสดง train 295/val103, CUDA, 50 epochs, เริ่ม LR 3e-4, best บันทึกเป็น `my_wall.pt`; การตรงกันของเลข val ทำให้เชื่อมโยงได้ แต่ checkpoint ปัจจุบันคือ `wall_unet.pt` จึงไม่อาจยืนยันตัวตนไฟล์จากค่า Dice อย่างเดียว |

สคริปต์ฝึกใน checkout ใช้ AdamW, cosine learning-rate schedule และ BCE+Dice เป็นค่าปริยาย; plaque augmentation และ wall augmentation ตามตารางข้างต้น. ค่า default ในสคริปต์ไม่ใช่หลักฐานว่า run ที่สร้างทุก checkpoint ใช้ค่าทุกตัวนั้น. Run config `runs/plaque_20261028_143631_262528/config.json` ยืนยันค่าของ automatic search run นั้น (เช่น lr 0.0003, batch 4, 100 epochs, seed 42) แต่ไม่ควรนำไปเหมารวมเป็นค่าของ fine-tune checkpoint ทุกครั้ง

## 6. การวัด IMT และ plaque

**IMT:** `imt_from_wall_mask` หา LI และ MA จากขอบบน/ล่างของบริเวณ wall ตามคอลัมน์ ใช้ probability ช่วยประมาณขอบ sub-pixel แล้ววัดระยะตั้งฉากกับผนัง ได้ mean/max หน่วย px (`imt.py`; `pipeline.py:124-129`). เมื่อมี scanner spacing ที่เป็น finite และมากกว่า 0 จึงแปลงเป็น mm. เมื่อ blank/ไม่มี calibration คง `mean_imt_px`/`max_imt_px`; ค่า 0.10 mm/px หากคำนวณไว้จะอยู่ใน estimate fields แยก แต่รายงานหลักไม่แสดงค่า mm หรือ demo estimate. Risk ไม่รับค่าดังกล่าว. classical CV fallback เป็น demo เช่นกัน (`pipeline.py:18-20,131-170`). ค่า 0 หรือติดลบไม่ถือเป็น calibration (`pipeline.py:68-70`; callback แปลง 0 เป็น `None`).

**Plaque:** พื้นที่ mask คือจำนวน pixel foreground; ความยาว/ความหนามาจาก contour และกรอบ min-area rectangle (`plaque.py`). ถ้ามี spacing `s` mm/px: length/thickness × `s`, area × `s²` เป็น mm². ถ้าไม่มี spacing รายงานหลักแสดงความยาว/ความหนาเป็น px และพื้นที่เป็น px²; ค่า estimate ที่ pipeline อาจคำนวณไว้จะไม่แสดง. ไม่มีการจัดชั้นขนาดหรือคำแนะนำจากมิติที่ไม่ calibrated. Risk ใช้เฉพาะ `area_mm2` ที่ calibrated ไม่ใช้ `area_mm2_est`.

## 7. คะแนนคัดกรอง

`assess_risk()` เป็นกฎบวกคะแนน ไม่ใช่โมเดลเรียนรู้และไม่ใช่เปอร์เซ็นต์โอกาส stroke (`risk.py:1-9,12-72`). ปัจจัยและคะแนน: พบ plaque +2; IMT ≥0.9 mm +1; total plaque area ≥60 mm² +1; อายุ ≥60 +1; SBP ≥140 mmHg +1; LDL ≥130 mg/dL +1; smoking +1; diabetes +1. คะแนนรวมสูงสุด 8; 0–1 low, 2–3 elevated, ≥4 high.

ต้องมี plaque assessment, calibrated IMT, calibrated plaque area และข้อมูลครบทั้งห้าปัจจัยผู้ป่วยก่อนแสดง tier; ค่าที่ขาดทำให้ “unavailable” พร้อมเหตุผล ไม่ถือ missing เป็นศูนย์ (`pipeline.py:23-42,172-191`; `risk.py:20-34`). ดังนั้น demo IMT และ IMT จาก assumed spacing ไม่สามารถทำให้ได้ IMT point. เกณฑ์เหล่านี้เป็น heuristic ที่ยังไม่ผ่าน validation กับผลลัพธ์ผู้ป่วย; ไม่มี label stroke หรือ clinical decision ในชุดข้อมูลนี้ จึงไม่มีหลักฐานบอก clinical risk accuracy.

## 8. ผลประเมินล่าสุดที่มีอยู่

ไม่รันการประเมินใหม่ตามคำขอ; ตารางนี้ถอดจาก `evaluation/accuracy_20261030/REPORT.md` และ `metrics.json`. Dice/IoU แบบ global คำนวณจาก pixel รวมทุกภาพ; mean per-image คือเฉลี่ย metric ของแต่ละภาพ. False positive plaque เป็นสัดส่วนภาพ GT-negative ที่มี predicted foreground อย่างน้อยหนึ่ง pixel ไม่ใช่ pixel-level FPR.

| โมเดล/ชุด | n | Global Dice | Global IoU | Mean image Dice / IoU | False positive / IMT MAE |
|---|---:|---:|---:|---:|---|
| Plaque checkpoint validation ที่บันทึก | 98 | .621561 | ไม่พบใน checkpoint | — | — |
| Plaque validation คำนวณซ้ำจาก checkpoint | 98 | .621561 | ไม่ได้รายงาน | — | ตรงกับ checkpoint ถึง precision ที่รายงาน |
| Plaque `plaque_v3/test` (exploratory) | 52 (28 positive, 24 negative) | .6682 | .5017 | .5967 / .5259 | negative มี prediction 11/24 = 45.83% |
| Wall checkpoint validation ที่บันทึก | 103 | .762632 | ไม่พบใน checkpoint | — | — |
| Wall validation, stretch | 103 | .762632 | .616333 global IoU | .715896 / .570884 | ตรงค่า checkpoint; ไม่พิสูจน์ training mode เพราะไม่มี config/run command |
| Wall validation, letterbox | 103 | .725780 | .569588 global IoU | .681620 / .530429 | ต่ำกว่า stretch |
| Wall CUBS test หลังตัดภาพซ้ำกับ train | 100 | .7446 | .5932 | .7305 / .5855 | IMT MAE .2740 mm (274 µm), n=100 |

**การแยกตัวเลข:** validation Dice ที่ checkpoint บันทึกไม่ใช่ held-out test. ผล test wall 102 ภาพก่อนตัดซ้ำมี Dice .7454, IoU .5942, mean Dice .7316/IoU .5868 และ IMT MAE .27444 mm; ไม่ใช้เป็นค่าหลักเพราะมี exact duplicates. IMT MAE คือความต่างระหว่าง geometry จาก mask ที่โมเดลทำนายกับ `gt_imt_mm` ที่สร้างจาก LI/MA profile และใช้ calibration ต่อภาพ; เป็นการเทียบวิธีวัด ไม่ใช่ independent clinical truth. Plaque test ถูกใช้เลือกโมเดลและ patient split ไม่ทราบ จึงควรตีความอย่างระมัดระวัง. ผลเหล่านี้ไม่บอกความแม่นยำของ risk tier หรือ stroke.

Overlays ที่บันทึกไว้: plaque ดี `evaluation/accuracy_20261030/plaque_good_test_526_dice0.91.png`; plaque พลาด `plaque_failure_test_536_dice0.00.png`; false positives `plaque_false_positive_test_499.png` และ `plaque_false_positive_test_500.png`; wall ดี `wall_good_083_dice0.91.png`; wall พลาด `wall_failure_062_dice0.34.png`. สีเขียวคือ ground truth, แดงคือ prediction-only, น้ำเงินคือส่วน label ที่โมเดลพลาด ตาม README/รายงาน evaluation.

## 9. ปัญหา พบแล้ว แก้แล้ว และยังไม่ยืนยัน

| ประเด็น | สถานะตาม snapshot |
|---|---|
| Demo/assumed-spacing IMT เคยไหลเข้าคะแนน | แก้แล้วในโค้ดปัจจุบัน: mm estimate แยกจาก calibrated value, demo ถูกกันออกจาก risk และข้อมูลไม่ครบแสดง unavailable. มี regression tests ในโครงการ แต่เอกสารนี้ไม่ได้รันทดสอบซ้ำ |
| Blank/zero spacing | callback แปลง blank, string ว่าง และ zero เป็น `None`; รายงานแสดงเฉพาะหน่วย px/px², ซ่อน assumed-scale mm/mm² และไม่จัดชั้น plaque จากขนาด. Risk unavailable พร้อมเหตุผลเรื่อง scanner calibration; pipeline ยังคงปฏิเสธ spacing ที่ไม่ finite หรือค่าติดลบ |
| Wall checkpoint resize provenance | แก้ความคลาดเคลื่อนของการตรวจสอบแล้ว: checkpoint ไม่มี `resize_mode`, แต่ loader เดิม fallback เป็น `stretch`; จึงไม่มี letterbox/restore mismatch ใน checkpoint นี้. Validation score .762632 ทำซ้ำได้ด้วย stretch ไม่ใช่ letterbox. คำสั่งฝึกจริงไม่ถูกเก็บ จึงยังยืนยัน training mode ไม่ได้. เปรียบเทียบ inference และ overlays ที่ `evaluation/wall_geometry_trace_20261030/REPORT.md` |
| IMT segmentation failure | ยืนยันบน CUBS `test/062`: มี mask ให้คำนวณ 18.762 px แต่ label-derived reference 4.864 px; Dice .339 และ IMT error .8686 mm เมื่อใช้ calibration จริง. ปัญหาคือ wall mask ผิด ไม่ใช่ขาด calibration. ดู overlay ใน evaluation trace |
| ความน่าเชื่อถือ test | plaque split exploratory และไม่ grouped ตามผู้ป่วย; CUBS มีภาพซ้ำข้าม train/test 2 ภาพ (ตัดออกจาก metric หลัก) และไม่ทราบ patient mapping สำหรับภาพอื่น. การไม่มี exact duplicate ไม่ยืนยัน patient independence |
| Clinical accuracy | ไม่มี outcome stroke/label risk decision; วัด clinical prediction accuracy ไม่ได้. Dice/IoU/IMT MAE ไม่ใช่หลักฐานว่า High/Low risk ถูกต้องทางคลินิก |

**สรุปสิ่งที่ยืนยันได้:** pipeline คำนวณ IMT เป็น px จาก LI/MA แม้ไม่มี calibration; conversion เป็น mm ต้องมี spacing. Checkpoint stretch ทำซ้ำ validation Dice ได้; sample 062 แสดงว่า model segmentation อาจผิดมากแม้ค่าพิกเซลคำนวณได้. **สิ่งที่ยังไม่ยืนยัน:** ภาพที่ผู้ใช้ upload จริงไม่มีใน workspace, training resize mode จริงไม่มี config, ความถูกต้องทางคลินิก และความเป็นอิสระระดับผู้ป่วยของ test. ดู evaluation trace สำหรับผล inference และข้อจำกัด. ไม่มีการฝึกโมเดลหรือเปลี่ยน threshold.
