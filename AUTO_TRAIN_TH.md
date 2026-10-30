# ฝึกอัตโนมัติจน validation F1 ถึง 0.90

วางโค้ดทั้งชุดทับสำเนาโฟลเดอร์ ti-artery-gpu เดิม ไม่รวม data และ weights ใน ZIP นี้
ต้องเตรียม data/plaque_v2/train และ val ตาม START_HERE_TH.md ก่อน

## คำสั่งฝึกต่อเนื่อง

```bash
python train_until_target.py --task plaque --data data/plaque_v2 --target 0.90 --max-runs 0
```

max-runs 0 หมายถึงทำ trial ใหม่ต่อไปจนถึงเป้า เกิด error หรือกด Ctrl+C
แต่ละ trial เริ่มจากศูนย์ ใช้ seed ใหม่ สลับขนาดภาพ 320/512 และ learning rate
เพื่อไม่เอาน้ำหนักจากโมเดลเดิมที่เคยเห็นข้อมูล validation ใหม่มาใช้
ค่าเริ่มต้น 100 epochs ต่อ trial, batch 4; ไม่ใช่การ resume โมเดลเดียวตลอดไป
หากไม่ระบุ max-runs ค่าเริ่มต้นคือ 6 trials แล้วหยุดถ้ายังไม่ถึงเป้า

ตัวอย่าง VRAM น้อย ให้เริ่มที่ 320 อย่างเดียวและลด batch:

```bash
python train_until_target.py --task plaque --data data/plaque_v2 --target 0.90 --max-runs 0 --sizes 320 --batch 2
```

ฝึก wall แยก (คะแนนคนละงาน อย่าเอามารวม):

```bash
python train_until_target.py --task wall --data data/cubs --target 0.90 --max-runs 0
```

## เป้าหมายที่วัด

เป้าหมายนี้คือ validation GLOBAL PIXEL F1/Dice ที่ความละเอียดฝึก threshold 0.5
คำนวณพิกเซลรวม validation ทั้งชุด ไม่ใช่ mean per-image, plaque-positive mean
หรือ image detection F1 และไม่ใช่คะแนน test หรือความแม่นยำการวัด IMT
จะหยุดหลัง epoch ที่ได้ค่าจริง >= 0.90 และบันทึก checkpoint เสร็จแล้ว
ไม่ใช้ตัวเลขปัดเศษในการตัดสิน ไม่อ่าน test ไม่ปรับ threshold อัตโนมัติ
หากเป้าหมายงานวิจัยกำหนด mean per-image หรือ plaque-positive 0.90 ต้องใช้การประเมินแยก
การทดลอง validation ซ้ำหลายครั้งอาจทำให้เลือกโมเดลที่เข้ากับ validation มากเกินไป
ต้องมี test ใหม่ที่กันไว้อิสระเพื่อยืนยันผล การรันไม่จำกัดไม่รับประกันว่าจะถึงเป้า

## ดูเวลา ไฟล์ และปัญหา

ทุกข้อความระบุเวลาไทยพร้อม +07:00 และแสดง output directory ตั้งแต่เริ่ม
ไฟล์จะอยู่ใน runs/plaque_YYYYMMDD_HHMMSS_PID/ (wall ใช้ชื่อ wall_...):

- status.json: สถานะล่าสุด เวลาที่เริ่ม/อัปเดต เวลาที่ผ่านไป คะแนนดีที่สุด และ path โมเดลดีที่สุด
- events.jsonl: เหตุการณ์เริ่มแต่ละ trial คะแนนแต่ละ epoch เวลา และรายละเอียด error
- config.json: ตัวเลือกที่ใช้รัน
- trial_0001/console.log: stdout/stderr/traceback ครบ พร้อมเวลาทุกบรรทัด
- trial_0001/best.pt: โมเดล validation ดีที่สุดใน trial นี้
- trial_0001/last.pt: โมเดล/optimizer/scheduler หลัง epoch ล่าสุดที่บันทึกสำเร็จ
- supervisor_error.log: traceback เมื่อผิดพลาดในตัวควบคุมเอง (ถ้ามี)

เมื่อเกิด error จะหยุด ไม่ retry ข้อมูลเสียหรือ OOM วนไม่สิ้นสุด
ERROR ระบุเวลา trainer_file, source_file/source_line หากมี traceback,
data_files หากตัวอ่านภาพรายงานชื่อไฟล์, problem, exit_code และ console_log
หาก OS ปิด process โดยไม่มี traceback จะไม่เดาไฟล์ต้นเหตุ ให้ดู exit code และ log
ใช้ workers=0 เป็นค่าเริ่มต้นเพื่อให้ระบุปัญหาจากไฟล์ภาพได้ตรงที่สุด
หากผิดพลาดก่อนสร้าง output directory เช่นสิทธิ์เขียนไม่พอ จะเห็นข้อความที่ Terminal
ไฟดับ/ปิดเครื่องทันทีไม่มีโอกาสเขียน ERROR; status อาจค้าง training ให้ตรวจ updated_at

สถานะ target_reached = ถึงเป้า validation, budget_exhausted = ครบ trial แล้วยังไม่ถึง,
error = มีปัญหา, interrupted = กด Ctrl+C
exit code: 0 ถึงเป้า, 2 หมด budget, 1 error, 130 ผู้ใช้หยุด

## หลังหยุด

เปิด status.json แล้วนำ path ใน best_checkpoint ไปประเมินด้วย eval_plaque.py
ตาม START_HERE_TH.md ดูคะแนนและ overlay ที่ความละเอียดต้นฉบับอีกครั้ง
อย่าคาดว่าคะแนนทุกนิยามจะเท่ากัน
หากต้องการฝึกโมเดลเดิมต่อ ใช้ train_plaque_unet.py พร้อม --resume และ --last
เป็น path last.pt ของ trial นั้น ใช้ size/base/arch/seed/lr/epochs เดิม
resume มีไว้ต่อ epoch budget เดิมที่ยังไม่จบ ไม่ใช่เพิ่ม epochs หลัง scheduler จบ
ตัว supervisor ไม่ resume งานเก่าโดยอัตโนมัติ; การรันคำสั่งใหม่จะเปิด output ใหม่

## การทดสอบของรุ่นนี้

ผ่าน unit tests 9 รายการ รวมตัวควบคุมด้วย trainer จำลอง:
หยุดเมื่อถึงเป้า ไม่หยุดจากการปัด 0.899999, รายงานเวลาไทย/ไฟล์ที่พัง,
ปฏิเสธ NaN และเริ่ม trial ถัดไปเมื่อยังไม่ถึงเป้า
ผ่าน Python compileall ยังไม่ได้รันฝึกเต็มด้วย GPU หรือยืนยัน F1 ใหม่
