import numpy as np
import pandas as pd

# สร้าง DataFrame ว่าง
data = []

# กำหนดจำนวนข้อมูล
num_samples = 1000
half_samples = num_samples // 2  # ครึ่งหนึ่งเป็น goal = 1 และอีกครึ่งเป็น goal = 0

# ✅ ข้อมูลที่เป็นประตู (goal = 1)
for _ in range(half_samples):
    distance = np.random.randint(5, 40)  # ใกล้ประตูมากขึ้น (5 - 40 เมตร)
    angle = np.random.randint(30, 150)   # อยู่ในช่วงมุมที่เหมาะสม (30° - 150°)
    speed = np.random.randint(60, 120)   # ความแรงสูงขึ้น (60 - 120 km/h)
    skill = np.random.randint(3, 6)      # ทักษะระดับกลางถึงสูง (3 - 5)
    goal = 1  # ทำประตูได้

    data.append([distance, angle, speed, skill, goal])

# ❌ ข้อมูลที่พลาด (goal = 0)
for _ in range(half_samples):
    distance = np.random.randint(40, 100)  # ไกลจากประตูมาก (40 - 100 เมตร)
    angle = np.random.choice(list(range(0, 30)) + list(range(150, 180)))  # มุมแคบเกินไปหรือกว้างเกินไป
    speed = np.random.randint(10, 60)  # ความเร็วต่ำเกินไป (10 - 60 km/h)
    skill = np.random.randint(1, 4)  # ทักษะต่ำถึงปานกลาง (1 - 3)
    goal = 0  # พลาดประตู

    data.append([distance, angle, speed, skill, goal])

# สร้าง DataFrame
df = pd.DataFrame(data, columns=['distance', 'angle', 'speed', 'skill', 'goal'])

# บันทึกเป็น CSV
df.to_csv("dataset.csv", index=False)

# แสดงข้อมูลตัวอย่าง
print(df.head(10))
