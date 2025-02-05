import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk  # เพิ่มการใช้ PIL
import pandas as pd

# โหลดโมเดลและ Scaler
model = tf.keras.models.load_model('model.h5')
scaler_scale = np.load('scaler.npy')
scaler_min = np.load('min.npy')

class FootballGoalPredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("Football Goal Prediction AI")
        master.geometry("700x400")  # ขนาดหน้าต่างเริ่มต้น
        master.configure(bg='#FFFFFF')  # เปลี่ยนพื้นหลังเป็นสีขาว
        
        # ตั้งค่า Style ใหม่ทั้งหมด
        self.configure_styles()
        
        # ================= ส่วนแสดงข้อมูล =================
        self.data_frame = ttk.LabelFrame(master, text="Dataset Preview", style='Custom.TLabelframe')
        self.data_frame.grid(row=0, column=0, padx=8, pady=8, sticky='nsew')
        
        # ตารางแสดงข้อมูล
        self.tree = ttk.Treeview(self.data_frame, columns=('distance', 'angle', 'speed', 'skill', 'goal'),
                         show='headings', height=6, style='Custom.Treeview')

        # กำหนดหัวข้อตาราง
        self.tree.heading('distance', text='Distance (m)', anchor='center')
        self.tree.heading('angle', text='Angle (°)', anchor='center')
        self.tree.heading('speed', text='Speed (km/h)', anchor='center')
        self.tree.heading('skill', text='Skill (1-5)', anchor='center')
        self.tree.heading('goal', text='Goal (1/0)', anchor='center')

        # กำหนดการจัดรูปแบบคอลัมน์
        self.tree.column('distance', width=90, anchor='center')
        self.tree.column('angle', width=90, anchor='center')
        self.tree.column('speed', width=90, anchor='center')
        self.tree.column('skill', width=90, anchor='center')
        self.tree.column('goal', width=90, anchor='center')
        self.input_frame = ttk.LabelFrame(...)
        self.input_frame.grid(...)

# กำหนดให้คอลัมน์ 0,1,2 มีน้ำหนักเท่ากัน (ปรับก่อนสร้างปุ่ม)
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_columnconfigure(1, weight=1)
        self.input_frame.grid_columnconfigure(2, weight=1)

# ปุ่ม Predict (แก้ไขส่วน grid)
        ttk.Button(...).grid(row=4, column=0, columnspan=3, pady=12, sticky='we')

        # เพิ่มรายการใน Treeview
        self.tree.pack(fill='both', expand=True, padx=8, pady=8)

        # ================= ส่วนป้อนข้อมูล =================
        self.input_frame = ttk.LabelFrame(master, text="Input Parameters", style='Custom.TLabelframe')
        self.input_frame.grid(row=1, column=0, padx=8, pady=8, sticky='nsew')
        
        # สร้าง Slider และ Entry
        self.create_slider('Distance (m):', 'distance', 0, 100, row=0)
        self.create_slider('Angle (°):', 'angle', 0, 180, row=1)
        self.create_slider('Speed (km/h):', 'speed', 0, 120, row=2)
        self.create_slider('Skill (1-5):', 'skill', 1, 5, row=3)

        # ปุ่ม Predict
        ttk.Button(self.input_frame, text="Predict Goal!", command=self.predict, 
                 style='Accent.TButton').grid(row=4, columnspan=3, pady=12, sticky='we')

        # ================= ส่วนแสดงผล =================
        self.result_frame = ttk.LabelFrame(master, text="Prediction Result", style='Custom.TLabelframe')
        self.result_frame.grid(row=0, column=1, rowspan=2, padx=8, pady=8, sticky='nsew')
        
        # แสดงผลลัพธ์
        self.result_label = ttk.Label(self.result_frame, text="Prediction Result: N/A", 
                                    style='Result.TLabel')
        self.result_label.pack(pady=40)

        # เพิ่ม Label สำหรับแสดงภาพผลลัพธ์
        self.result_image_label = ttk.Label(self.result_frame)
        self.result_image_label.pack(pady=20)

        # ================= โหลดข้อมูลตัวอย่าง =================
        self.load_data()

    def configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # กำหนด Style สำหรับ Frame
        style.configure('Custom.TLabelframe', font=('Arial', 10), borderwidth=2, relief='ridge', background='#FFFFFF')
        style.configure('Custom.TLabelframe.Label', font=('Arial', 10, 'bold'), background='#FFFFFF')
        
        # กำหนด Style สำหรับ Treeview
        style.configure('Custom.Treeview', font=('Arial', 9), rowheight=22, background='#FFFFFF')
        style.configure('Custom.Treeview.Heading', font=('Arial', 10, 'bold'), anchor='center', 
                        background='#E0E0E0', relief='flat', padding=(8, 4))  # ปรับการเว้นระยะและพื้นหลัง

        # เปลี่ยนการเน้น (Highlight) ให้เหมาะสม
        style.map('Custom.Treeview',
                  background=[('selected', '#B0E57C')])  # สีเมื่อเลือกแถว

        # กำหนด Style สำหรับปุ่ม
        style.configure('Accent.TButton', font=('Arial', 12), foreground='white', 
                      background='#4CAF50', padding=8)
        
        # กำหนด Style สำหรับ Label ผลลัพธ์
        style.configure('Result.TLabel', font=('Arial', 14, 'bold'))

    def create_slider(self, label, name, min_val, max_val, row):
        frame = ttk.Frame(self.input_frame)
        frame.grid(row=row, column=0, columnspan=3, sticky='we', pady=6)
        
        ttk.Label(frame, text=label, width=16, font=('Arial', 10)).pack(side='left', padx=6)
        var = tk.DoubleVar()
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var,
                          command=lambda v, n=name: self.update_entry(n, v))
        slider.pack(side='left', fill='x', expand=True, padx=6)
        
        entry = ttk.Entry(frame, width=8, font=('Arial', 10), justify='center')
        entry.pack(side='left', padx=6)
        setattr(self, f'{name}_var', var)
        setattr(self, f'{name}_entry', entry)

    def update_entry(self, name, value):
        entry = getattr(self, f'{name}_entry')
        entry.delete(0, tk.END)
        entry.insert(0, f"{float(value):.1f}")

    def load_data(self):
        # ใส่ข้อมูลตัวอย่างใน Treeview
        df = pd.read_csv('data/dataset.csv')
        for _, row in df.head(6).iterrows():
            self.tree.insert('', 'end', values=(f"{row['distance']:.1f}",
                                                f"{row['angle']:.1f}",
                                                f"{row['speed']:.1f}",
                                                f"{row['skill']:.1f}",
                                                row['goal']))

        def predict(self):
            try:
        # ... (โค้ดส่วนอื่นเหมือนเดิม)

        # อัปเดตความน่าจะเป็นและข้อความ
                probability = prediction * 100
                self.progress['value'] = probability
                color = '#4CAF50' if probability > 50 else '#F44336'
        
        # กำหนดข้อความตามผลลัพธ์
                result_text = "Shot on goal ⚽" if probability > 50 else "Missed the goal ❌"
        
        # แสดงผลพร้อมข้อความ
                self.result_label.config(
                    text=f"Probability: {probability:.2f}%\n{result_text}", 
                    foreground=color
        )

        # ... (ส่วนอัปเดตกราฟเหมือนเดิม)

            except Exception as e:
             messagebox.showerror("Error", f"Invalid input: {str(e)}")

    def show_image(self, image_path):
        # โหลดและแสดงภาพ
        image = Image.open(image_path)
        image = image.resize((100, 100), Image.ANTIALIAS)  # ปรับขนาดภาพ
        photo = ImageTk.PhotoImage(image)

        self.result_image_label.config(image=photo)
        self.result_image_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = FootballGoalPredictionApp(root)
    root.mainloop()
