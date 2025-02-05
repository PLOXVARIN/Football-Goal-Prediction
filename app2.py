import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageTk

# โหลดโมเดลและ Scaler
model = tf.keras.models.load_model('model.h5')
scaler_scale = np.load('scaler.npy')
scaler_min = np.load('min.npy')

class FootballGoalPredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("Football Goal Prediction AI")
        master.geometry("800x600")  # ปรับขนาดหน้าจอให้สมส่วนมากขึ้น
        master.configure(bg='#2E8B57')
        
        # ตั้งค่า Grid เพื่อควบคุมการขยาย
        master.grid_columnconfigure(0, weight=0)  # คอลัมน์ซ้ายไม่ขยาย
        master.grid_columnconfigure(1, weight=1)  # คอลัมน์ขวาขยายได้
        master.grid_rowconfigure(0, weight=0)     # แถวบนไม่ขยาย
        master.grid_rowconfigure(1, weight=0)     # แถวล่างไม่ขยาย
        
        self.configure_styles()
        self.create_widgets()
        self.load_data()

    def configure_styles(self):
        """กำหนดรูปแบบ GUI ทั้งหมด"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Style สำหรับ Frame
        style.configure('Custom.TLabelframe', 
                        font=('Arial', 12, 'bold'), 
                        borderwidth=2, 
                        relief='ridge',
                        background='#FFFFFF')
        
        style.configure('Custom.TLabelframe.Label', 
                        foreground='#3F51B5', 
                        background='#FFFFFF',
                        font=('Arial', 12, 'bold'))
        
        # Style สำหรับตาราง
        style.configure('Custom.Treeview', 
                        font=('Arial', 10),
                        rowheight=25,
                        background='#FFFFFF',
                        fieldbackground='#FFFFFF',
                        foreground='#333333')
        
        # Style สำหรับปุ่ม
        style.configure('Accent.TButton', 
                        font=('Arial', 12, 'bold'), 
                        foreground='white', 
                        background='#4CAF50', 
                        padding=8)
        
        # Style สำหรับผลลัพธ์
        style.configure('Result.TLabel', 
                        font=('Arial', 16, 'bold'),
                        foreground='#3F51B5',
                        background='#FFFFFF')

    def create_widgets(self):
        """สร้างส่วนประกอบ GUI ทั้งหมด"""
        # ================= ส่วนแสดงข้อมูล =================
        self.data_frame = ttk.LabelFrame(self.master, text="Dataset Preview", style='Custom.TLabelframe')
        self.data_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.data_frame.grid_propagate(False)
        self.data_frame.config(width=350, height=200)  # ปรับขนาด Frame ให้เล็กลง
        
        # ตารางข้อมูล
        self.tree = ttk.Treeview(self.data_frame, columns=('distance', 'angle', 'speed', 'skill', 'goal'), 
                                 show='headings', height=6, style='Custom.Treeview')
        for col in ['distance', 'angle', 'speed', 'skill', 'goal']:
            self.tree.heading(col, text=col.capitalize(), anchor='center')
            self.tree.column(col, width=65, anchor='center')
        self.tree.pack(fill='both', expand=True, padx=5, pady=5)

        # ================= ส่วนป้อนข้อมูล =================
        self.input_frame = ttk.LabelFrame(self.master, text="Input Parameters", style='Custom.TLabelframe')
        self.input_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        self.input_frame.grid_propagate(False)
        self.input_frame.config(width=350, height=250)  # ปรับขนาด Frame ให้เล็กลง
        
        # สร้าง Slider
        self.create_slider('Distance (m):', 'distance', 0, 100, 0)
        self.create_slider('Angle (°):', 'angle', 0, 180, 1)
        self.create_slider('Speed (km/h):', 'speed', 0, 120, 2)
        self.create_slider('Skill (1-5):', 'skill', 1, 5, 3)
        
        # ปุ่ม Predict (อยู่ตรงกลาง)
        ttk.Button(self.input_frame, text="Predict Goal!", command=self.predict, 
                   style='Accent.TButton').grid(row=4, column=0, columnspan=3, pady=10, sticky='ew')

        # ================= ส่วนแสดงผล =================
        self.result_frame = ttk.LabelFrame(self.master, text="Prediction Result", style='Custom.TLabelframe')
        self.result_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky='nsew')
        self.result_frame.config(width=400, height=500)  # ปรับขนาด Frame ให้เล็กลง
        
        # ภาพผลลัพธ์
        self.canvas = tk.Canvas(self.result_frame, width=380, height=300, bg='white', highlightthickness=0)
        self.canvas.pack(pady=10, fill='both', expand=True)
        
        # Progress Bar
        self.progress = ttk.Progressbar(self.result_frame, length=380, mode='determinate')
        self.progress.pack(pady=10)
        
        # ข้อความผลลัพธ์
        self.result_label = ttk.Label(self.result_frame, text="Probability: 0.00%", style='Result.TLabel')
        self.result_label.pack(pady=10)

    def create_slider(self, label, name, min_val, max_val, row):
        """สร้าง Slider และ Entry"""
        frame = ttk.Frame(self.input_frame)
        frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        
        ttk.Label(frame, text=label, width=15, style='TLabel').pack(side='left', padx=5)
        var = tk.DoubleVar()
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var,
                           command=lambda v, n=name: self.update_entry(n, v))
        slider.pack(side='left', fill='x', expand=True, padx=5)
        
        entry = ttk.Entry(frame, width=6, font=('Arial', 10), justify='center')
        entry.pack(side='left', padx=5)
        setattr(self, f'{name}_var', var)
        setattr(self, f'{name}_entry', entry)

    def update_entry(self, name, value):
        """อัปเดตค่าใน Entry เมื่อเลื่อน Slider"""
        entry = getattr(self, f'{name}_entry')
        entry.delete(0, tk.END)
        entry.insert(0, f"{float(value):.1f}")

    def load_data(self):
        """โหลดข้อมูลตัวอย่างเข้า Treeview"""
        try:
            df = pd.read_csv('data/dataset.csv')
            for _, row in df.head(10).iterrows():
                self.tree.insert('', 'end', values=(
                    f"{row['distance']:.1f}",
                    f"{row['angle']:.1f}",
                    f"{row['speed']:.1f}",
                    f"{row['skill']:.1f}",
                    row['goal']
                ))
        except FileNotFoundError:
            messagebox.showerror("Error", "File 'data/dataset.csv' not found!")

    def predict(self):
        """ทำนายผลและอัปเดต GUI"""
        try:
            inputs = [
                float(self.distance_entry.get()),
                float(self.angle_entry.get()),
                float(self.speed_entry.get()),
                float(self.skill_entry.get())
            ]
            
            inputs_norm = (np.array(inputs) - scaler_min) / scaler_scale
            prediction = model.predict(inputs_norm.reshape(1, -1), verbose=0)[0][0]
            
            # อัปเดตภาพ
            self.canvas.delete("all")
            img = Image.open("images/goal.png" if prediction > 0.5 else "images/no_goal.png")
            img = img.resize((380, 200))  # กำหนดขนาดภาพให้สอดคล้องกับ Canvas
            self.current_image = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor='nw', image=self.current_image)
            
            # อัปเดตผลลัพธ์
            probability = prediction * 100
            self.progress['value'] = probability
            color = '#4CAF50' if probability > 50 else '#F44336'
            result_text = "Shot on goal ⚽" if probability > 50 else "Missed the goal ❌"
            self.result_label.config(
                text=f"Probability: {result_text}", 
                foreground=color
            )

        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FootballGoalPredictionApp(root)
    root.mainloop()
