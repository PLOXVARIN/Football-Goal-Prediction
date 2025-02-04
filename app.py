import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# โหลดโมเดลและ Scaler
model = tf.keras.models.load_model('model.h5')
scaler_scale = np.load('scaler.npy')
scaler_min = np.load('min.npy')

class FootballGoalPredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("Football Goal Prediction AI")
        master.geometry("1000x700")
        master.configure(bg='#2E8B57')

        # โหลดภาพพื้นหลัง
        self.field_image = ImageTk.PhotoImage(Image.open("images/field.jpg").resize((400, 250)))
        
        # ================= ส่วนแสดงข้อมูล =================
        self.data_frame = ttk.LabelFrame(master, text="Dataset Preview")
        self.data_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        # ตารางแสดงข้อมูล
        self.tree = ttk.Treeview(self.data_frame, columns=('distance', 'angle', 'speed', 'skill', 'goal'), show='headings')
        self.tree.heading('distance', text='Distance (m)')
        self.tree.heading('angle', text='Angle (°)')
        self.tree.heading('speed', text='Speed (km/h)')
        self.tree.heading('skill', text='Skill (1-5)')
        self.tree.heading('goal', text='Goal (1/0)')
        self.tree.pack()

        # โหลดข้อมูลตัวอย่าง
        self.load_data()

        # ================= ส่วนป้อนข้อมูล =================
        self.input_frame = ttk.LabelFrame(master, text="Input Parameters")
        self.input_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        # สร้าง Slider และ Entry
        self.create_slider('Distance (m):', 'distance', 0, 100, row=0)
        self.create_slider('Angle (°):', 'angle', 0, 180, row=1)
        self.create_slider('Speed (km/h):', 'speed', 0, 120, row=2)
        self.create_slider('Skill (1-5):', 'skill', 1, 5, row=3)

        # ปุ่ม Predict
        ttk.Button(self.input_frame, text="Predict Goal!", command=self.predict).grid(row=4, columnspan=2, pady=10)

        # ================= ส่วนแสดงผล =================
        self.result_frame = ttk.LabelFrame(master, text="Prediction Result")
        self.result_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky='nsew')

        # แสดงภาพสนาม
        self.canvas = tk.Canvas(self.result_frame, width=400, height=250)
        self.canvas.create_image(0, 0, anchor='nw', image=self.field_image)
        self.canvas.pack(pady=10)

        # แสดงผลความน่าจะเป็น
        self.result_label = ttk.Label(self.result_frame, text="Probability: 0%", font=('Arial', 16))
        self.result_label.pack(pady=10)

        # แสดงกราฟ
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=self.result_frame)
        self.canvas_graph.get_tk_widget().pack()

    def create_slider(self, label, name, min_val, max_val, row):
        ttk.Label(self.input_frame, text=label).grid(row=row, column=0, padx=5, pady=5, sticky='e')
        setattr(self, f'{name}_var', tk.DoubleVar())
        slider = ttk.Scale(self.input_frame, from_=min_val, to=max_val, 
                          variable=getattr(self, f'{name}_var'),
                          command=lambda v, n=name: self.update_entry(n, v))
        slider.grid(row=row, column=1, padx=5, pady=5, sticky='we')
        entry = ttk.Entry(self.input_frame, width=5)
        entry.grid(row=row, column=2, padx=5, pady=5)
        setattr(self, f'{name}_entry', entry)

    def update_entry(self, name, value):
        entry = getattr(self, f'{name}_entry')
        entry.delete(0, tk.END)
        entry.insert(0, f"{float(value):.1f}")

    def load_data(self):
        df = pd.read_csv('data/dataset.csv')
        for _, row in df.head(10).iterrows():
            self.tree.insert('', 'end', values=(row['distance'], row['angle'], row['speed'], row['skill'], row['goal']))

    def predict(self):
        try:
            # รับค่าจาก GUI
            inputs = [
                float(self.distance_entry.get()),
                float(self.angle_entry.get()),
                float(self.speed_entry.get()),
                float(self.skill_entry.get())
            ]

            # Normalize ข้อมูล
            inputs_norm = (np.array(inputs) - scaler_min) / scaler_scale
            prediction = model.predict(inputs_norm.reshape(1, -1))[0][0]

            # แสดงผล
            probability = f"{prediction*100:.1f}%"
            result_text = "GOAL! 🎉" if prediction > 0.5 else "No Goal... 😢"
            self.result_label.config(text=f"Probability: {probability}\n{result_text}")

            # อัปเดตกราฟ
            self.ax.clear()
            self.ax.bar(['No Goal', 'Goal'], [1-prediction, prediction], color=['red', 'green'])
            self.ax.set_ylim(0, 1)
            self.canvas_graph.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FootballGoalPredictionApp(root)
    root.mainloop()