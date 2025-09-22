# src/interface/components/camera_widget.py

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

class CameraWidget:
    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.Frame(parent)
        self.label = ttk.Label(self.frame, text="Cámara no iniciada")
        self.label.pack(expand=True)
    
    def update_frame(self, frame):
        """Actualizar el frame mostrado"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            self.label.configure(image=frame_tk)
            self.label.image = frame_tk
        except Exception as e:
            print(f"Error actualizando frame: {e}")