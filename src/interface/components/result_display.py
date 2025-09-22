# src/interface/components/result_display.py

import tkinter as tk
from tkinter import ttk

class ResultDisplay:
    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
        # Letra detectada
        self.letter_var = tk.StringVar(value="-")
        self.letter_label = ttk.Label(
            self.frame, 
            textvariable=self.letter_var,
            font=('Arial', 48, 'bold')
        )
        self.letter_label.pack(pady=10)
        
        # Texto formado
        self.text_widget = tk.Text(self.frame, height=8, width=25)
        self.text_widget.pack(pady=5)
    
    def update_letter(self, letter):
        """Actualizar la letra mostrada"""
        self.letter_var.set(letter if letter else "-")
    
    def add_letter(self):
        """Agregar letra al texto"""
        letter = self.letter_var.get()
        if letter and letter != "-":
            self.text_widget.insert(tk.END, letter)
    
    def add_space(self):
        """Agregar espacio al texto"""
        self.text_widget.insert(tk.END, " ")
    
    def clear_text(self):
        """Limpiar el texto"""
        self.text_widget.delete(1.0, tk.END)
        self.letter_var.set("-")
    
    def get_text(self):
        """Obtener el texto actual"""
        return self.text_widget.get(1.0, tk.END).strip()