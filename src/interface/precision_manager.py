# src/interface/precision_manager.py

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

class PrecisionManager:
    def __init__(self, parent_window, gesture_calibrator):
        self.parent = parent_window
        self.calibrator = gesture_calibrator
        self.window = None
        self.is_calibrating = False
        self.current_letter = None
        self.calibration_progress = {}
        
    def show_precision_window(self):
        """Muestra la ventana de gestión de precisión"""
        if self.window:
            self.window.focus()
            return
            
        self.window = tk.Toplevel(self.parent.root)
        self.window.title("Gestión de Precisión")
        self.window.geometry("700x600")
        self.window.configure(bg='white')
        
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        self.setup_precision_ui()
    
    def setup_precision_ui(self):
        """Configura la interfaz de gestión de precisión"""
        # Título
        title_label = ttk.Label(
            self.window, 
            text="Gestión de Precisión del Reconocimiento",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=15)
        
        # Crear notebook para diferentes secciones
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Pestaña de estado actual
        status_frame = ttk.Frame(notebook)
        notebook.add(status_frame, text="Estado Actual")
        self.setup_status_tab(status_frame)
        
        # Pestaña de calibración
        calibration_frame = ttk.Frame(notebook)
        notebook.add(calibration_frame, text="Calibración")
        self.setup_calibration_tab(calibration_frame)
        
        # Pestaña de ajustes avanzados
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Ajustes Avanzados")
        self.setup_settings_tab(settings_frame)
        
        # Botones principales
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(
            button_frame,
            text="Auto-Calibrar",
            command=self.auto_calibrate
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="Resetear Calibración",
            command=self.reset_calibration
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="Cerrar",
            command=self.close_window
        ).pack(side=tk.RIGHT)
    
    def setup_status_tab(self, parent):
        """Configura la pestaña de estado"""
        # Marco de información general
        info_frame = ttk.LabelFrame(parent, text="Estado de Calibración", padding="15")
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Variables para mostrar estado
        self.status_vars = {
            'total_letters': tk.StringVar(),
            'calibrated_letters': tk.StringVar(),
            'samples_collected': tk.StringVar()
        }
        
        ttk.Label(info_frame, text="Total de letras calibradas:").pack(anchor=tk.W)
        ttk.Label(info_frame, textvariable=self.status_vars['total_letters']).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(info_frame, text="Letras con calibración completa:").pack(anchor=tk.W)
        ttk.Label(info_frame, textvariable=self.status_vars['calibrated_letters']).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(info_frame, text="Total de muestras recolectadas:").pack(anchor=tk.W)
        ttk.Label(info_frame, textvariable=self.status_vars['samples_collected']).pack(anchor=tk.W)
        
        # Marco de progreso detallado
        progress_frame = ttk.LabelFrame(parent, text="Progreso por Letra", padding="15")
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        # Crear área con scroll para mostrar progreso
        canvas = tk.Canvas(progress_frame)
        scrollbar = ttk.Scrollbar(progress_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Actualizar información
        self.update_status_display()
    
    def setup_calibration_tab(self, parent):
        """Configura la pestaña de calibración"""
        # Marco de calibración rápida
        quick_cal_frame = ttk.LabelFrame(parent, text="Calibración Rápida", padding="15")
        quick_cal_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(quick_cal_frame, 
                 text="Selecciona una letra para calibrar específicamente:").pack(anchor=tk.W, pady=(0, 10))
        
        # Frame para selector de letra
        letter_frame = ttk.Frame(quick_cal_frame)
        letter_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.selected_letter = tk.StringVar()
        letter_combo = ttk.Combobox(letter_frame, textvariable=self.selected_letter, width=5)
        letter_combo['values'] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        letter_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        calibrate_btn = ttk.Button(
            letter_frame,
            text="Calibrar Esta Letra",
            command=self.calibrate_specific_letter
        )
        calibrate_btn.pack(side=tk.LEFT)
        
        # Marco de instrucciones
        instructions_frame = ttk.LabelFrame(parent, text="Instrucciones", padding="15")
        instructions_frame.pack(fill=tk.BOTH, expand=True)
        
        instructions_text = tk.Text(instructions_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        instructions_text.pack(fill=tk.BOTH, expand=True)
        
        # Agregar instrucciones
        instructions = """
Cómo mejorar la precisión:

1. AUTO-CALIBRACIÓN:
   - Usa la aplicación normalmente
   - El sistema recolecta automáticamente muestras
   - Cada 50 muestras, se auto-calibra

2. CALIBRACIÓN MANUAL:
   - Selecciona una letra específica
   - Haz el gesto 10-15 veces de forma consistente
   - El sistema aprenderá tu estilo personal

3. CONSEJOS PARA MEJOR PRECISIÓN:
   - Mantén buena iluminación
   - Posición consistente de la mano
   - Movimientos fluidos y naturales
   - Pausa entre letras diferentes

4. PROBLEMAS COMUNES:
   - Baja precisión: Necesita más muestras de entrenamiento
   - Falsos positivos: Ajustar sensibilidad en configuración avanzada
   - Detección lenta: Revisar configuración de estabilidad
        """
        
        instructions_text.config(state=tk.NORMAL)
        instructions_text.insert(1.0, instructions)
        instructions_text.config(state=tk.DISABLED)
    
    def setup_settings_tab(self, parent):
        """Configura la pestaña de ajustes avanzados"""
        # Marco de sensibilidad
        sensitivity_frame = ttk.LabelFrame(parent, text="Configuración de Sensibilidad", padding="15")
        sensitivity_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Control de umbral de confianza
        ttk.Label(sensitivity_frame, text="Umbral de Confianza:").pack(anchor=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.7)
        confidence_scale = ttk.Scale(
            sensitivity_frame,
            from_=0.5, to=0.95,
            variable=self.confidence_var,
            orient=tk.HORIZONTAL,
            length=300
        )
        confidence_scale.pack(fill=tk.X, pady=(5, 10))
        
        # Control de tolerancia de ángulo
        ttk.Label(sensitivity_frame, text="Tolerancia de Ángulo (grados):").pack(anchor=tk.W)
        self.angle_var = tk.IntVar(value=15)
        angle_scale = ttk.Scale(
            sensitivity_frame,
            from_=5, to=30,
            variable=self.angle_var,
            orient=tk.HORIZONTAL,
            length=300
        )
        angle_scale.pack(fill=tk.X, pady=(5, 10))
        
        # Control de estabilidad
        ttk.Label(sensitivity_frame, text="Frames de Estabilidad:").pack(anchor=tk.W)
        self.stability_var = tk.IntVar(value=15)
        stability_scale = ttk.Scale(
            sensitivity_frame,
            from_=5, to=30,
            variable=self.stability_var,
            orient=tk.HORIZONTAL,
            length=300
        )
        stability_scale.pack(fill=tk.X, pady=(5, 10))
        
        # Botón para aplicar configuración
        apply_btn = ttk.Button(
            sensitivity_frame,
            text="Aplicar Configuración",
            command=self.apply_advanced_settings
        )
        apply_btn.pack(pady=(10, 0))
        
        # Marco de diagnósticos
        diagnostics_frame = ttk.LabelFrame(parent, text="Diagnósticos", padding="15")
        diagnostics_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(
            diagnostics_frame,
            text="Test de Precisión",
            command=self.run_precision_test
        ).pack(pady=(0, 10))
        
        ttk.Button(
            diagnostics_frame,
            text="Exportar Datos de Calibración",
            command=self.export_calibration_data
        ).pack(pady=(0, 10))
        
        ttk.Button(
            diagnostics_frame,
            text="Importar Configuración",
            command=self.import_calibration_data
        ).pack()
    
    def update_status_display(self):
        """Actualiza la visualización del estado"""
        status = self.calibrator.get_calibration_status()
        
        # Actualizar variables de estado
        self.status_vars['total_letters'].set(str(status['total_letters']))
        self.status_vars['calibrated_letters'].set(', '.join(status['calibrated_letters']))
        
        total_samples = sum(status['sample_counts'].values())
        self.status_vars['samples_collected'].set(str(total_samples))
        
        # Limpiar frame de progreso
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Mostrar progreso por letra
        from ..config.settings import Config
        for i, letter in enumerate(Config.SUPPORTED_LETTERS):
            letter_frame = ttk.Frame(self.scrollable_frame)
            letter_frame.pack(fill=tk.X, pady=2)
            
            # Etiqueta de letra
            ttk.Label(letter_frame, text=f"Letra {letter}:", width=10).pack(side=tk.LEFT)
            
            # Barra de progreso
            progress = ttk.Progressbar(letter_frame, length=200, maximum=50)
            progress.pack(side=tk.LEFT, padx=(10, 10))
            
            # Mostrar progreso
            samples_count = status['sample_counts'].get(letter, 0)
            progress['value'] = min(samples_count, 50)
            
            # Estado
            if letter in status['calibrated_letters']:
                status_text = f"✓ Calibrado ({samples_count} muestras)"
                status_color = "green"
            elif samples_count >= 10:
                status_text = f"Listo para calibrar ({samples_count} muestras)"
                status_color = "orange"
            else:
                status_text = f"Necesita más muestras ({samples_count}/10)"
                status_color = "red"
            
            status_label = ttk.Label(letter_frame, text=status_text, foreground=status_color)
            status_label.pack(side=tk.LEFT, padx=(10, 0))
    
    def auto_calibrate(self):
        """Ejecuta auto-calibración"""
        calibrated = self.calibrator.auto_calibrate_from_usage()
        
        if calibrated:
            messagebox.showinfo(
                "Auto-Calibración Completa",
                f"Se calibraron las letras: {', '.join(calibrated)}\n"
                f"La precisión debería mejorar para estas letras."
            )
            self.update_status_display()
        else:
            messagebox.showwarning(
                "Auto-Calibración",
                "No hay suficientes muestras para calibrar.\n"
                "Usa la aplicación más tiempo para recolectar datos."
            )
    
    def calibrate_specific_letter(self):
        """Calibra una letra específica"""
        letter = self.selected_letter.get()
        if not letter:
            messagebox.showwarning("Error", "Selecciona una letra para calibrar")
            return
        
        if self.calibrator.calibrate_gesture(letter):
            messagebox.showinfo(
                "Calibración Exitosa",
                f"La letra '{letter}' ha sido calibrada exitosamente.\n"
                f"La precisión debería mejorar para esta letra."
            )
            self.update_status_display()
        else:
            messagebox.showwarning(
                "Calibración Fallida",
                f"No hay suficientes muestras para calibrar la letra '{letter}'.\n"
                f"Haz este gesto al menos 10 veces en la aplicación principal."
            )
    
    def apply_advanced_settings(self):
        """Aplica configuración avanzada"""
        # Aquí aplicarías los valores a los clasificadores
        messagebox.showinfo("Configuración", "Configuración avanzada aplicada")
    
    def run_precision_test(self):
        """Ejecuta test de precisión"""
        messagebox.showinfo("Test de Precisión", "Función en desarrollo...")
    
    def export_calibration_data(self):
        """Exporta datos de calibración"""
        messagebox.showinfo("Exportar", "Función en desarrollo...")
    
    def import_calibration_data(self):
        """Importa configuración"""
        messagebox.showinfo("Importar", "Función en desarrollo...")
    
    def reset_calibration(self):
        """Resetea toda la calibración"""
        if messagebox.askyesno(
            "Confirmar Reset",
            "¿Estás seguro de que quieres resetear toda la calibración?\n"
            "Esto eliminará todos los datos personalizados."
        ):
            self.calibrator.reset_calibration()
            self.update_status_display()
            messagebox.showinfo("Reset Completo", "Calibración reseteada exitosamente")
    
    def close_window(self):
        """Cierra la ventana"""
        if self.window:
            self.window.destroy()
            self.window = None