# src/interface/reference_gallery.py

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

class ReferenceGallery:
    def __init__(self, parent_window):
        self.parent = parent_window
        self.window = None
        
        # Diccionario con descripciones de cada letra
        self.letter_descriptions = {
            'A': 'Puño cerrado con pulgar al lado',
            'B': 'Cuatro dedos extendidos juntos, pulgar doblado',
            'C': 'Mano en forma de C curva',
            'D': 'Índice extendido, otros dedos doblados con pulgar',
            'E': 'Dedos curvados hacia la palma',
            'F': 'Pulgar e índice formando círculo, otros extendidos',
            'G': 'Índice y pulgar extendidos horizontalmente',
            'H': 'Índice y medio extendidos horizontalmente',
            'I': 'Solo meñique extendido',
            'J': 'Meñique extendido con movimiento curvo',
            'K': 'Índice y medio en V, pulgar toca el medio',
            'L': 'Índice y pulgar en forma de L',
            'M': 'Pulgar bajo tres dedos doblados',
            'N': 'Pulgar bajo dos dedos doblados',
            'O': 'Todos los dedos formando círculo',
            'P': 'Similar a K pero orientado hacia abajo',
            'Q': 'Similar a G pero orientado hacia abajo',
            'R': 'Índice y medio cruzados',
            'S': 'Puño cerrado con pulgar sobre dedos',
            'T': 'Puño con pulgar entre índice y medio',
            'U': 'Índice y medio juntos hacia arriba',
            'V': 'Índice y medio separados en V',
            'W': 'Tres dedos extendidos (índice, medio, anular)',
            'X': 'Índice parcialmente doblado como gancho',
            'Y': 'Pulgar y meñique extendidos',
            'Z': 'Índice extendido haciendo movimiento zigzag'
        }
    
    def show_gallery(self):
        """Muestra la galería de referencias"""
        if self.window:
            self.window.focus()
            return
            
        self.window = tk.Toplevel(self.parent.root)
        self.window.title("Galería de Referencias - Alfabeto de Señas")
        self.window.geometry("900x700")
        self.window.configure(bg='white')
        
        # Protocolo de cierre
        self.window.protocol("WM_DELETE_WINDOW", self.close_gallery)
        
        self.setup_gallery_ui()
    
    def setup_gallery_ui(self):
        """Configura la interfaz de la galería"""
        # Título
        title_label = ttk.Label(
            self.window, 
            text="Alfabeto de Lenguaje de Señas - Referencias",
            font=('Arial', 18, 'bold')
        )
        title_label.pack(pady=15)
        
        # Frame principal con scroll
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Canvas con scrollbar
        canvas = tk.Canvas(main_frame, bg='white')
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Crear grid de letras
        self.create_letters_grid(scrollable_frame)
        
        # Empaquetar canvas y scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Frame inferior con botones
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Botón para cargar imágenes
        load_images_btn = ttk.Button(
            button_frame,
            text="Cargar Imágenes de Referencia",
            command=self.load_reference_images
        )
        load_images_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botón para entrenar modelo
        train_btn = ttk.Button(
            button_frame,
            text="Mejorar Detección",
            command=self.improve_detection
        )
        train_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botón cerrar
        close_btn = ttk.Button(
            button_frame,
            text="Cerrar",
            command=self.close_gallery
        )
        close_btn.pack(side=tk.RIGHT)
    
    def create_letters_grid(self, parent):
        """Crea el grid de letras con imágenes reales"""
        from ..config.settings import Config
        
        # Crear grid de 4 columnas
        cols = 4
        for i, letter in enumerate(Config.SUPPORTED_LETTERS):
            row = i // cols
            col = i % cols
            
            # Frame para cada letra
            letter_frame = ttk.LabelFrame(parent, text=f"Letra {letter}", padding="10")
            letter_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Intentar cargar imagen real
            image_widget = self.create_image_widget(letter_frame, letter)
            image_widget.pack(pady=5)
            
            # Descripción de la letra
            description = self.letter_descriptions.get(letter, "Descripción no disponible")
            desc_label = ttk.Label(
                letter_frame,
                text=description,
                font=('Arial', 9),
                wraplength=150,
                justify='center'
            )
            desc_label.pack(pady=5)
            
            # Botón para practicar esta letra específica
            practice_btn = ttk.Button(
                letter_frame,
                text=f"Practicar {letter}",
                command=lambda l=letter: self.practice_specific_letter(l)
            )
            practice_btn.pack(pady=5)
        
        # Configurar expansión del grid
        for i in range(cols):
            parent.columnconfigure(i, weight=1)
    
    def create_image_widget(self, parent, letter):
        """Crea el widget de imagen para una letra específica"""
        # Buscar imagen en diferentes formatos
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        image_path = None
        
        for ext in image_extensions:
            potential_path = os.path.join("assets", "reference_images", f"{letter}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path:
            try:
                # Cargar y redimensionar imagen
                original_image = Image.open(image_path)
                
                # Redimensionar manteniendo aspecto
                max_size = (120, 120)
                original_image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convertir para tkinter
                photo = ImageTk.PhotoImage(original_image)
                
                # Crear label con imagen
                image_label = tk.Label(parent, image=photo, relief='ridge', bd=2)
                image_label.image = photo  # Mantener referencia
                
                return image_label
                
            except Exception as e:
                print(f"Error cargando imagen para {letter}: {e}")
                # Fallback a placeholder
                return self.create_placeholder_widget(parent, letter)
        else:
            # No se encontró imagen, crear placeholder
            return self.create_placeholder_widget(parent, letter)
    
    def create_placeholder_widget(self, parent, letter):
        """Crea un placeholder cuando no hay imagen disponible"""
        placeholder = tk.Label(
            parent,
            text=f"{letter}",
            font=('Arial', 48, 'bold'),
            bg='lightgray',
            fg='darkblue',
            width=6,
            height=3,
            relief='sunken'
        )
        return placeholder
    
    def practice_specific_letter(self, letter):
        """Inicia modo práctica para una letra específica"""
        from tkinter import messagebox
        
        messagebox.showinfo(
            "Modo Práctica",
            f"Practica la letra '{letter}': {self.letter_descriptions[letter]}\n\n"
            f"Haz el gesto frente a la cámara y observa si se detecta correctamente."
        )
        
        # Enfocar la ventana principal
        self.parent.root.focus()
    
    def load_reference_images(self):
        """Carga imágenes de referencia desde una carpeta"""
        from tkinter import filedialog, messagebox
        
        # Crear carpeta de assets si no existe
        assets_path = os.path.join("assets", "reference_images")
        os.makedirs(assets_path, exist_ok=True)
        
        # Verificar qué imágenes ya existen
        existing_images = []
        missing_images = []
        
        from ..config.settings import Config
        
        for letter in Config.SUPPORTED_LETTERS:
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.gif']:
                if os.path.exists(os.path.join(assets_path, f"{letter}{ext}")):
                    existing_images.append(f"{letter}{ext}")
                    found = True
                    break
            if not found:
                missing_images.append(letter)
        
        # Mostrar estado actual
        status_msg = f"Estado actual de las imágenes:\n\n"
        status_msg += f"✅ Imágenes encontradas: {len(existing_images)}\n"
        status_msg += f"❌ Imágenes faltantes: {len(missing_images)}\n\n"
        
        if existing_images:
            status_msg += f"Existentes: {', '.join(existing_images[:10])}\n"
            if len(existing_images) > 10:
                status_msg += f"... y {len(existing_images) - 10} más\n"
        
        if missing_images:
            status_msg += f"\nFaltantes: {', '.join(missing_images)}\n"
        
        status_msg += f"\nRuta: {os.path.abspath(assets_path)}"
        
        messagebox.showinfo("Estado de Imágenes de Referencia", status_msg)
        
        # Preguntar si quiere abrir la carpeta
        if messagebox.askyesno("Abrir Carpeta", "¿Quieres abrir la carpeta para agregar más imágenes?"):
            import subprocess
            import platform
            
            try:
                if platform.system() == "Windows":
                    subprocess.Popen(f'explorer "{os.path.abspath(assets_path)}"')
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen(["open", os.path.abspath(assets_path)])
                else:  # Linux
                    subprocess.Popen(["xdg-open", os.path.abspath(assets_path)])
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo abrir la carpeta: {e}")
        
        # Ofrecer recargar la galería
        if messagebox.askyesno("Recargar", "¿Quieres recargar la galería para ver los cambios?"):
            self.close_gallery()
            self.show_gallery()
    
    def improve_detection(self):
        """Mejora la detección usando las imágenes de referencia"""
        from tkinter import messagebox
        
        messagebox.showinfo(
            "Mejorar Detección",
            "Esta función analizará las imágenes de referencia para:\n\n"
            "• Ajustar los parámetros de detección\n"
            "• Calibrar la sensibilidad por letra\n"
            "• Optimizar los algoritmos de clasificación\n\n"
            "Implementación en desarrollo..."
        )
    
    def close_gallery(self):
        """Cierra la galería"""
        if self.window:
            self.window.destroy()
            self.window = None