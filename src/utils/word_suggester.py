# src/utils/word_suggester.py
"""
Sistema inteligente de sugerencias de palabras
"""

from typing import List, Optional, Dict, Tuple
from .word_dictionary import WordDictionary

class WordSuggester:
    def __init__(self):
        self.dictionary = WordDictionary()
        self.current_word = ""
        self.current_suggestions = []
        self.word_history = []
        self.max_suggestions = 5
        
        # Configuración
        self.auto_suggest_enabled = True
        self.min_letters_for_suggest = 2
        
    def update_current_word(self, text: str) -> List[str]:
        """
        Actualiza la palabra actual y genera sugerencias
        
        Args:
            text: Texto completo actual
            
        Returns:
            Lista de sugerencias
        """
        # Extraer última palabra (después del último espacio)
        words = text.strip().split()
        
        if words:
            self.current_word = words[-1].upper()
        else:
            self.current_word = ""
        
        # Generar sugerencias
        if len(self.current_word) >= self.min_letters_for_suggest:
            self.current_suggestions = self._generate_suggestions()
        else:
            self.current_suggestions = []
        
        return self.current_suggestions
    
    def _generate_suggestions(self) -> List[str]:
        """
        Genera sugerencias inteligentes basadas en palabra actual
        """
        if not self.current_word:
            return []
        
        suggestions = []
        
        # 1. Buscar palabras que empiezan con el prefijo
        prefix_matches = self.dictionary.search_words(
            self.current_word, 
            max_results=self.max_suggestions
        )
        suggestions.extend(prefix_matches)
        
        # 2. Si la palabra es corta y no hay coincidencias exactas,
        #    buscar palabras similares
        if len(suggestions) < self.max_suggestions and len(self.current_word) >= 3:
            similar = self.dictionary.get_similar_words(
                self.current_word, 
                max_results=self.max_suggestions - len(suggestions)
            )
            
            # Agregar solo las que no están ya
            for word in similar:
                if word not in suggestions:
                    suggestions.append(word)
        
        # 3. Verificar si hay corrección ortográfica sugerida
        correction = self.dictionary.suggest_correction(self.current_word)
        if correction and correction not in suggestions:
            # Agregar al inicio si es una corrección
            suggestions.insert(0, correction)
        
        # 4. Ordenar por relevancia (frecuencia de uso)
        suggestions.sort(
            key=lambda w: self.dictionary.get_word_frequency(w),
            reverse=True
        )
        
        return suggestions[:self.max_suggestions]
    
    def get_quick_suggestions(self) -> List[str]:
        """
        Retorna palabras de acceso rápido
        (para botones siempre visibles)
        """
        return self.dictionary.get_quick_words()
    
    def get_phrase_suggestions(self) -> List[str]:
        """
        Retorna frases completas comunes
        """
        return self.dictionary.get_common_phrases()
    
    def complete_word(self, suggestion: str, full_text: str) -> str:
        """
        Completa la palabra actual con una sugerencia
        
        Args:
            suggestion: Palabra sugerida seleccionada
            full_text: Texto completo actual
            
        Returns:
            Nuevo texto con la palabra completada
        """
        words = full_text.strip().split()
        
        if words:
            # Reemplazar última palabra
            words[-1] = suggestion
        else:
            # Si no hay palabras, agregar la sugerencia
            words = [suggestion]
        
        # Agregar a historial
        self.word_history.append(suggestion)
        
        # Mantener solo últimas 20 palabras en historial
        if len(self.word_history) > 20:
            self.word_history = self.word_history[-20:]
        
        return ' '.join(words) + ' '
    
    def add_phrase(self, phrase: str, full_text: str) -> str:
        """
        Agrega una frase completa al texto
        
        Args:
            phrase: Frase a agregar
            full_text: Texto actual
            
        Returns:
            Nuevo texto con la frase agregada
        """
        if full_text.strip():
            return full_text.strip() + ' ' + phrase + ' '
        else:
            return phrase + ' '
    
    def get_word_info(self, word: str) -> Dict:
        """
        Obtiene información sobre una palabra
        """
        word = word.upper().strip()
        
        info = {
            'word': word,
            'is_valid': self.dictionary.is_valid_word(word),
            'frequency': self.dictionary.get_word_frequency(word),
            'correction': self.dictionary.suggest_correction(word),
            'similar': self.dictionary.get_similar_words(word, max_results=3)
        }
        
        return info
    
    def get_next_word_predictions(self, full_text: str) -> List[str]:
        """
        Predice posibles siguientes palabras basado en contexto
        (Versión simple basada en patrones comunes)
        """
        words = full_text.strip().upper().split()
        
        if not words:
            # Si no hay texto, sugerir saludos
            return ['HOLA', 'BUENOS DIAS', 'BUENAS TARDES']
        
        last_word = words[-1]
        
        # Patrones comunes
        patterns = {
            'HOLA': ['COMO', 'QUE', 'BUENOS'],
            'BUENOS': ['DIAS', 'TARDES'],
            'BUENAS': ['TARDES', 'NOCHES'],
            'COMO': ['ESTAS', 'TE', 'SE'],
            'QUE': ['TAL', 'HACES', 'PASA'],
            'MUCHAS': ['GRACIAS'],
            'POR': ['FAVOR', 'QUE'],
            'TENGO': ['HAMBRE', 'SED', 'SUEÑO'],
            'NECESITO': ['AYUDA', 'AGUA', 'COMIDA'],
            'MI': ['MAMA', 'PAPA', 'HERMANO'],
            'ME': ['LLAMO', 'GUSTA', 'DUELE']
        }
        
        predictions = patterns.get(last_word, [])
        
        # Si no hay predicciones específicas, sugerir palabras comunes
        if not predictions:
            predictions = ['Y', 'PERO', 'GRACIAS', 'POR FAVOR']
        
        return predictions[:3]
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analiza el texto completo y retorna estadísticas
        """
        words = text.strip().split()
        
        analysis = {
            'word_count': len(words),
            'valid_words': 0,
            'invalid_words': [],
            'suggestions': {}
        }
        
        for word in words:
            word = word.upper().strip()
            if self.dictionary.is_valid_word(word):
                analysis['valid_words'] += 1
            else:
                analysis['invalid_words'].append(word)
                
                # Sugerir corrección
                correction = self.dictionary.suggest_correction(word)
                if correction:
                    analysis['suggestions'][word] = correction
        
        return analysis
    
    def get_recent_words(self, count: int = 5) -> List[str]:
        """Retorna las últimas palabras usadas"""
        return self.word_history[-count:] if self.word_history else []
    
    def clear_history(self):
        """Limpia el historial de palabras"""
        self.word_history.clear()
        self.current_word = ""
        self.current_suggestions = []
    
    def enable_auto_suggest(self, enabled: bool):
        """Activa/desactiva sugerencias automáticas"""
        self.auto_suggest_enabled = enabled
    
    def set_min_letters(self, min_letters: int):
        """Configura mínimo de letras para sugerir"""
        self.min_letters_for_suggest = max(1, min(5, min_letters))