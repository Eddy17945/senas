# src/utils/word_sentence_manager.py
"""
Gestor que separa la palabra actual de la oración completa
"""

class WordSentenceManager:
    def __init__(self):
        # Palabra que se está escribiendo ahora
        self.current_word = ""
        
        # Oración completa (palabras ya confirmadas)
        self.complete_sentence = ""
        
        # Historial de palabras confirmadas
        self.word_history = []
        
        # Historial de oraciones completas
        self.sentence_history = []
        
        # Buffer temporal para manejar gestos
        self.gesture_buffer = ""
        
    def add_letter(self, letter):
        """
        Agrega una letra a la palabra actual
        """
        letter = letter.upper().strip()
        if letter and len(letter) == 1:
            self.current_word += letter
            self.gesture_buffer = ""  # Limpiar buffer al agregar letra
            return True
        return False
    
    def add_gesture_to_buffer(self, gesture):
        """
        Agrega un gesto al buffer temporal (para palabras completas por gesto)
        """
        self.gesture_buffer = gesture.upper().strip()
        return True
    
    def confirm_gesture_word(self):
        """
        Confirma la palabra del buffer de gestos como palabra actual
        """
        if self.gesture_buffer:
            self.current_word = self.gesture_buffer
            self.gesture_buffer = ""
            return True
        return False
    
    def delete_last_letter(self):
        """
        Borra la última letra de la palabra actual o de la oración
        """
        if self.current_word:
            self.current_word = self.current_word[:-1]
            return True
        elif self.complete_sentence:
            # Si no hay palabra actual, borrar de la oración
            self.complete_sentence = self.complete_sentence[:-1]
            return True
        return False
    
    def add_space(self):
        """
        Confirma la palabra actual y la agrega a la oración completa
        """
        if self.current_word:
            # Agregar palabra al historial
            if self.current_word not in self.word_history:
                self.word_history.insert(0, self.current_word)
                # Mantener solo las últimas 10 palabras
                self.word_history = self.word_history[:10]
            
            # Agregar palabra a la oración
            if self.complete_sentence:
                self.complete_sentence += " " + self.current_word
            else:
                self.complete_sentence = self.current_word
            
            # Limpiar palabra actual
            self.current_word = ""
            return True
        return False
    
    def add_complete_sentence(self, text: str) -> bool:
        """
        Agrega una oración o palabra completa directamente
        
        MEJORADO: Ahora maneja tanto palabras únicas como oraciones completas
        
        Args:
            text: Texto completo a agregar (puede ser una palabra o varias)
            
        Returns:
            True si se agregó exitosamente
        """
        if not text:
            return False
        
        try:
            text = text.strip().upper()
            
            # Limpiar palabra actual si existe
            self.clear_current_word()
            
            # Verificar si es una sola palabra o múltiples
            words = text.split()
            
            if len(words) == 1:
                # Es una sola palabra - agregarla como palabra actual y confirmar
                word = words[0]
                
                # Agregar letra por letra
                for letter in word:
                    self.add_letter(letter)
                
                # Confirmar la palabra (agregar espacio)
                self.add_space()
                
                print(f"[DEBUG WordSentenceManager] Palabra única agregada: {word}")
                
            else:
                # Son múltiples palabras - agregar cada una
                for word in words:
                    # Agregar letra por letra
                    for letter in word:
                        self.add_letter(letter)
                    
                    # Confirmar la palabra
                    self.add_space()
                
                print(f"[DEBUG WordSentenceManager] Oración completa agregada: {text}")
            
            # Agregar al historial de oraciones si no es una palabra muy corta
            if len(text) > 3 and text not in self.sentence_history:
                self.sentence_history.insert(0, text)
                self.sentence_history = self.sentence_history[:10]
            
            return True
            
        except Exception as e:
            print(f"[ERROR WordSentenceManager] Error agregando texto completo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_word_by_gesture(self, word: str) -> bool:
        """
        NUEVO: Método específico para palabras detectadas por gesto
        Las agrega directamente a la oración
        
        Args:
            word: Palabra detectada por gesto (ej: "HOLA", "GRACIAS")
            
        Returns:
            True si se agregó exitosamente
        """
        if not word:
            return False
        
        try:
            word = word.strip().upper()
            
            # Limpiar palabra actual
            self.clear_current_word()
            
            # Agregar letra por letra a la palabra actual
            for letter in word:
                self.current_word += letter
            
            # Confirmar inmediatamente (agregar a oración completa)
            success = self.add_space()
            
            if success:
                print(f"[DEBUG WordSentenceManager] Palabra por gesto agregada: {word}")
            
            return success
            
        except Exception as e:
            print(f"[ERROR WordSentenceManager] Error agregando palabra por gesto: {e}")
            return False
    
    def clear_current_word(self):
        """
        Limpia solo la palabra actual
        """
        self.current_word = ""
        self.gesture_buffer = ""
        return True
    
    def clear_sentence(self):
        """
        Limpia la oración completa
        """
        self.complete_sentence = ""
        return True
    
    def clear_all(self):
        """
        Limpia todo (palabra actual y oración)
        """
        self.current_word = ""
        self.complete_sentence = ""
        self.gesture_buffer = ""
        return True
    
    def get_current_word(self):
        """
        Retorna la palabra actual (o el buffer si existe)
        """
        if self.gesture_buffer:
            return self.gesture_buffer
        return self.current_word
    
    def get_complete_sentence(self):
        """
        Retorna la oración completa
        """
        return self.complete_sentence
    
    def get_full_text(self):
        """
        Retorna el texto completo (oración + palabra actual)
        """
        if self.current_word:
            if self.complete_sentence:
                return f"{self.complete_sentence} {self.current_word}"
            return self.current_word
        return self.complete_sentence
    
    def get_word_history(self, max_count=10):
        """
        Retorna el historial de palabras usadas
        """
        return self.word_history[:max_count]
    
    def get_sentence_history(self, max_count=10):
        """
        Retorna el historial de oraciones usadas
        """
        return self.sentence_history[:max_count]
    
    def reuse_word(self, word):
        """
        Reutiliza una palabra del historial
        """
        word = word.upper().strip()
        if word:
            self.current_word = word
            return True
        return False
    
    def reuse_sentence(self, sentence):
        """
        Reutiliza una oración del historial
        """
        return self.add_complete_sentence(sentence)
    
    def get_statistics(self):
        """
        Retorna estadísticas del uso
        """
        return {
            'current_word_length': len(self.current_word),
            'sentence_length': len(self.complete_sentence),
            'sentence_word_count': len(self.complete_sentence.split()) if self.complete_sentence else 0,
            'total_words_used': len(self.word_history),
            'total_sentences_used': len(self.sentence_history)
        }
    
    def undo_last_word(self):
        """
        Deshace la última palabra agregada a la oración
        """
        if self.complete_sentence:
            words = self.complete_sentence.split()
            if words:
                # La última palabra vuelve a ser la palabra actual
                self.current_word = words[-1]
                # Recomponer la oración sin la última palabra
                self.complete_sentence = " ".join(words[:-1])
                return True
        return False
    
    def finalize_sentence(self):
        """
        Finaliza la oración actual (útil para TTS)
        Confirma cualquier palabra actual y retorna la oración completa
        """
        if self.current_word:
            self.add_space()
        
        sentence = self.complete_sentence
        
        if sentence:
            # Agregar al historial si no está
            if sentence not in self.sentence_history:
                self.sentence_history.insert(0, sentence)
                self.sentence_history = self.sentence_history[:10]
        
        return sentence
    
    def start_new_sentence(self):
        """
        Inicia una nueva oración (guarda la actual en historial y limpia)
        """
        sentence = self.finalize_sentence()
        self.clear_all()
        return sentence