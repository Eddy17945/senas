# src/utils/word_dictionary.py
"""
Diccionario de palabras comunes en español para el traductor de señas
"""

class WordDictionary:
    def __init__(self):
        # Palabras organizadas por categorías
        self.categories = {
            'saludos': [
                'HOLA', 'ADIOS', 'BUENOS DIAS', 'BUENAS TARDES', 
                'BUENAS NOCHES', 'BIENVENIDO', 'HASTA LUEGO', 
                'HASTA PRONTO', 'NOS VEMOS'
            ],
            'cortesia': [
                'GRACIAS', 'POR FAVOR', 'DISCULPA', 'PERDON',
                'DE NADA', 'CON PERMISO', 'MUCHAS GRACIAS',
                'LO SIENTO', 'DISCULPE'
            ],
            'familia': [
                'MAMA', 'PAPA', 'HERMANO', 'HERMANA', 'HIJO',
                'HIJA', 'ABUELO', 'ABUELA', 'TIO', 'TIA',
                'PRIMO', 'PRIMA', 'FAMILIA', 'PADRE', 'MADRE'
            ],
            'basicas': [
                'SI', 'NO', 'OK', 'BIEN', 'MAL', 'MUY', 'POCO',
                'MUCHO', 'NADA', 'TODO', 'ALGO', 'SIEMPRE',
                'NUNCA', 'AHORA', 'DESPUES', 'ANTES', 'HOY',
                'AYER', 'MAÑANA'
            ],
            'necesidades': [
                'AGUA', 'COMIDA', 'BAÑO', 'AYUDA', 'MEDICINA',
                'DOLOR', 'CANSADO', 'HAMBRE', 'SED', 'SUEÑO',
                'FRIO', 'CALOR', 'ENFERMO', 'DOCTOR'
            ],
            'preguntas': [
                'QUE', 'COMO', 'CUANDO', 'DONDE', 'QUIEN',
                'POR QUE', 'CUANTO', 'CUAL'
            ],
            'verbos': [
                'QUIERO', 'PUEDO', 'TENGO', 'SOY', 'ESTOY',
                'VOY', 'HAGO', 'NECESITO', 'QUIERO', 'AMO',
                'ME GUSTA', 'ENTIENDO', 'SE', 'VEO', 'OYO'
            ],
            'lugares': [
                'CASA', 'ESCUELA', 'TRABAJO', 'HOSPITAL',
                'TIENDA', 'CALLE', 'PARQUE', 'IGLESIA'
            ],
            'emociones': [
                'FELIZ', 'TRISTE', 'ENOJADO', 'CONTENTO',
                'PREOCUPADO', 'NERVIOSO', 'TRANQUILO', 'CANSADO'
            ]
        }
        
        # Crear lista completa de todas las palabras
        self.all_words = []
        for category_words in self.categories.values():
            self.all_words.extend(category_words)
        
        # Eliminar duplicados y ordenar
        self.all_words = sorted(list(set(self.all_words)))
        
        # Frases completas comunes
        self.common_phrases = [
            'HOLA COMO ESTAS',
            'MUY BIEN GRACIAS',
            'MI MAMA ME AMA',
            'BUENOS DIAS A TODOS',
            'GRACIAS POR TU AYUDA',
            'NECESITO AYUDA',
            'TENGO HAMBRE',
            'TENGO SED',
            'VOY AL BAÑO',
            'NO ENTIENDO',
            'COMO TE LLAMAS',
            'ME LLAMO',
            'MUCHO GUSTO',
            'HASTA LUEGO',
            'NOS VEMOS MAÑANA',
            'QUE TENGAS BUEN DIA',
            'FELIZ CUMPLEAÑOS',
            'TE QUIERO',
            'TE AMO',
            'LO SIENTO MUCHO'
        ]
        
        # Palabras rápidas (más usadas)
        self.quick_words = [
            'HOLA', 'GRACIAS', 'SI', 'NO', 'AYUDA',
            'AGUA', 'MAMA', 'PAPA', 'POR FAVOR', 'ADIOS'
        ]
        
        # Correcciones ortográficas comunes
        self.corrections = {
            'OLA': 'HOLA',
            'GRASIAS': 'GRACIAS',
            'GRAXIAS': 'GRACIAS',
            'MAMÁ': 'MAMA',
            'PAPÁ': 'PAPA',
            'HERMANO': 'HERMANO',
            'K': 'QUE',
            'Q': 'QUE',
            'XQ': 'POR QUE',
            'PORQ': 'POR QUE',
            'TBN': 'TAMBIEN',
            'TMB': 'TAMBIEN',
            'BN': 'BIEN',
            'ML': 'MAL'
        }
    
    def get_all_words(self):
        """Retorna todas las palabras del diccionario"""
        return self.all_words.copy()
    
    def get_category_words(self, category):
        """Retorna palabras de una categoría específica"""
        return self.categories.get(category, [])
    
    def get_quick_words(self):
        """Retorna palabras de acceso rápido"""
        return self.quick_words.copy()
    
    def get_common_phrases(self):
        """Retorna frases comunes completas"""
        return self.common_phrases.copy()
    
    def search_words(self, prefix, max_results=5):
        """
        Busca palabras que empiezan con el prefijo dado
        
        Args:
            prefix: Texto a buscar
            max_results: Máximo de resultados a retornar
            
        Returns:
            Lista de palabras que coinciden
        """
        if not prefix:
            return []
        
        prefix = prefix.upper().strip()
        
        # Buscar palabras que empiezan con el prefijo
        matches = [word for word in self.all_words 
                  if word.startswith(prefix)]
        
        # Ordenar por longitud (palabras más cortas primero)
        matches.sort(key=len)
        
        return matches[:max_results]
    
    def suggest_correction(self, word):
        """
        Sugiere corrección ortográfica si existe
        
        Args:
            word: Palabra a corregir
            
        Returns:
            Palabra corregida o None
        """
        word = word.upper().strip()
        return self.corrections.get(word)
    
    def is_valid_word(self, word):
        """Verifica si una palabra está en el diccionario"""
        return word.upper().strip() in self.all_words
    
    def get_word_frequency(self, word):
        """
        Retorna frecuencia de uso estimada (1-10)
        Palabras más comunes tienen mayor frecuencia
        """
        word = word.upper().strip()
        
        if word in self.quick_words:
            return 10
        elif word in self.categories['saludos']:
            return 9
        elif word in self.categories['cortesia']:
            return 9
        elif word in self.categories['basicas']:
            return 8
        elif word in self.categories['familia']:
            return 7
        elif word in self.categories['necesidades']:
            return 8
        else:
            return 5
    
    def get_similar_words(self, word, max_results=3):
        """
        Encuentra palabras similares (para corrección)
        Usa distancia de Levenshtein simplificada
        """
        if not word:
            return []
        
        word = word.upper().strip()
        
        # Si la palabra está correcta, retornar vacío
        if word in self.all_words:
            return []
        
        similar = []
        
        for dict_word in self.all_words:
            # Calcular similitud simple
            similarity = self._simple_similarity(word, dict_word)
            
            if similarity > 0.6:  # 60% de similitud
                similar.append((dict_word, similarity))
        
        # Ordenar por similitud
        similar.sort(key=lambda x: x[1], reverse=True)
        
        return [w[0] for w in similar[:max_results]]
    
    def _simple_similarity(self, word1, word2):
        """
        Calcula similitud simple entre dos palabras
        Retorna valor entre 0 y 1
        """
        if not word1 or not word2:
            return 0.0
        
        # Si son iguales
        if word1 == word2:
            return 1.0
        
        # Si una contiene a la otra
        if word1 in word2 or word2 in word1:
            return 0.8
        
        # Contar caracteres en común
        set1 = set(word1)
        set2 = set(word2)
        common = len(set1.intersection(set2))
        total = len(set1.union(set2))
        
        if total == 0:
            return 0.0
        
        return common / total
    
    def add_custom_word(self, word, category='custom'):
        """Agrega una palabra personalizada al diccionario"""
        word = word.upper().strip()
        
        if word and word not in self.all_words:
            self.all_words.append(word)
            self.all_words.sort()
            
            if category not in self.categories:
                self.categories[category] = []
            
            if word not in self.categories[category]:
                self.categories[category].append(word)
            
            return True
        
        return False
    
    def get_statistics(self):
        """Retorna estadísticas del diccionario"""
        return {
            'total_words': len(self.all_words),
            'total_phrases': len(self.common_phrases),
            'quick_words': len(self.quick_words),
            'categories': len(self.categories),
            'corrections': len(self.corrections)
        }