# src/utils/sentence_bank.py
"""
Banco de oraciones completas organizadas por categorías
"""

class SentenceBank:
    def __init__(self):
        # Oraciones organizadas por categorías
        self.categories = {
            'saludos': {
                'name': '👋 Saludos',
                'icon': '👋',
                'sentences': [
                    'HOLA',
                    'HOLA COMO ESTAS',
                    'BUENOS DIAS',
                    'BUENOS DIAS A TODOS',
                    'BUENAS TARDES',
                    'BUENAS NOCHES',
                    'BIENVENIDO',
                    'BIENVENIDOS A TODOS',
                    'HOLA QUE TAL',
                    'HOLA COMO TE VA',
                    'QUE TENGAS BUEN DIA',
                    'QUE TENGAS BUENA TARDE',
                    'HASTA LUEGO',
                    'HASTA PRONTO',
                    'NOS VEMOS',
                    'NOS VEMOS MAÑANA',
                    'CHAO',
                    'ADIOS',
                    'QUE TE VAYA BIEN',
                    'CUIDATE MUCHO'
                ]
            },
            'cortesia': {
                'name': '🙏 Cortesía',
                'icon': '🙏',
                'sentences': [
                    'GRACIAS',
                    'MUCHAS GRACIAS',
                    'GRACIAS POR TODO',
                    'GRACIAS POR TU AYUDA',
                    'MIL GRACIAS',
                    'TE AGRADEZCO MUCHO',
                    'POR FAVOR',
                    'POR FAVOR AYUDAME',
                    'DISCULPA',
                    'DISCULPA LA MOLESTIA',
                    'PERDON',
                    'LO SIENTO',
                    'LO SIENTO MUCHO',
                    'DE NADA',
                    'CON PERMISO',
                    'PERDONA LA DEMORA',
                    'NO HAY PROBLEMA',
                    'ESTA BIEN',
                    'CLARO QUE SI',
                    'CON GUSTO'
                ]
            },
            'familia': {
                'name': '👨‍👩‍👧 Familia',
                'icon': '👨‍👩‍👧',
                'sentences': [
                    'MI MAMA',
                    'MI PAPA',
                    'MI MAMA ME AMA',
                    'AMO A MI MAMA',
                    'AMO A MI PAPA',
                    'MI HERMANO',
                    'MI HERMANA',
                    'MIS HERMANOS',
                    'MI FAMILIA',
                    'TE QUIERO',
                    'TE QUIERO MUCHO',
                    'TE AMO',
                    'TE AMO MAMA',
                    'TE AMO PAPA',
                    'MI HIJO',
                    'MI HIJA',
                    'MIS HIJOS',
                    'MI ABUELO',
                    'MI ABUELA',
                    'MIS ABUELOS'
                ]
            },
            'necesidades': {
                'name': '🆘 Necesidades',
                'icon': '🆘',
                'sentences': [
                    'NECESITO AYUDA',
                    'AYUDAME POR FAVOR',
                    'TENGO HAMBRE',
                    'TENGO SED',
                    'QUIERO AGUA',
                    'QUIERO COMIDA',
                    'VOY AL BAÑO',
                    'NECESITO IR AL BAÑO',
                    'ME DUELE',
                    'ME DUELE LA CABEZA',
                    'ME SIENTO MAL',
                    'ESTOY ENFERMO',
                    'NECESITO UN DOCTOR',
                    'NECESITO MEDICINA',
                    'TENGO FRIO',
                    'TENGO CALOR',
                    'ESTOY CANSADO',
                    'TENGO SUEÑO',
                    'QUIERO DESCANSAR',
                    'NECESITO DORMIR'
                ]
            },
            'emociones': {
                'name': '😊 Emociones',
                'icon': '😊',
                'sentences': [
                    'ESTOY FELIZ',
                    'ESTOY MUY CONTENTO',
                    'ME SIENTO BIEN',
                    'ME SIENTO MAL',
                    'ESTOY TRISTE',
                    'ESTOY ENOJADO',
                    'ESTOY PREOCUPADO',
                    'ESTOY NERVIOSO',
                    'TENGO MIEDO',
                    'ESTOY TRANQUILO',
                    'ESTOY EMOCIONADO',
                    'ME SIENTO SOLO',
                    'ME SIENTO ACOMPAÑADO',
                    'ESTOY AGRADECIDO',
                    'ESTOY ORGULLOSO',
                    'ME SIENTO CONFUNDIDO',
                    'ESTOY SORPRENDIDO',
                    'ME ALEGRA VERTE',
                    'ME DA GUSTO',
                    'QUE ALEGRIA'
                ]
            },
            'conversacion': {
                'name': '💬 Conversación',
                'icon': '💬',
                'sentences': [
                    'COMO TE LLAMAS',
                    'ME LLAMO',
                    'MUCHO GUSTO',
                    'ENCANTADO DE CONOCERTE',
                    'QUE EDAD TIENES',
                    'TENGO AÑOS',
                    'DE DONDE ERES',
                    'SOY DE',
                    'DONDE VIVES',
                    'VIVO EN',
                    'QUE HACES',
                    'TRABAJO EN',
                    'ESTUDIO EN',
                    'ENTIENDES',
                    'NO ENTIENDO',
                    'PUEDES REPETIR',
                    'HABLA MAS DESPACIO',
                    'ESTA CLARO',
                    'COMPRENDO',
                    'DE ACUERDO'
                ]
            },
            'preguntas': {
                'name': '❓ Preguntas',
                'icon': '❓',
                'sentences': [
                    'QUE',
                    'QUE ES ESO',
                    'QUE PASA',
                    'QUE HACES',
                    'COMO',
                    'COMO ESTAS',
                    'COMO SE HACE',
                    'CUANDO',
                    'CUANDO ES',
                    'CUANDO VAMOS',
                    'DONDE',
                    'DONDE ESTA',
                    'DONDE VAMOS',
                    'QUIEN',
                    'QUIEN ES',
                    'QUIEN VIENE',
                    'POR QUE',
                    'POR QUE NO',
                    'CUANTO CUESTA',
                    'QUE HORA ES'
                ]
            },
            'respuestas': {
                'name': '✅ Respuestas',
                'icon': '✅',
                'sentences': [
                    'SI',
                    'SI CLARO',
                    'SI POR SUPUESTO',
                    'NO',
                    'NO GRACIAS',
                    'NO PUEDO',
                    'TAL VEZ',
                    'QUIZAS',
                    'NO SE',
                    'NO ESTOY SEGURO',
                    'PUEDE SER',
                    'ESTA BIEN',
                    'PERFECTO',
                    'EXCELENTE',
                    'MUY BIEN',
                    'ASI ES',
                    'CORRECTO',
                    'INCORRECTO',
                    'ESO ES TODO',
                    'NADA MAS'
                ]
            },
            'tiempo': {
                'name': '⏰ Tiempo',
                'icon': '⏰',
                'sentences': [
                    'HOY',
                    'MAÑANA',
                    'AYER',
                    'AHORA',
                    'DESPUES',
                    'ANTES',
                    'MAS TARDE',
                    'MAS TEMPRANO',
                    'ESTA MAÑANA',
                    'ESTA TARDE',
                    'ESTA NOCHE',
                    'LA SEMANA PASADA',
                    'LA PROXIMA SEMANA',
                    'EL MES PASADO',
                    'EL PROXIMO MES',
                    'EL AÑO PASADO',
                    'EL PROXIMO AÑO',
                    'TODOS LOS DIAS',
                    'A VECES',
                    'NUNCA'
                ]
            },
            'lugares': {
                'name': '📍 Lugares',
                'icon': '📍',
                'sentences': [
                    'VOY A CASA',
                    'ESTOY EN CASA',
                    'VOY A LA ESCUELA',
                    'ESTOY EN LA ESCUELA',
                    'VOY AL TRABAJO',
                    'ESTOY EN EL TRABAJO',
                    'VOY AL HOSPITAL',
                    'VOY A LA TIENDA',
                    'VOY AL PARQUE',
                    'ESTOY EN EL PARQUE',
                    'VOY A LA IGLESIA',
                    'VOY AL DOCTOR',
                    'ESTOY EN LA CALLE',
                    'VOY AL CENTRO',
                    'QUIERO IR A',
                    'VAMOS A',
                    'ESTAMOS EN',
                    'VENGO DE',
                    'LLEGO DE',
                    'SALGO DE'
                ]
            }
        }
        
        # Contador de uso para estadísticas
        self.usage_count = {}
        
    def get_categories(self):
        """Retorna lista de categorías disponibles"""
        return list(self.categories.keys())
    
    def get_category_info(self, category):
        """Retorna información de una categoría"""
        return self.categories.get(category, {})
    
    def get_sentences(self, category):
        """Retorna oraciones de una categoría"""
        cat_data = self.categories.get(category, {})
        return cat_data.get('sentences', [])
    
    def get_all_sentences(self):
        """Retorna todas las oraciones de todas las categorías"""
        all_sentences = []
        for category in self.categories.values():
            all_sentences.extend(category['sentences'])
        return all_sentences
    
    def search_sentences(self, query, max_results=10):
        """
        Busca oraciones que contengan el texto dado
        """
        query = query.upper().strip()
        if not query:
            return []
        
        results = []
        for category_key, category in self.categories.items():
            for sentence in category['sentences']:
                if query in sentence:
                    results.append({
                        'sentence': sentence,
                        'category': category_key,
                        'category_name': category['name']
                    })
        
        return results[:max_results]
    
    def get_most_used(self, count=10):
        """Retorna las oraciones más usadas"""
        sorted_sentences = sorted(
            self.usage_count.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [s[0] for s in sorted_sentences[:count]]
    
    def register_usage(self, sentence):
        """Registra el uso de una oración"""
        sentence = sentence.upper().strip()
        if sentence in self.usage_count:
            self.usage_count[sentence] += 1
        else:
            self.usage_count[sentence] = 1
    
    def get_category_by_sentence(self, sentence):
        """Encuentra la categoría de una oración"""
        sentence = sentence.upper().strip()
        for category_key, category in self.categories.items():
            if sentence in category['sentences']:
                return category_key
        return None
    
    def add_custom_sentence(self, sentence, category='custom'):
        """Agrega una oración personalizada"""
        sentence = sentence.upper().strip()
        
        if category not in self.categories:
            self.categories[category] = {
                'name': '⭐ Personalizadas',
                'icon': '⭐',
                'sentences': []
            }
        
        if sentence not in self.categories[category]['sentences']:
            self.categories[category]['sentences'].append(sentence)
            return True
        
        return False
    
    def get_statistics(self):
        """Retorna estadísticas del banco de oraciones"""
        total_sentences = sum(
            len(cat['sentences']) 
            for cat in self.categories.values()
        )
        
        return {
            'total_categories': len(self.categories),
            'total_sentences': total_sentences,
            'total_used': len(self.usage_count),
            'most_used': self.get_most_used(5)
        }