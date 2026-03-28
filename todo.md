Tarea Final: Sistema Agentic RAG con smolagents
Turismo cafetero inteligente — Construyendo un agente de recomendación con retrieval aumentado

## Contexto
Colombia es el tercer productor mundial de café y sus regiones cafeteras (Huila, Nariño, Eje Cafetero, Cauca, Sierra Nevada) atraen cada año a miles de turistas interesados en conocer el proceso del café de especialidad. Sin embargo, la información sobre fincas, rutas, precios y temporadas está dispersa en páginas web, guías locales y conocimiento de los propios caficultores.

En este taller van a construir un agente inteligente capaz de planificar rutas de turismo cafetero combinando búsqueda web en tiempo real, una base de conocimiento interna y herramientas de cálculo propias. El sistema debe demostrar los principios de Agentic RAG (Retrieval-Augmented Generation) vistos en el curso.

## Objetivos de aprendizaje
Al completar este taller, el estudiante será capaz de:

1. Implementar un agente CodeAgent de smolagents que utilice múltiples herramientas.
2. Construir un Tool como clase de Python (subclase de Tool) con retriever interno, siguiendo el patrón del curso.
3. Crear un Tool con el decorador @tool para cálculos personalizados.
4. Diseñar una base de conocimiento usando Document de LangChain, RecursiveCharacterTextSplitter y BM25Retriever.
5. Combinar búsqueda web (DuckDuckGoSearchTool) con retrieval local en un solo agente.
6. Explicar las ventajas de Agentic RAG frente a RAG tradicional.

## Descripción del problema
Una agencia de turismo en Neiva, Huila, quiere ofrecer a sus clientes un asistente virtual que les ayude a planificar rutas de café de especialidad. El asistente debe ser capaz de:

* Responder preguntas sobre fincas cafeteras, procesos de café, temporadas de cosecha y precios usando información interna de la agencia (base de conocimiento).
* Complementar esa información con búsqueda web cuando el usuario pregunte algo que no está en la base interna (hospedaje actualizado, vuelos, noticias recientes).
* Calcular tiempos de viaje por carretera entre destinos cafeteros usando coordenadas geográficas y la fórmula de Haversine, con un factor de corrección para carreteras de montaña.
* Integrar todo en un agente que decida autónomamente qué herramienta usar según la consulta del usuario.

## 1. Búsqueda web básica 20%
Cree un agente que use DuckDuckGoSearchTool para buscar información sobre fincas cafeteras abiertas al turismo en Huila. El agente debe recibir una consulta en lenguaje natural y devolver una respuesta sintetizada.

### Requisitos
* Usar CodeAgent con InferenceClientModel.
* La consulta debe pedir información específica: fincas, experiencias de cata, y cómo llegar desde Neiva.
* Ejecutar el agente y mostrar la respuesta completa.

### Pregunta para el informe
- ¿Qué limitaciones tiene este enfoque cuando la información que necesita el usuario es muy específica o no está en internet?

## 2. Base de conocimiento con retriever 30%
Construya una base de conocimiento interna sobre turismo cafetero y un Tool que la consulte usando BM25.

### Requisitos
* Crear mínimo 8 documentos en la base de conocimiento. Cada documento debe tener campos text, source y region. Los documentos deben cubrir al menos: fincas (mínimo 3 regiones diferentes), procesos de café (lavado, honey, natural), temporadas de cosecha, y precios.
* Usar Document de LangChain para envolver los datos.
* Usar RecursiveCharacterTextSplitter con chunk_size=500 y chunk_overlap=50.
* Crear una clase CoffeeRouteRetrieverTool que herede de smolagents.Tool. La clase debe definir los atributos name, description, inputs y output_type como atributos de clase, e implementar el método forward().
* El __init__ debe recibir los documentos procesados y crear un BM25Retriever con k=5.
* Crear un CodeAgent que use únicamente este tool y ejecutar una consulta sobre fincas en Huila con procesos honey.

### Estructura esperada de la clase (referencia)
```python
class CoffeeRouteRetrieverTool(Tool):
    name = "coffee_route_retriever"
    description = "Busca información sobre rutas de turismo cafetero..."
    inputs = {
        "query": {
            "type": "string",
            "description": "La consulta a realizar...",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=5)

    def forward(self, query: str) -> str:
        # Implementar la búsqueda y formateo de resultados
        ...
```

### Preguntas para el informe
* ¿Por qué se usa una subclase de Tool en vez del decorador @tool para este caso?
* ¿Qué papel juega el docstring de la descripción del tool en el comportamiento del agente?
* ¿Qué pasaría si cambia k=5 por k=2? Pruébelo y compare los resultados.

## 3. Tool de cálculo personalizado 20%
Cree un tool con el decorador @tool que calcule el tiempo de viaje por carretera entre dos puntos en Colombia.

### Requisitos
* Usar la fórmula de Haversine para calcular la distancia en línea recta.
* Aplicar un factor de corrección de 1.6 para simular carreteras de montaña colombianas.
* Velocidad promedio por defecto: 45 km/h (carreteras de montaña).
* El tool debe recibir tuplas (latitud, longitud) para origen y destino.
* Retornar un string con la distancia estimada y el tiempo de viaje.
* Verificar que funciona calculando el tiempo de Neiva a Pitalito.

### Pregunta para el informe
* ¿Por qué es mejor que el agente use este tool en vez de pedirle al LLM que haga el cálculo directamente?


## 4. Agente combinado 30%
Combine las tres partes anteriores en un solo agente que use las tres herramientas simultáneamente.

### Requisitos
* El agente debe tener: CoffeeRouteRetrieverTool, DuckDuckGoSearchTool y calculate_road_travel_time.
* Agregar "pandas" a additional_authorized_imports.
* Configurar max_steps=15.
* Ejecutar la siguiente tarea (o una equivalente que usted diseñe):

    >> Soy un turista en Neiva, Huila. Quiero hacer una ruta de café de especialidad de 3 días visitando fincas en Huila y regiones cercanas. Para cada destino necesito: nombre de la finca, actividades disponibles, precio del tour, y tiempo de viaje por carretera desde Neiva. Busca también opciones de hospedaje actualizadas cerca de Pitalito. Organiza todo en un DataFrame de pandas.

* El agente debe usar autónomamente las tres herramientas según lo que necesite.
* Bonus · +10%
* Configure agent.planning_interval = 3 y ejecute de nuevo la misma tarea. Compare los logs de ambas ejecuciones y explique en su informe cómo el planning periódico afecta el comportamiento del agente: ¿cambia el número de pasos?, ¿mejora la calidad?, ¿usa las herramientas de forma diferente?

## Especificaciones técnicas
### Entorno
* Python 3.10+, Miniconda recomendado.
* Paquetes: smolagents, langchain-community, langchain-text-splitters, python-dotenv, pandas.
* Token de HuggingFace con permisos de inferencia (HF_TOKEN).
* Modelo: Qwen/Qwen2.5-Coder-32B-Instruct vía InferenceClientModel() sin especificar provider.
### Entrega
* Un archivo .py o notebook .ipynb con el código completo y ejecutable.
* Un informe breve (1-2 páginas) respondiendo las preguntas de cada parte.
* Capturas de pantalla o logs de la ejecución mostrando el comportamiento del agente.

Criterios de evaluación
Criterio	Peso
### Criterios de evaluación
* Parte 1 funciona y responde coherentemente	20%
* Parte 2: Tool como clase correctamente implementado, KB con mínimo 8 documentos	30%
* Parte 3: Haversine implementado correctamente, tool funcional	20%
* Parte 4: Agente combinado ejecuta la tarea usando las 3 herramientas	30%
* Bonus Comparación con planning_interval	+10%
* Calidad del informe y respuestas a las preguntas de reflexión	Incluido en cada sección

### Coordenadas útiles para la Parte 3
| Lugar | Latitud | Longitud |
|-------|---------|----------|
| Neiva, Huila | 2.9273 | -75.2819 |
| Pitalito, Huila | 1.8547 | -76.0486 |
| San Agustín, Huila | 1.8833 | -76.2833 |
| Inzá, Cauca | 2.5547 | -76.0656 |
| Buesaco, Nariño | 1.3833 | -77.1500 |
| Armenia, Quindío | 4.5339 | -75.6811 |
| Manizales, Caldas | 5.0689 | -75.5174 |
| Santa Marta, Magdalena | 11.2404 | -74.1990 |

## Recursos
* [HF Agents Course — Retrieval Agents](https://huggingface.co/learn/agents-course/en/unit2/smolagents/retrieval_agents) · Material base de este taller Links to an external site. 
* [Documentación smolagents](https://huggingface.co/docs/smolagents/en/tutorials/tools) — Tools · Referencia para implementación de tools Links to an external site. 
* [Agentic RAG Cookbook](https://huggingface.co/learn/cookbook/agent_rag) · Receta completa de RAG agéntico con smolagents Links to an external site. 
* [Fórmula de Haversine](https://en.wikipedia.org/wiki/Haversine_formula) · Referencia matemática para el cálculo de distancias