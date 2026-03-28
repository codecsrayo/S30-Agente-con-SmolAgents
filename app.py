# ============================================================
# Agentic RAG — Turismo Cafetero Inteligente
# Curso HuggingFace Agents · Tarea Final
# ============================================================

import math
import yaml

from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, tool, Tool
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI


# ============================================================
# PARTE 2 · Base de conocimiento interna (≥ 8 documentos)
# ============================================================

_RAW_DOCS = [
    # ── Fincas en Huila ─────────────────────────────────────
    {
        "text": (
            "Finca La Siberia está en Pitalito, Huila, a 2.300 msnm. "
            "Ofrece tours de 4 horas con proceso honey: los granos se despulpan "
            "pero se secan con la miel del mucílago adherida, generando dulzor y "
            "notas de fruta tropical. Precio: $80.000 COP por persona. "
            "Incluye cata de 3 cafés especiales. Capacidad: 12 personas. "
            "Contacto: fincalasiberia@gmail.com."
        ),
        "source": "base_interna",
        "region": "Huila",
    },
    {
        "text": (
            "Finca El Paraíso está en San Agustín, Huila, a 1.800 msnm. "
            "Reconocida por su proceso natural: el fruto entero se seca al sol "
            "20–30 días antes de despulpar, generando sabores frutales intensos "
            "y notas de moras y ciruela. Tour de día completo: $120.000 COP. "
            "Incluye almuerzo típico y senderos ecológicos. "
            "Temporada alta: abril–junio y octubre–diciembre."
        ),
        "source": "base_interna",
        "region": "Huila",
    },
    {
        "text": (
            "Finca Las Acacias en La Plata, Huila, especializada en proceso lavado. "
            "El proceso lavado consiste en despulpar, fermentar en tanques 24–48 horas "
            "y lavar con agua limpia, produciendo perfiles limpios y cítricos. "
            "Tour de 3 horas: $65.000 COP. Grupos hasta 20 personas. "
            "Accesible desde Neiva en 2.5 horas por carretera."
        ),
        "source": "base_interna",
        "region": "Huila",
    },
    # ── Finca en Nariño ─────────────────────────────────────
    {
        "text": (
            "Finca La Esmeralda en Buesaco, Nariño, a 1.950 msnm. "
            "Famosa por su café lavado de perfil brillante y acidez cítrica. "
            "Nariño tiene dos cosechas: principal (abril–julio) y traviesa (octubre–enero). "
            "Tour con cupping profesional: $90.000 COP. Variedades Castillo y Caturra. "
            "Premiada en la Taza de la Excelencia 2022."
        ),
        "source": "base_interna",
        "region": "Nariño",
    },
    # ── Fincas en Eje Cafetero ───────────────────────────────
    {
        "text": (
            "Finca El Ocaso en Salento, Quindío, a 1.700 msnm. "
            "Ofrece procesos honey y natural. Tour estándar: $75.000 COP. "
            "Tour avanzado con laboratorio de tueste: $150.000 COP. "
            "Hospedaje en cabaña: $200.000 COP por noche doble. "
            "A 15 minutos de Salento a pie o en Willys."
        ),
        "source": "base_interna",
        "region": "Quindío",
    },
    {
        "text": (
            "Finca Villarazo cerca de Armenia, Quindío. Proyecto familiar con variedad "
            "Geisha proceso honey de alta calidad. Tour: $85.000 COP, incluye desayuno "
            "cafetero típico. Temporada de cosecha: octubre–enero (principal). "
            "Máximo 8 personas por grupo. Contacto: +57 310 555 1234."
        ),
        "source": "base_interna",
        "region": "Quindío",
    },
    # ── Finca en Cauca ───────────────────────────────────────
    {
        "text": (
            "Finca Café del Macizo en Inzá, Cauca, a 1.900 msnm. "
            "Proceso natural y honey experimental. Cosecha principal octubre–enero. "
            "Tour de 6 horas: $100.000 COP, incluye recorrido por bosque nativo. "
            "Café Inzá tiene Denominación de Origen. "
            "Se puede combinar con el Parque Arqueológico de Tierradentro (30 min)."
        ),
        "source": "base_interna",
        "region": "Cauca",
    },
    # ── Temporadas ──────────────────────────────────────────
    {
        "text": (
            "Temporadas de cosecha por región: "
            "Huila — principal: octubre–diciembre, traviesa: abril–junio. "
            "Nariño — principal: abril–julio, traviesa: octubre–enero. "
            "Eje Cafetero (Caldas, Quindío, Risaralda) — principal: octubre–enero, "
            "mitaca: abril–junio. Cauca — principal: octubre–enero. "
            "La temporada alta turística coincide con la cosecha principal (octubre–enero)."
        ),
        "source": "base_interna",
        "region": "General",
    },
    # ── Precios ─────────────────────────────────────────────
    {
        "text": (
            "Precios de tours cafeteros en Colombia (2024–2025): "
            "Tour básico 2–3 h: $50.000–$80.000 COP. "
            "Tour completo con almuerzo 5–6 h: $90.000–$130.000 COP. "
            "Tour con cupping profesional: $120.000–$180.000 COP. "
            "Hospedaje en finca por noche/persona: $80.000–$250.000 COP. "
            "Paquetes de 3 días todo incluido: $600.000–$1.200.000 COP."
        ),
        "source": "base_interna",
        "region": "General",
    },
    # ── Procesos de beneficio ────────────────────────────────
    {
        "text": (
            "Procesos de beneficio del café de especialidad: "
            "LAVADO: fermentación en agua, perfil limpio, notas cítricas y florales. "
            "HONEY: secado con miel adherida, balance entre lavado y natural, notas dulces. "
            "NATURAL: secado del fruto entero, sabores frutales intensos, "
            "notas de moras, ciruela y chocolate. "
            "ANAERÓBICO: fermentación sin oxígeno, perfiles exóticos y únicos."
        ),
        "source": "base_interna",
        "region": "General",
    },
]


def _build_knowledge_base() -> list[Document]:
    """Procesa los documentos crudos con RecursiveCharacterTextSplitter."""
    docs = [
        Document(
            page_content=d["text"],
            metadata={"source": d["source"], "region": d["region"]},
        )
        for d in _RAW_DOCS
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


class CoffeeRouteRetrieverTool(Tool):
    """
    Parte 2: Herramienta de retrieval con BM25 sobre la base de conocimiento
    interna de turismo cafetero colombiano.

    Se implementa como subclase de Tool (en vez de @tool) porque necesita
    inicializar y almacenar el BM25Retriever como estado de instancia,
    lo que no es posible con el decorador.
    """

    name = "coffee_route_retriever"
    description = (
        "Busca información en la base de conocimiento interna sobre turismo "
        "cafetero en Colombia: fincas específicas, procesos de café (lavado, "
        "honey, natural, anaeróbico), temporadas de cosecha, precios de tours "
        "y características de regiones (Huila, Nariño, Quindío, Cauca). "
        "Úsalo ANTES de buscar en internet cuando la pregunta sea sobre "
        "fincas concretas, precios orientativos o procesos de beneficio."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "Consulta en lenguaje natural sobre turismo cafetero, "
                "fincas, procesos o regiones cafeteras de Colombia."
            ),
        }
    }
    output_type = "string"

    def __init__(self, docs: list[Document], **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=5)

    def forward(self, query: str) -> str:
        results = self.retriever.invoke(query)
        if not results:
            return "No se encontró información relevante en la base de conocimiento interna."
        parts = []
        for i, doc in enumerate(results, 1):
            region = doc.metadata.get("region", "N/A")
            parts.append(f"[Resultado {i} — Región: {region}]\n{doc.page_content}")
        return "\n\n".join(parts)


# ============================================================
# PARTE 3 · Tool de cálculo: tiempo de viaje con Haversine
# ============================================================

@tool
def calculate_road_travel_time(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    speed_kmh: float = 45.0,
) -> str:
    """Calcula la distancia y el tiempo de viaje por carretera entre dos puntos
    en Colombia usando la fórmula de Haversine con factor de corrección 1.6
    para carreteras de montaña. Velocidad por defecto: 45 km/h.

    Usar este tool es preferible a pedirle al LLM el cálculo directamente,
    porque garantiza precisión matemática reproducible y evita alucinaciones.

    Coordenadas útiles:
      Neiva, Huila:        lat=2.9273, lon=-75.2819
      Pitalito, Huila:     lat=1.8547, lon=-76.0486
      San Agustín, Huila:  lat=1.8833, lon=-76.2833
      Inzá, Cauca:         lat=2.5547, lon=-76.0656
      Buesaco, Nariño:     lat=1.3833, lon=-77.1500
      Armenia, Quindío:    lat=4.5339, lon=-75.6811
      Manizales, Caldas:   lat=5.0689, lon=-75.5174
      Santa Marta:         lat=11.2404, lon=-74.1990

    Args:
        origin_lat: Latitud del origen en grados decimales.
        origin_lon: Longitud del origen en grados decimales.
        dest_lat: Latitud del destino en grados decimales.
        dest_lon: Longitud del destino en grados decimales.
        speed_kmh: Velocidad promedio en km/h (default 45 para montaña colombiana).
    """
    R = 6371.0  # Radio de la Tierra en km

    lat1 = math.radians(origin_lat)
    lon1 = math.radians(origin_lon)
    lat2 = math.radians(dest_lat)
    lon2 = math.radians(dest_lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    straight_km = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Factor de corrección 1.6 para carreteras de montaña colombianas
    road_km = straight_km * 1.6
    total_hours = road_km / speed_kmh
    hours = int(total_hours)
    minutes = int((total_hours - hours) * 60)

    return (
        f"Distancia en línea recta: {straight_km:.1f} km | "
        f"Distancia estimada por carretera: {road_km:.1f} km | "
        f"Tiempo estimado a {speed_kmh:.0f} km/h: {hours}h {minutes}min"
    )


# ============================================================
# Modelo LLM
# ============================================================

model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    custom_role_conversions=None,
)

# ============================================================
# Prompts
# ============================================================

with open("prompts.yaml", "r") as f:
    prompt_templates = yaml.safe_load(f)

# ============================================================
# PARTE 4 · Agente combinado (RAG + Web + Haversine)
# ============================================================

processed_docs = _build_knowledge_base()
coffee_retriever = CoffeeRouteRetrieverTool(docs=processed_docs)

agent = CodeAgent(
    model=model,
    tools=[
        coffee_retriever,            # Parte 2: base de conocimiento interna
        DuckDuckGoSearchTool(),       # Parte 1 & 4: búsqueda web en tiempo real
        calculate_road_travel_time,   # Parte 3: cálculo Haversine
        FinalAnswerTool(),
    ],
    max_steps=15,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,          # Cambiar a 3 para activar el bonus
    name=None,
    description=None,
    prompt_templates=prompt_templates,
    additional_authorized_imports=["pandas"],
)

GradioUI(agent).launch()
