from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
import math
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI


# ── Herramienta original del template ────────────────────────────────────
@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


# ── Herramienta A: Calculadora de áreas ──────────────────────────────────
@tool
def calculate_area(shape: str, dimension1: float, dimension2: float = 0.0) -> str:
    """Calculates the area of a geometric shape.
    Supported shapes: circle, rectangle, triangle, square.
    Args:
        shape: Name of the shape (circle, rectangle, triangle, square).
        dimension1: Primary dimension (radius, width, base, or side length).
        dimension2: Secondary dimension (height for rectangle/triangle; 0 otherwise).
    """
    shape = shape.strip().lower()
    if shape == "circle":
        area = math.pi * dimension1 ** 2
        return f"Area of circle (radius={dimension1}): {area:.4f} square units."
    elif shape == "rectangle":
        area = dimension1 * dimension2
        return f"Area of rectangle ({dimension1} x {dimension2}): {area:.4f} square units."
    elif shape == "triangle":
        area = 0.5 * dimension1 * dimension2
        return f"Area of triangle (base={dimension1}, height={dimension2}): {area:.4f} square units."
    elif shape == "square":
        area = dimension1 ** 2
        return f"Area of square (side={dimension1}): {area:.4f} square units."
    else:
        return f"Shape '{shape}' not supported. Use: circle, rectangle, triangle, square."


# ── Herramienta B: Tipo de cambio en vivo ────────────────────────────────
@tool
def get_exchange_rate(base_currency: str, target_currency: str) -> str:
    """Fetches the current exchange rate between two currencies.
    Common codes: USD, EUR, GBP, JPY, COP, MXN, BRL, CAD.
    Args:
        base_currency: ISO 4217 source currency code. Example: 'USD'
        target_currency: ISO 4217 target currency code. Example: 'COP'
    """
    base = base_currency.strip().upper()[:3]
    target = target_currency.strip().upper()[:3]
    try:
        r = requests.get(
            f"https://open.er-api.com/v6/latest/{base}",
            timeout=8
        )
        r.raise_for_status()
        data = r.json()
        if target not in data["rates"]:
            return f"Currency {target} not supported."
        rate = data["rates"][target]
        return f"Exchange rate: 1 {base} = {rate:.2f} {target}."
    except Exception as e:
        return f"Error: {str(e)}"


# ── FinalAnswerTool personalizado ─────────────────────────────────────────
class CustomFinalAnswerTool(FinalAnswerTool):
    """FinalAnswerTool con prefijo, firma y contador de caracteres."""
    AUTHOR_NAME = "Brayan Rayo"

    def forward(self, answer: str) -> str:
        char_count = len(answer)
        return (
            f"🤖 **Agente dice:**\n"
            f"{answer}\n\n"
            f"---\n"
            f"✍️ *Procesado por {self.AUTHOR_NAME}*\n"
            f"📊 *Longitud de la respuesta: {char_count} caracteres*"
        )


# ── Modelo ────────────────────────────────────────────────────────────────
model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)

# ── Herramienta de imágenes del Hub ───────────────────────────────────────
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# ── Prompts ───────────────────────────────────────────────────────────────
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# ── Agente ────────────────────────────────────────────────────────────────
agent = CodeAgent(
    model=model,
    tools=[
        image_generation_tool,
        get_current_time_in_timezone,
        calculate_area,          # ← nueva
        get_exchange_rate,       # ← nueva
        DuckDuckGoSearchTool(),
        CustomFinalAnswerTool(), # ← reemplaza el FinalAnswerTool original
    ],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()