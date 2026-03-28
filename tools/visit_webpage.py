import re
from smolagents.tools import Tool


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = "Visits a webpage at the given url and reads its content as a markdown string."
    inputs = {"url": {"type": "string", "description": "The url of the webpage to visit."}}
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_initialized = True

    def forward(self, url: str) -> str:
        try:
            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException
            from smolagents.utils import truncate_content
        except ImportError as e:
            raise ImportError(
                "Instala `markdownify` y `requests`: pip install markdownify requests"
            ) from e

        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            markdown_content = markdownify(response.text).strip()
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)  # bug fix: re ya importado
            return truncate_content(markdown_content, 10000)
        except requests.exceptions.Timeout:
            return "La solicitud tardó demasiado. Intenta más tarde o verifica la URL."
        except RequestException as e:
            return f"Error al obtener la página: {str(e)}"
        except Exception as e:
            return f"Error inesperado: {str(e)}"
