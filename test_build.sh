#!/bin/bash
# ============================================================
# test_build.sh — Simula la compilación de HF Spaces localmente
# Uso: bash test_build.sh
# ============================================================

set -e  # parar si cualquier comando falla
REPO_DIR="$(pwd)"
VENV_DIR="$REPO_DIR/.venv_test"

echo "=============================================="
echo " Test de compilación local — HF Spaces"
echo " Directorio: $REPO_DIR"
echo "=============================================="
echo ""

# ── 1. Verificar que estamos en el repo correcto ───────────────
if [ ! -f "app.py" ] || [ ! -f "requirements.txt" ]; then
    echo "❌ ERROR: Ejecuta este script desde la raíz del repo"
    echo "   (donde están app.py y requirements.txt)"
    exit 1
fi
echo "✅ Directorio correcto"

# ── 2. Crear entorno virtual limpio ───────────────────────────
echo ""
echo "── Creando entorno virtual limpio..."
python3 -m venv "$VENV_DIR" --clear
source "$VENV_DIR/bin/activate"
echo "✅ Entorno virtual: $VENV_DIR"

# ── 3. Instalar dependencias (igual que HF Spaces) ────────────
echo ""
echo "── Instalando requirements.txt..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ""
echo "── Versiones instaladas:"
pip show smolagents        | grep -E "^(Name|Version)"
pip show langchain-core    | grep -E "^(Name|Version)"
pip show langchain-community | grep -E "^(Name|Version)"
pip show rank-bm25         | grep -E "^(Name|Version)"
pip show gradio            | grep -E "^(Name|Version)"

# ── 4. Verificar imports críticos ─────────────────────────────
echo ""
echo "── Verificando imports críticos..."
python3 - <<'PYEOF'
import sys

checks = [
    ("smolagents",             "from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, tool, Tool"),
    ("langchain_core.docs",    "from langchain_core.documents import Document"),
    ("langchain_text_splitters","from langchain_text_splitters import RecursiveCharacterTextSplitter"),
    ("BM25Retriever",          "from langchain_community.retrievers import BM25Retriever"),
    ("gradio",                 "import gradio"),
    ("yaml",                   "import yaml"),
    ("math / requests",        "import math, requests"),
    ("pytz",                   "import pytz"),
    ("pandas",                 "import pandas"),
    ("duckduckgo_search",      "from duckduckgo_search import DDGS"),
]

errors = []
for name, stmt in checks:
    try:
        exec(stmt)
        print(f"  ✅ {name}")
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        errors.append((name, str(e)))

if errors:
    print(f"\n❌ {len(errors)} import(s) fallaron:")
    for name, err in errors:
        print(f"   • {name}: {err}")
    sys.exit(1)
else:
    print("\n✅ Todos los imports OK")
PYEOF

# ── 5. Verificar sintaxis de todos los .py ────────────────────
echo ""
echo "── Verificando sintaxis de archivos Python..."
python3 - <<'PYEOF'
import ast, sys, pathlib

files = list(pathlib.Path(".").rglob("*.py"))
files = [f for f in files if ".venv_test" not in str(f)]
errors = []

for f in files:
    try:
        ast.parse(f.read_text(encoding="utf-8"))
        print(f"  ✅ {f}")
    except SyntaxError as e:
        print(f"  ❌ {f}: {e}")
        errors.append(str(f))

if errors:
    print(f"\n❌ Errores de sintaxis en: {errors}")
    sys.exit(1)
else:
    print(f"\n✅ Sintaxis OK en {len(files)} archivos")
PYEOF

# ── 6. Verificar que app.py importa sin errores ───────────────
echo ""
echo "── Importando app.py (sin lanzar Gradio)..."
python3 - <<'PYEOF'
import sys, unittest.mock

# Parchamos GradioUI.launch() para que no levante el servidor
with unittest.mock.patch("Gradio_UI.GradioUI.launch", return_value=None):
    try:
        import app
        print("✅ app.py importado y ejecutado sin errores")
    except Exception as e:
        print(f"❌ Error al importar app.py: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
PYEOF

# ── 7. Verificar que prompts.yaml existe y es válido ──────────
echo ""
echo "── Verificando prompts.yaml..."
python3 - <<'PYEOF'
import yaml, sys
try:
    with open("prompts.yaml") as f:
        data = yaml.safe_load(f)
    keys = list(data.keys()) if isinstance(data, dict) else []
    print(f"✅ prompts.yaml válido — claves: {keys}")
except Exception as e:
    print(f"❌ prompts.yaml: {e}")
    sys.exit(1)
PYEOF

# ── Resumen ───────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " ✅ BUILD EXITOSO — listo para HF Spaces"
echo "=============================================="
echo ""
echo "Para publicar al Space:"
echo "  git remote add hf https://huggingface.co/spaces/codecspy/S30-Agente-con-SmolAgents"
echo "  git push hf main --force"
echo ""

deactivate
