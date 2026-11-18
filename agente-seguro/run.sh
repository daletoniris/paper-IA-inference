#!/bin/bash
# =============================================================================
# run.sh - Lanzador del agente (Linux / macOS / Raspberry Pi)
# =============================================================================
# Crea el entorno virtual si no existe, instala dependencias y ejecuta.
# Uso: ./run.sh
# =============================================================================

set -e

cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo "  Agente Seguro - Setup y Ejecucion"
echo "============================================================"
echo ""

# --- Paso 1: Verificar Python ---
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 no esta instalado."
    echo "Instalar con: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[1/3] Python $PYTHON_VERSION detectado."

# --- Paso 2: Crear entorno virtual si no existe ---
if [ ! -d ".venv" ]; then
    echo "[2/3] Creando entorno virtual..."
    python3 -m venv .venv
    echo "       Entorno virtual creado."
else
    echo "[2/3] Entorno virtual ya existe."
fi

# --- Paso 3: Activar e instalar dependencias ---
echo "[3/3] Verificando dependencias..."
source .venv/bin/activate
pip install --quiet openai python-dotenv 2>/dev/null
echo "       Listo."

# --- Verificar configuracion ---
if [ ! -f ".env" ]; then
    echo ""
    echo "  ATENCION: No encontre el archivo .env"
    echo "  Copialo desde el ejemplo:"
    echo "    cp .env.example .env"
    echo "  Y editalo con tu configuracion (backend, API key, modelo)."
    echo ""
    exit 1
fi

# Leer backend del .env
BACKEND=$(grep -E '^BACKEND=' .env 2>/dev/null | cut -d= -f2 | tr -d ' "' || echo "")

if [ "$BACKEND" = "ollama" ]; then
    # Verificar Ollama solo si el backend es ollama
    OLLAMA_HOST=$(grep -E '^OLLAMA_HOST=' .env 2>/dev/null | cut -d= -f2 | tr -d ' "' || echo "http://localhost:11434")
    
    echo ""
    echo "  Backend: Ollama ($OLLAMA_HOST)"
    
    if ! curl -s "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
        echo "  AVISO: Ollama no responde en $OLLAMA_HOST"
        echo ""
        
        if [ "$OLLAMA_HOST" = "http://localhost:11434" ]; then
            echo "  Opciones:"
            echo "    1. Iniciar Ollama: ollama serve"
            echo "    2. Instalar Ollama: curl -fsSL https://ollama.com/install.sh | sh"
            echo "    3. Cambiar a OpenRouter: editar .env y poner BACKEND=openrouter"
        else
            echo "  Verifica que el servidor Ollama este corriendo."
            echo "  O cambia a OpenRouter editando .env."
        fi
        echo ""
        exit 1
    fi
    echo "  Ollama OK."
else
    echo ""
    echo "  Backend: OpenRouter"
fi

echo ""

# --- Ejecutar el agente ---
python3 agent.py

echo ""
echo "Agente finalizado."
