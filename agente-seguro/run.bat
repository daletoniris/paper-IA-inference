@echo off
REM ==========================================================================
REM run.bat - Lanzador del agente TokioAI Mini (Windows)
REM ==========================================================================
REM Crea el entorno virtual si no existe, instala dependencias y ejecuta.
REM ==========================================================================

cd /d "%~dp0"
echo.
echo ============================================================
echo   TokioAI Mini Agent - Setup y Ejecucion
echo ============================================================
echo.

REM --- Crear entorno virtual si no existe ---
if not exist ".venv" (
    echo [1/3] Creando entorno virtual...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: No se pudo crear el entorno virtual.
        echo Asegurate de tener Python 3.10+ instalado.
        pause
        exit /b 1
    )
    echo       Entorno virtual creado.
) else (
    echo [1/3] Entorno virtual ya existe.
)

REM --- Activar entorno virtual ---
echo [2/3] Activando entorno virtual...
call .venv\Scripts\activate.bat

REM --- Instalar dependencias ---
echo [3/3] Verificando dependencias...
pip install --quiet openai python-dotenv
echo       Listo.
echo.

REM --- Ejecutar el agente ---
python agent.py

REM --- Al salir ---
echo.
echo Agente finalizado. Presiona una tecla para cerrar.
pause
