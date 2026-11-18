# Agente Avanzado -- TokioAI

## Que es esto?

En el paper construimos un **agente seguro** con 5 capas de defensa:
validacion de inputs, registro de herramientas, sanitizacion de outputs,
segmentacion de permisos, y auditoria completa.

Eso es lo correcto para **produccion** y **entornos criticos**.

Pero para **experimentar, aprender y hacer cosas de verdad** (pentesting,
DevOps, hacking, automatizacion), necesitas un agente **sin restricciones
artificiales** -- uno que pueda ejecutar comandos, leer archivos del
sistema, conectarse por SSH a otros equipos, y resolver problemas reales.

**TokioAI** es exactamente eso.

---

## TokioAI -- Agente de IA para Hacking y DevOps

TokioAI es un agente de IA de codigo abierto especializado en:

- **Cyberseguridad**: pentesting, auditoria, analisis forense, OSINT
- **DevOps**: despliegue, monitoreo, CI/CD, infraestructura cloud
- **Ingenieria**: desarrollo full-stack, bases de datos, APIs
- **Hacking**: CTF, reverse engineering, exploit development
- **IoT**: control de robots, sensores, actuadores, Raspberry Pi

### Diferencias con el agente seguro del paper

| Caracteristica        | Agente Seguro (paper)    | TokioAI (avanzado)           |
|-----------------------|--------------------------|------------------------------|
| Herramientas          | 5 (limitadas)            | Ilimitadas (shell, SSH, etc) |
| Ejecuta comandos      | NO (solo genera scripts) | SI (ejecucion directa)       |
| Acceso al sistema     | Solo lectura basica      | Completo (root si necesario) |
| Conexiones remotas    | NO                       | SSH, IAP tunnels, APIs       |
| Memoria persistente   | JSON local simple        | Markdown con busqueda        |
| Modelos soportados    | 1 (Ollama o OpenRouter)  | 10+ (Claude, Gemini, Kimi..)|
| Multi-maquina         | NO                       | SI (Pi, GCP VMs, routers)    |
| Seguridad             | 5 capas restrictivas     | Confianza en el operador     |

### Por que las restricciones del paper no aplican aqui?

El paper explica que las 5 capas de seguridad son para **proteger un
sistema de un modelo que podria recibir inputs maliciosos de usuarios
no confiables** (ej: un chatbot publico, un asistente de atencion al
cliente).

TokioAI es diferente: **el operador ES el experto**. No hay usuarios
no confiables inyectando prompts. El agente es una extension de tus
manos -- como tener un terminal inteligente que entiende lenguaje
natural.

En ese contexto, las restricciones del agente seguro serian un
estorbo. No queres que tu herramienta de pentesting te diga "no puedo
ejecutar nmap porque no esta en el TOOL_REGISTRY".

---

## Instalacion

### Opcion 1: Desde el repositorio (recomendado)

```bash
git clone https://github.com/TokioAI/tokioai.git
cd tokioai
pip install -r requirements.txt
cp .env.example .env
# Editar .env con tu API key
python3 -m tokioai
```

### Opcion 2: Solo probar (sin instalar)

```bash
# Requisito: tener Ollama corriendo con un modelo
ollama pull qwen3:4b

# Clonar y ejecutar
git clone https://github.com/TokioAI/tokioai.git
cd tokioai
pip install -r requirements.txt
echo 'TOKIOAI_PROVIDER=ollama' > .env
python3 -m tokioai
```

---

## Que puede hacer TokioAI que el agente del paper no?

### 1. Ejecutar comandos del sistema
```
Tu > mostrame el uso de disco y los procesos que mas CPU consumen
  [execute_local] df -h
  [execute_local] ps aux --sort=-%cpu | head -10
AI > El disco esta al 77% de uso. Los procesos que mas CPU consumen son...
```

### 2. Conectarse a servidores remotos
```
Tu > conectate a mi Raspberry Pi y fijate si Ollama esta corriendo
  [execute_raspi] systemctl status ollama
AI > Ollama esta activo y corriendo en la Pi. Modelo cargado: qwen3:4b
```

### 3. Auditar seguridad
```
Tu > hace un scan completo del servidor 192.168.1.100
  [execute_local] nmap -sV -sC 192.168.1.100
  [execute_local] nikto -h 192.168.1.100
AI > Encontre 3 puertos abiertos: 22 (SSH), 80 (nginx 1.24), 443 (HTTPS)...
     Vulnerabilidad: nginx expone version en headers. Recomiendo...
```

### 4. Escribir y ejecutar codigo
```
Tu > crea un script que monitoree la temperatura de la CPU y me avise si sube de 80 grados
  [write_file] /home/user/cpu_monitor.py
  [execute_local] python3 /home/user/cpu_monitor.py &
AI > Script creado y ejecutandose en background. Te avisara si la CPU supera 80C.
```

### 5. Memoria persistente entre sesiones
```
Tu > recorda que el servidor de produccion es 10.0.1.50 y el de staging es 10.0.1.51
AI > Guardado en memoria. La proxima vez que preguntes por los servidores, ya lo se.

[... 3 dias despues ...]

Tu > cual era la IP de staging?
AI > 10.0.1.51 (lo guardaste el martes pasado)
```

---

## Relacion con el paper

Este agente avanzado es la **continuacion natural** del paper:

1. **Ejercicios 1-6**: Construiste un transformer, lo entrenaste, hiciste
   inferencia, implementaste tool calling, y armaste un agente completo.

2. **Agente seguro**: Aprendiste las 5 capas de seguridad que necesitas
   para sistemas de produccion con usuarios no confiables.

3. **Agente avanzado (TokioAI)**: Ahora tenes un agente real, sin
   restricciones artificiales, que usa todo lo que aprendiste pero
   orientado a **resolver problemas reales** de hacking y DevOps.

La progresion completa del paper es:

```
Neurona -> Transformer -> LLM -> Inferencia -> Tool Calling -> Agente Seguro
                                                                     |
                                                                     v
                                                              Agente Avanzado
                                                               (TokioAI)
```

---

## Links

- **Repositorio TokioAI**: [github.com/TokioAI/tokioai](https://github.com/TokioAI/tokioai)
- **Paper completo**: Ver directorio `paper/` en este repositorio
- **Agente seguro**: Ver directorio `agente-seguro/` en este repositorio

---

## Licencia

Uso educativo y experimental. TokioAI es open-source.
