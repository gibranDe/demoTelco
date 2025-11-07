# 🚀 Demo RAG Incidencias – Python + MongoDB + VoyageAI + OpenAI

Demo de búsqueda semántica (RAG) con MongoDB Atlas Vector Search, embeddings de VoyageAI y modelos de OpenAI.

---

## ✅ Requisitos

- Python 3.10+
- MongoDB Atlas
- Claves API de VoyageAI y OpenAI

---

## 🔧 Instalación

**1. Crear entorno virtual**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Instalar dependencias**
```bash
pip install -r requirements.txt
```

**3. Configurar variables de entorno**

Crear archivo `.env`:
```bash
MONGODB_URI=tu_uri
VOYAGE_API_KEY=tu_key
OPENAI_API_KEY=tu_key
```

**4. Ejecutar**
```bash
python app.py
```

**5. Abrir** → http://127.0.0.1:8000/