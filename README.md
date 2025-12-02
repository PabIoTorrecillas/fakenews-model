
# Fake News Detector

Detector de noticias falsas usando BERT y Big Data.

##  Quick Start

### Entrenar Modelo (Modelo subido al respositario, no es necesario re-entrenarlo)
Para entrenar el modelo es necesario una tarjeta grafica, para mi fue una 3070 con una duracion de 30 minutos
```bash
python scripts/04_train_bert.py
```

### Backend API
Ejecutar despues de entrenamiento
```bash
cd backend
pip install -r requirements.txt
python app.py
```

API disponible en: http://localhost:8000

##  Estructura

- `data/` - Datasets
- `models/` - Modelos entrenados
- `backend/` - API FastAPI
- `backend/test` - Test de API y modelo
- `chrome-extension/` - Extensión de Chrome
- `scripts/` - Scripts de entrenamiento
- `notebooks/` - Análisis

##  Documentación (API)


Ver `docs/` para más información.

