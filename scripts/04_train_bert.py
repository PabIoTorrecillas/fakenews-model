# scripts/04_train_bert.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN GLOBAL
# ============================================

CONFIG = {
    'model_name': 'distilbert-base-uncased',  # Modelo base
    'max_length': 512,                         # Longitud máxima de tokens
    'batch_size': 16,                          # Batch size (ajusta según GPU)
    'learning_rate': 2e-5,                     # Learning rate
    'num_epochs': 3,                           # Épocas de entrenamiento
    'warmup_steps': 500,                       # Warmup steps
    'weight_decay': 0.01,                      # Regularización
    'output_dir': './models/bert_fakenews_v1',
    'logging_dir': './results/logs',
    'save_steps': 1000,                        # Guardar cada N steps
    'eval_steps': 500,                         # Evaluar cada N steps
    'seed': 42
}

# ============================================
# CLASE DATASET PERSONALIZADA
# ============================================

class FakeNewsDataset(Dataset):
    """
    Dataset personalizado para PyTorch
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenizar
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ============================================
# FUNCIÓN DE MÉTRICAS
# ============================================

def compute_metrics(pred):
    """
    Calcula métricas durante el entrenamiento
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ============================================
# CARGAR DATOS
# ============================================

def load_data():
    """
    Carga los datasets finales
    """
    print("📂 Cargando datos...")
    
    train_df = pd.read_csv('data/final/train_final.csv')
    valid_df = pd.read_csv('data/final/valid_final.csv')
    test_df = pd.read_csv('data/final/test_final.csv')
    
    print(f"✅ Datos cargados:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Valid: {len(valid_df)} samples")
    print(f"   Test:  {len(test_df)} samples")
    
    return train_df, valid_df, test_df

# ============================================
# PREPARAR DATASETS DE PYTORCH
# ============================================

def prepare_datasets(train_df, valid_df, test_df, tokenizer):
    """
    Crea datasets de PyTorch
    """
    print("\n🔄 Preparando datasets de PyTorch...")
    
    train_dataset = FakeNewsDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=CONFIG['max_length']
    )
    
    valid_dataset = FakeNewsDataset(
        texts=valid_df['text'].tolist(),
        labels=valid_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=CONFIG['max_length']
    )
    
    test_dataset = FakeNewsDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=CONFIG['max_length']
    )
    
    print("✅ Datasets preparados")
    
    return train_dataset, valid_dataset, test_dataset

# ============================================
# ENTRENAR MODELO
# ============================================

def train_model():
    """
    Función principal de entrenamiento
    """
    print("=" * 60)
    print("🚀 INICIANDO ENTRENAMIENTO DE BERT")
    print("=" * 60)
    
    # Set seed para reproducibilidad
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Verificar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Cargar datos
    train_df, valid_df, test_df = load_data()
    
    # Cargar tokenizer y modelo
    print(f"\n🔄 Cargando modelo: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=2,  # Binario: Fake (0) o Real (1)
        problem_type="single_label_classification"
    )
    
    print("✅ Modelo y tokenizer cargados")
    
    # Preparar datasets
    train_dataset, valid_dataset, test_dataset = prepare_datasets(
        train_df, valid_df, test_df, tokenizer
    )
    
    # Configurar argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'] * 2,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        warmup_steps=CONFIG['warmup_steps'],
        logging_dir=CONFIG['logging_dir'],
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=CONFIG['eval_steps'],
        save_strategy="steps",
        save_steps=CONFIG['save_steps'],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        report_to="tensorboard",
        seed=CONFIG['seed'],
        fp16=torch.cuda.is_available(),  # Mixed precision si hay GPU
    )
    
    # Crear Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # ENTRENAR
    print("\n" + "=" * 60)
    print("🧠 COMENZANDO ENTRENAMIENTO")
    print("=" * 60)
    print(f"Épocas: {CONFIG['num_epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print(f"Total steps: ~{len(train_dataset) // CONFIG['batch_size'] * CONFIG['num_epochs']}")
    print("=" * 60 + "\n")
    
    start_time = datetime.now()
    
    trainer.train()
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "=" * 60)
    print(f"✅ ENTRENAMIENTO COMPLETADO en {training_time:.2f} minutos")
    print("=" * 60)
    
    # Guardar modelo final
    print("\n💾 Guardando modelo final...")
    model.save_pretrained(CONFIG['output_dir'])
    tokenizer.save_pretrained(CONFIG['output_dir'])
    
    # Guardar configuración
    with open(f"{CONFIG['output_dir']}/training_config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    print(f"✅ Modelo guardado en: {CONFIG['output_dir']}")
    
    # ============================================
    # EVALUACIÓN EN TEST SET
    # ============================================
    
    print("\n" + "=" * 60)
    print("📊 EVALUACIÓN EN TEST SET")
    print("=" * 60)
    
    test_results = trainer.predict(test_dataset)
    
    y_true = test_results.label_ids
    y_pred = test_results.predictions.argmax(-1)
    
    # Métricas detalladas
    print("\n📈 Classification Report:")
    print(classification_report(
        y_true, y_pred, 
        target_names=['Fake', 'Real'],
        digits=4
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n🔢 Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Fake  Real")
    print(f"Actual Fake   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       Real   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Calcular métricas adicionales
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    final_metrics = {
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'training_time_minutes': float(training_time),
        'total_samples': len(train_df) + len(valid_df) + len(test_df),
        'model_name': CONFIG['model_name']
    }
    
    # Guardar métricas
    with open(f"{CONFIG['output_dir']}/test_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print("\n📊 MÉTRICAS FINALES:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    print("\n" + "=" * 60)
    print("🎉 ¡PROCESO COMPLETADO!")
    print("=" * 60)
    
    return trainer, final_metrics

# ============================================
# FUNCIÓN DE PRUEBA RÁPIDA
# ============================================

def test_single_prediction(text, model_path=None):
    """
    Prueba el modelo con un texto individual
    """
    if model_path is None:
        model_path = CONFIG['output_dir']
    
    print(f"\n🧪 Probando predicción con texto nuevo...")
    
    # Cargar modelo
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Tokenizar
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True
    ).to(device)
    
    # Predicción
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
    
    label = "Real" if prediction == 1 else "Fake"
    
    print(f"\n📝 Texto: {text[:100]}...")
    print(f"🎯 Predicción: {label}")
    print(f"📊 Confianza: {confidence*100:.2f}%")
    
    return label, confidence

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    # Entrenar modelo
    trainer, metrics = train_model()
    
    # Prueba rápida
    test_examples = [
        "Breaking: Scientists discover cure for cancer in groundbreaking study",
        "You won't believe what this celebrity did! Doctors hate him!",
        "The President announced new economic policies during today's press conference",
    ]
    
    print("\n" + "=" * 60)
    print("🧪 PRUEBAS CON EJEMPLOS")
    print("=" * 60)
    
    for example in test_examples:
        test_single_prediction(example)
        print()
    
    print("\n✅ Todo listo! Revisa los resultados en:")
    print(f"   - Modelo: {CONFIG['output_dir']}")
    print(f"   - Logs: {CONFIG['logging_dir']}")
    print(f"   - Tensorboard: tensorboard --logdir={CONFIG['logging_dir']}")