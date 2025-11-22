# scripts/01_clean_liar.py

import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    """
    Limpia texto: remueve URLs, caracteres especiales, espacios extra
    """
    if pd.isna(text):
        return ""
    
    # Convertir a string
    text = str(text)
    
    # Remover URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remover emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remover caracteres especiales pero mantener puntuación básica
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Remover espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    # Trim
    text = text.strip()
    
    return text

def process_liar_dataset():
    """
    Procesa LIAR dataset y lo estandariza
    """
    print("🔄 Procesando LIAR Dataset...")
    
    # Definir columnas
    COLUMN_NAMES = [
        'id', 'label', 'statement', 'subject', 'speaker',
        'job_title', 'state', 'party', 
        'barely_true_count', 'false_count', 'half_true_count',
        'mostly_true_count', 'pants_fire_count', 'context'
    ]
    
    # Cargar datos
    train_df = pd.read_csv('data/raw/liar/train.tsv', sep='\t', 
                           header=None, names=COLUMN_NAMES)
    valid_df = pd.read_csv('data/raw/liar/valid.tsv', sep='\t', 
                           header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv('data/raw/liar/test.tsv', sep='\t', 
                          header=None, names=COLUMN_NAMES)
    
    # Concatenar todos
    df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    
    print(f"  Total samples: {len(df)}")
    
    # Convertir labels a binario
    def label_to_binary(label):
        REAL = ['true', 'mostly-true']
        FAKE = ['false', 'barely-true', 'pants-on-fire']
        
        if label in REAL:
            return 1
        elif label in FAKE:
            return 0
        else:
            return None  # half-true
    
    df['label_binary'] = df['label'].apply(label_to_binary)
    df = df.dropna(subset=['label_binary'])
    
    print(f"  After filtering 'half-true': {len(df)}")
    
    # Crear texto completo
    df['text'] = df['statement'] + ' ' + df['context'].fillna('')
    df['text'] = df['text'].apply(clean_text)
    
    # Filtrar textos muy cortos (menos de 10 palabras)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= 10]
    
    print(f"  After filtering short texts: {len(df)}")
    
    # Crear dataframe estandarizado
    df_clean = pd.DataFrame({
        'text': df['text'],
        'label': df['label_binary'].astype(int),
        'source': 'liar',
        'subject': df['subject'],
        'length': df['word_count']
    })
    
    # Guardar
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    df_clean.to_csv('data/processed/liar_cleaned.csv', index=False)
    
    print(f"✅ LIAR procesado: {len(df_clean)} samples")
    print(f"   Fake: {(df_clean['label']==0).sum()}")
    print(f"   Real: {(df_clean['label']==1).sum()}")
    
    return df_clean

if __name__ == "__main__":
    process_liar_dataset()