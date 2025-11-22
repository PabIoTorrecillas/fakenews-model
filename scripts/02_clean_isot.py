# scripts/02_clean_isot.py

import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    """
    Limpia texto (misma función que antes)
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def process_isot_dataset():
    """
    Procesa ISOT Fake News Dataset
    """
    print("🔄 Procesando ISOT Dataset...")
    
    # Cargar datos
    fake_df = pd.read_csv('data/raw/isot/Fake.csv')
    true_df = pd.read_csv('data/raw/isot/True.csv')
    
    print(f"  Fake articles: {len(fake_df)}")
    print(f"  Real articles: {len(true_df)}")
    
    # Agregar labels
    fake_df['label'] = 0  # Fake
    true_df['label'] = 1  # Real
    
    # Concatenar
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Combinar título + texto
    df['text'] = df['title'] + '. ' + df['text']
    df['text'] = df['text'].apply(clean_text)
    
    # Filtrar textos muy cortos o muy largos
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df = df[(df['word_count'] >= 50) & (df['word_count'] <= 1000)]
    
    print(f"  After filtering by length: {len(df)}")
    
    # Remover duplicados
    df = df.drop_duplicates(subset=['text'])
    print(f"  After removing duplicates: {len(df)}")
    
    # Crear dataframe estandarizado
    df_clean = pd.DataFrame({
        'text': df['text'],
        'label': df['label'],
        'source': 'isot',
        'subject': df['subject'],
        'length': df['word_count']
    })
    
    # Guardar
    df_clean.to_csv('data/processed/isot_cleaned.csv', index=False)
    
    print(f"✅ ISOT procesado: {len(df_clean)} samples")
    print(f"   Fake: {(df_clean['label']==0).sum()}")
    print(f"   Real: {(df_clean['label']==1).sum()}")
    
    return df_clean

if __name__ == "__main__":
    process_isot_dataset()