# scripts/03_merge_datasets.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def merge_all_datasets():
    """
    Une todos los datasets limpios en uno solo
    """
    print("🔄 Unificando todos los datasets...")
    
    # Cargar datasets procesados
    liar_df = pd.read_csv('data/processed/liar_cleaned.csv')
    isot_df = pd.read_csv('data/processed/isot_cleaned.csv')
    
    print(f"\n📊 Datasets individuales:")
    print(f"  LIAR: {len(liar_df)} samples")
    print(f"  ISOT: {len(isot_df)} samples")
    
    # Concatenar
    df_all = pd.concat([liar_df, isot_df], ignore_index=True)
    
    print(f"\n✅ Dataset unificado: {len(df_all)} samples")
    
    # Shuffle
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Balance de clases
    print(f"\n📊 Balance de clases:")
    print(f"  Fake (0): {(df_all['label']==0).sum()} ({(df_all['label']==0).sum()/len(df_all)*100:.1f}%)")
    print(f"  Real (1): {(df_all['label']==1).sum()} ({(df_all['label']==1).sum()/len(df_all)*100:.1f}%)")
    
    # BALANCEAR CLASES (opcional pero recomendado)
    min_class_size = df_all['label'].value_counts().min()
    
    df_fake = df_all[df_all['label'] == 0].sample(min_class_size, random_state=42)
    df_real = df_all[df_all['label'] == 1].sample(min_class_size, random_state=42)
    
    df_balanced = pd.concat([df_fake, df_real], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n⚖️ Dataset balanceado: {len(df_balanced)} samples")
    print(f"  Fake: {(df_balanced['label']==0).sum()}")
    print(f"  Real: {(df_balanced['label']==1).sum()}")
    
    # Split: 80% train, 10% valid, 10% test
    train_df, temp_df = train_test_split(
        df_balanced, 
        test_size=0.2, 
        random_state=42,
        stratify=df_balanced['label']
    )
    
    valid_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42,
        stratify=temp_df['label']
    )
    
    print(f"\n📂 Split final:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df_balanced)*100:.1f}%)")
    print(f"  Valid: {len(valid_df)} samples ({len(valid_df)/len(df_balanced)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df_balanced)*100:.1f}%)")
    
    # Guardar
    Path('data/final').mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv('data/final/train_final.csv', index=False)
    valid_df.to_csv('data/final/valid_final.csv', index=False)
    test_df.to_csv('data/final/test_final.csv', index=False)
    
    print("\n✅ Datasets finales guardados en data/final/")
    
    # Estadísticas finales
    print("\n📊 Estadísticas de longitud de texto:")
    print(f"  Train - Mean: {train_df['length'].mean():.0f} words, Median: {train_df['length'].median():.0f}")
    print(f"  Valid - Mean: {valid_df['length'].mean():.0f} words, Median: {valid_df['length'].median():.0f}")
    print(f"  Test  - Mean: {test_df['length'].mean():.0f} words, Median: {test_df['length'].median():.0f}")
    
    return train_df, valid_df, test_df

if __name__ == "__main__":
    merge_all_datasets()