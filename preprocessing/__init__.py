# preprocessing/__init__.py
"""
Preprocessing modülü için gerekli fonksiyonlar
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_age_group_column(df, age_col='age', new_col='age_group'):
    """
    Yaş grupları oluşturur
    """
    bins = [0, 20, 40, 60, 100]
    labels = ['0-20', '21-40', '41-60', '61+']
    df[new_col] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
    return df


def apply_loc_mean_age(train_df, test_df, age_col='age', group_col='localization', new_col='loc_mean_age'):
    """
    Eğitim verisinden 'localization' gruplarına göre ortalama yaş hesaplar ve hem eğitim hem test verisine uygular.
    """
    mean_age_map = train_df.groupby(group_col)[age_col].mean().to_dict()

    # Uygula
    train_df[new_col] = train_df[group_col].map(mean_age_map)
    test_df[new_col] = test_df[group_col].map(mean_age_map)

    return train_df, test_df, mean_age_map


def age_dev_from_loc_mean(traindf, testdf, selectedCol1='age', selectedCol2='loc_mean_age',
                          new_col='age_dev_from_loc_mean'):
    """
    Yaş sapma hesaplaması
    """
    traindf[new_col] = traindf[selectedCol1] - traindf[selectedCol2]
    testdf[new_col] = testdf[selectedCol1] - testdf[selectedCol2]
    return traindf, testdf

#__init__.py dosyasının sonuna ekleyin:
__all__ = ['create_age_group_column', 'apply_loc_mean_age', 'age_dev_from_loc_mean']