from build_features import build_global_features
import polars as pl

pretrain_paths = [
    "data/pretrain_part_1.parquet",
    "data/pretrain_part_2.parquet",
    "data/pretrain_part_3.parquet"
]

train_paths = [
    "data/train_part_1.parquet",
    "data/train_part_2.parquet",
    "data/train_part_3.parquet"
]

labels = "data/train_labels.parquet"

def add_labels(df, labels):
    return df.join(
        labels,
        on=["customer_id", "event_id"],
        how="left"
    )

pretest_path = "data/pretest.parquet"

# Генерация/загрузка глобальных фич
def load_global_features(path="features/global_features"):
    import os
    
    features = {}
    
    for file in os.listdir(path):
        name = file.replace(".parquet", "")
        features[name] = pl.read_parquet(f"{path}/file")
    
    return features

all_paths = pretrain_paths + train_paths + pretest_path

global_features = build_global_features(all_paths)

global_features = load_global_features()

# Генерация локальных фич
for p in train_paths:
    df = pl.read_parquet(p)
    df = add_labels(df, labels)
    #df = build_local_features(df, global_features)
    #df = add_time_rolling(df)
    
    df.write_parquet(p.replace(".parquet", "_features.parquet"))