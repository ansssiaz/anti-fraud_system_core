from build_features import build_global_features, build_features, target_encode
import polars as pl
from pathlib import Path
import os
from tqdm import tqdm
import time
from log import log, log_df

def prepare_data():
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
    
    pretest_path = "data/pretest.parquet"
    test_path = "data/test.parquet"
    
    all_train_features_save_path = "faetures/train_features.parquet"
    
    labels_path = "data/train_labels.parquet"
    
    all_paths = pretrain_paths + train_paths + [pretest_path] + [test_path]
    
    labels = pl.read_parquet(labels_path)

    # Предобработка данных
    prepared_paths = []

    for p in tqdm(all_paths, desc="Preparing data"):
        prepared_save_path = Path("prepared") / Path(p).name
        
        if prepared_save_path.exists():
            log(f"Skip prepared: {prepared_save_path}")
            if p != test_path:
                prepared_paths.append(prepared_save_path)
            continue
        
        start = time.time()
        
        df = pl.read_parquet(p)
        log_df(df, f"Loaded {p}")
        
        df = df.with_columns(
            pl.col("event_dttm").str.strptime(pl.Datetime, strict=False),
            pl.col("compromised").cast(pl.Int8, strict=False)
        )
       
        prepared_save_path.parent.mkdir(exist_ok=True)
        df.write_parquet(prepared_save_path)
        
        log(f"Saved prepared: {prepared_save_path} in {time.time() - start:.2f}s")
        
        if p != test_path:
            prepared_paths.append(prepared_save_path)
        
    # Генерация глобальных фич на pretrain + train + pretest
    global_features_save_path = Path("features/global_features")
    
    def load_global_features(path):
        features = {}
        for file in os.listdir(path):
            name = file.replace(".parquet", "")
            features[name] = pl.read_parquet(f"{path}/{file}")
        return features

    if global_features_save_path.exists() and len(os.listdir(global_features_save_path)) > 0:
        log("Loading global features...")
        global_features = load_global_features(global_features_save_path)
    else:
        log("Building global features...")
        start = time.time()
        
        global_features = build_global_features(prepared_paths)
        
        log(f"Global features done in {time.time() - start:.2f}s")
        
        global_features_save_path.mkdir(parents=True, exist_ok=True)
        
    for name, df_feat in global_features.items():
        global_feature_save_path = global_features_save_path / f"{name}.parquet"
        df_feat.write_parquet(global_feature_save_path)

        log_df(df_feat, f"Global feature: {name}")
        log(f"Saved: {global_feature_save_path}")
    
    # Генерация фич для train
    target_encoding_cols = ["mcc_code", "event_type_nm", "channel_indicator_type", "pos_cd"]
    
    train_features_save_paths = []
    
    for p in tqdm(train_paths, desc="Building train features"):
        train_features_save_path = Path("features") / Path(p).name.replace(".parquet", "_features.parquet")
        
        if train_features_save_path.exists():
            log(f"Skip train features: {train_features_save_path}")
            train_features_save_paths.append(train_features_save_path)
            continue
        
        start = time.time()
        
        df = pl.read_parquet(Path("prepared") / Path(p).name)
        log_df(df, f"Loaded train {p}")
        
        df = df.join(labels, on=["customer_id", "event_id"], how="left")
        
        df = build_features(df, global_features)
        log_df(df, "Train feature engineering done")
        
        train_features_save_path.parent.mkdir(exist_ok=True)
        df.write_parquet(train_features_save_path)
        
        log(f"Saved train features: {train_features_save_path} in {time.time() - start:.2f}s")
        train_features_save_paths.append(train_features_save_path)
    
    # Объединение train dataframes, target encoding
    if all_train_features_save_path.exists():
        log(f"Skip target encoding train: {all_train_features_save_path}")
    else:
        start = time.time()
        
        dfs_train = [pl.read_parquet(p) for p in train_features_save_paths]
        df_train_all= pl.concat(dfs_train)
        df_train_all = df_train_all.filter(pl.col("target").is_not_null())
            
        df_train_all = target_encode(
            df_train=df_train_all,
            df_apply = df_train_all,
            cols = target_encoding_cols,
            alpha = 50
        )
        
        all_train_features_save_path.parent.mkdir(exist_ok=True)
        df_train_all.write_parquet(all_train_features_save_path)
        
        log(f"Saved all train features: {all_train_features_save_path} in {time.time() - start:.2f}s")
            
    # Генерация фич для test
    test_features_save_path = Path("features") / Path(test_path).name.replace(".parquet", "_features.parquet")
        
    if test_features_save_path.exists():
        log(f"Skip test features: {test_features_save_path}")
    else:
        start = time.time()
        
        df_test = pl.read_parquet(Path("prepared") / Path(test_path).name)
        log_df(df_test, f"Loaded test {test_path}")
        
        df_test = build_features(df_test, global_features)
        log_df(df_test, "Test feature engineering done")
        
        df_test = target_encode(
            df_train=df_train_all,
            df_apply = df_test,
            cols = target_encoding_cols,
            alpha = 50
        )
        
        test_features_save_path.parent.mkdir(exist_ok=True)
        df_test.write_parquet(test_features_save_path)
        
        log(f"Saved test features: {test_features_save_path} in {time.time() - start:.2f}s")

    # Target-encoding для pretest
    # pretest_features_save_path = Path("features/pretest_features.parquet")
    
    # if not pretest_features_save_path.exists():
    #     df_pretest = pl.read_parquet("prepared/pretest.parquet")

    #     df_train_all = pl.concat([
    #         pl.read_parquet(p) for p in train_features_save_paths
    #     ])

        # df_pretest = target_encode(
        #     df_train = df_train_all,
        #     df_apply= df_pretest,
        #     cols = target_encoding_cols,
        #     alpha = 50
        # )
    
        # df_pretest.write_parquet(pretest_features_save_path)
    
    #return train_features_save_paths, pretest_features_save_path
    
    return all_train_features_save_path, test_features_save_path