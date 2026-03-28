from prepare_data import prepare_data
import polars as pl
def main():
    train_features_paths = prepare_data()
    #train_features_paths, pretest_features_path = prepare_data()
    
    dfs_train = [pl.read_parquet(p) for p in train_features_paths]
    
    df_train = pl.concat(dfs_train)
    
    df_train = df_train.filter(pl.col("target").is_not_null())
    
if __name__ == "__main__":
    main()