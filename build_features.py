import polars as pl
from pathlib import Path

def build_global_features(paths, save_path = "features/global_features"):
    dfs = []
    
    for p in paths:
        df = pl.read_parquet(p).select([
            "mcc_code",
            "event_desc",
            "event_type_nm",
            "channel_indicator_type",
            "currency_iso_cd",
            "pos_cd",
            "accept_language",
            "browser_language",
            "timezone",
            "operating_system_type"
        ])
        dfs.append(df)
        
    df_all = pl.concat(dfs)
    
    # Frequency encoding категориальных признаков
    def frequency_encode(df, col, name):
        return df.group_by(col).len().rename({"len": name})
    
    features = {
        "mcc_freq": frequency_encode(df_all, "mcc_code", "mcc_freq"),
        "event_desc_freq": frequency_encode(df_all, "event_desc", "event_desc_freq"),
        "event_type_freq": frequency_encode(df_all, "event_type_nm", "event_type_freq"),
        "channel_freq": frequency_encode(df_all,"channel_indicator_type", "channel_freq"),
        "subchannel_freq": frequency_encode(df_all,"subchannel_indicator_type", "subchannel_freq"),
        "currency_freq": frequency_encode(df_all, "currency_iso_cd", "currency_freq"),
        "pos_freq": frequency_encode(df_all, "pos_cd", "pos_freq"),
        "accept_lang_freq": frequency_encode(df_all, "accept_language", "accept_lang_freq"),
        "browser_lang_freq": frequency_encode(df_all, "browser_language", "browser_lang_freq"),
        "timezone_freq": frequency_encode(df_all, "timezone", "timezone_freq"),
        "os_freq": frequency_encode(df_all, "operating_system_type", "os_freq")
    }
    
    Path(save_path).mkdir(exist_ok=True)
    
    for name, df_feat in features.items():
        df_feat.write_parquet(f"{save_path}/{name}.parquet")

    return features 