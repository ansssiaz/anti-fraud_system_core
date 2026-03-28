import polars as pl
from pathlib import Path

# Создание глобальных фич (frequency и cross-frequency features)
def build_global_features(paths):
    dfs = []
    
    for p in paths:
        df = pl.read_parquet(p).head(1_000_000).select([
            "event_dttm",
            "mcc_code",
            "event_desc",
            "event_type_nm",
            "channel_indicator_type",
            "channel_indicator_sub_type",
            "currency_iso_cd",
            "pos_cd",
            "accept_language",
            "browser_language",
            "timezone",
            "operating_system_type"
        ])
        dfs.append(df)
        
    df_all = pl.concat(dfs)
    
    df_all = df_all.sort(["customer_id", "event_dttm"])
    
    df_all = df_all.with_columns(pl.col("event_dttm").dt.hour().alias("hour"))
    
    # Frequency encoding категориальных признаков
    def frequency_encode(df, col, name):
        return df.group_by(col).len().rename({"len": name})
    
    frequency_features = {
        "mcc_freq": frequency_encode(df_all, "mcc_code", "mcc_freq"),
        "event_desc_freq": frequency_encode(df_all, "event_desc", "event_desc_freq"),
        "event_type_freq": frequency_encode(df_all, "event_type_nm", "event_type_freq"),
        "channel_freq": frequency_encode(df_all,"channel_indicator_type", "channel_freq"),
        "subchannel_freq": frequency_encode(df_all,"channel_indicator_sub_type", "subchannel_freq"),
        "currency_freq": frequency_encode(df_all, "currency_iso_cd", "currency_freq"),
        "pos_freq": frequency_encode(df_all, "pos_cd", "pos_freq"),
        "accept_lang_freq": frequency_encode(df_all, "accept_language", "accept_lang_freq"),
        "browser_lang_freq": frequency_encode(df_all, "browser_language", "browser_lang_freq"),
        "timezone_freq": frequency_encode(df_all, "timezone", "timezone_freq"),
        "os_freq": frequency_encode(df_all, "operating_system_type", "os_freq")
    }
    
    cross_frequency_features = {
        "mcc_channel_freq": frequency_encode(df_all, ["mcc_code", "channel_indicator_type"], "mcc_channel_freq"),
        "mcc_timezone_freq": frequency_encode(df_all, ["mcc_code", "timezone"], "mcc_timezone_freq"),
        "os_channel_freq": frequency_encode(df_all, ["operating_system_type", "channel_indicator_type"], "os_channel_freq"),
        "event_type_hour_freq" : frequency_encode(df_all, ["event_type_nm", "hour"], "event_type_hour_freq"),
        "mcc_currency_freq" : frequency_encode(df_all, ["mcc_code", "currency_iso_cd"], "mcc_currency_freq")
    }
    
    frequency_features.update(cross_frequency_features)

    return frequency_features 

# Target-encoding категориальных признаков на train и pretest
def target_encode(df_train, df_apply, cols, alpha):
    global_mean = df_train.select(pl.col("target").mean()).item()
    
    encodings = {}
    
    df_train = df_train.sort(["customer_id", "event_dttm"])
    
    for col in cols:
        cumsum = (pl.col("target").shift(1).cum_sum().over(col))
        cumcount = pl.col("target").shift(1).cum_count().over(col)
        
        te = ((cumsum + global_mean * alpha) / (cumcount + alpha)).alias(f"{col}_te")
        df_train = df_train.with_columns([te])
        
        #mapping = (df_train.select([col, f"{col}_te"]).unique(subset=col))
        mapping = df_train.group_by(col).agg(
            ((pl.col("target").mean())).alias(f"{col}_te")
        )
        encodings[col] = mapping
    
    df_apply_out = df_apply.clone()
    
    for col in cols:
        mapping = encodings[col]
        df_apply_out = df_apply_out.join(mapping, on=col, how="left")
        #df_apply_out = df_apply_out.fill_null({f"{col}_te": global_mean})
        df_apply_out = df_apply_out.with_columns(
            pl.col(f"{col}_te").fill_null(global_mean)
        )
    
    return df_apply_out

def build_features(df, global_features):
    df = df.sort(["customer_id", "event_dttm"])
    
    # Временные фичи
    df = df.with_columns([
        pl.col("event_dttm").dt.hour().alias("hour"),
        pl.col("event_dttm").dt.weekday().alias("weekday")
    ])
    
    # Время между операциями + стат.хар-ки
    df = df.with_columns([
        (pl.col("event_dttm") - pl.col("event_dttm").shift(1))
        .over("customer_id")
        .dt.total_seconds()
        .fill_null(0)
        .alias("time_diff")
    ])
    
    df = df.with_columns([
        pl.col("time_diff").mean().alias("mean_time_diff"),
        pl.col("time_diff").median().alias("median_time_diff"),
        pl.col("time_diff").max().alias("max_time_diff")
    ])
    
    # Пороговые признаки по времени операций
    df = df.with_columns([
        (pl.col("time_diff") < 10).cast(pl.Int8).alias("td_lt_10s"),
        (pl.col("time_diff") < 60).cast(pl.Int8).alias("td_lt_1m"),
        (pl.col("time_diff") < 300).cast(pl.Int8).alias("td_lt_5m"),
        (pl.col("time_diff") < 1800).cast(pl.Int8).alias("td_lt_30m")
    ])
    
    # Доля быстрых операций в одной сессии по отношению к прошлым быстрым операциям клиента
    df = df.with_columns([
        (pl.col("td_lt_1m").sum().over("session_id") / 
        pl.col("td_lt_1m").shift(1).cum_sum().over("customer_id"))
        .alias("fast_ops_ratio")
    ])
    
    # Rolling-фичи по клиенту и времени
    df = add_time_rolling(df)
    
    # Логарифмированная сумма операции
    df = df.with_columns([
        pl.col("operaton_amt").fill_null(0),
        pl.col("operaton_amt").log1p().alias("log_amt")
    ])
    
    # Фичи по сумме операции на клиента
    df = df.with_columns([
        pl.len().over("customer_id").alias("num_events"),
        
        pl.col("operaton_amt").mean().over("customer_id").alias("mean_amt"),
        pl.col("operaton_amt").std().over("customer_id").alias("std_amt"),
        pl.col("operaton_amt").max().over("customer_id").alias("max_amt")
    ])
    
    # Z-сумма операций
    df = df.with_columns([
        (
            (pl.col("operaton_amt")) - (pl.col("mean_amt")) /
            (pl.col("std_amt") + 1e-6)
        ).alias("amt_zscore")
    ])
    
    # Фичи с отсутствием данных о подключении/девайсе
    df = df.with_columns([
        pl.col("device_system_version").is_null().cast(pl.Int8).alias("device_missing"),
        pl.col("browser_language").is_null().cast(pl.Int8).alias("browser_missing"),
        pl.col("timezone").is_null().cast(pl.Int8).alias("timezone_missing"),
    ])
    
    # Частые признаки мошеннических операций
    df = df.with_columns([
        (pl.col("phone_voip_call_state") == 1).cast(pl.Int8).alias("voip_flag"),
        (pl.col("web_rdp_connection") == 1).cast(pl.Int8).alias("rdp_flag"),
        (pl.col("compromised") == 1).cast(pl.Int8).alias("compromised_flag")
    ])
    
    # Фичи по сессиям
    df = df.sort(["customer_id", "session_id", "event_dttm"])
    
    df = df.with_columns([
        pl.len().over("session_id").alias("session_num_events"),
        pl.col("operaton_amt").sum().over("session_id").alias("session_sum_amt"),
        pl.col("operaton_amt").mean().over("session_id").alias("session_mean_amt"),
        pl.col("operaton_amt").median().over("session_id").alias("session_median_amt"),
        pl.col("operaton_amt").max().over("session_id").alias("session_max_amt"),
        
        # Временные признаки сессии
        (pl.col("event_dttm").max().over("session_id") - pl.col("event_dttm").min().over("session_id")).dt.total_seconds().alias("session_duration"),
        pl.col("time_diff").mean().over("session_id").alias("session_avg_time_diff"),
        
        # Кол-во уникальных типов операций, каналов, ОС
        pl.col("event_type_nm").n_unique().over("session_id").alias("session_unique_event_types"),
        pl.col("channel_indicator_type").n_unique().over("session_id").alias("session_unique_channel_types"),
        pl.col("operating_system_type").n_unique().over("session_id").alias("session_unique_os_types")
    ]) 
        
        # Cross-feature с клиентом
    df = df.with_columns([
        pl.col("session_num_events").mean().over("customer_id").alias("cust_avg_num_events_per_session"),
        pl.col("session_num_events").max().over("customer_id").alias("cust_max_num_events_per_session"),
         
        pl.col("session_sum_amt").mean().over("customer_id").alias("cust_avg_sum_amt_per_session"),
        pl.col("session_sum_amt").max().over("customer_id").alias("cust_max_sum_amt_per_session")
        ])
    
    # Frequency-features
    df = df.join(global_features["mcc_freq"], on="mcc_code", how="left")
    df = df.join(global_features["event_desc_freq"], on="event_desc", how="left")
    df = df.join(global_features["event_type_freq"], on="event_type_nm", how="left")
    df = df.join(global_features["channel_freq"], on="channel_indicator_type", how="left")
    df = df.join(global_features["subchannel_freq"], on="channel_indicator_sub_type", how="left")
    df = df.join(global_features["currency_freq"], on="currency_iso_cd", how="left")
    df = df.join(global_features["pos_freq"], on="pos_cd", how="left")
    df = df.join(global_features["accept_lang_freq"], on="accept_language", how="left")
    df = df.join(global_features["browser_lang_freq"], on="browser_language", how="left")
    df = df.join(global_features["timezone_freq"], on="timezone", how="left")
    df = df.join(global_features["os_freq"], on="operating_system_type", how="left")
    
    # Cross-frequency features
    df = df.join(global_features["event_type_hour_freq"], on=["event_type_nm", "hour"], how="left")
    df = df.join(global_features["mcc_channel_freq"], on=["mcc_code", "channel_indicator_type"], how="left")
    df = df.join(global_features["mcc_timezone_freq"], on=["mcc_code", "timezone"], how="left")
    df = df.join(global_features["os_channel_freq"], on=["operating_system_type", "channel_indicator_type"], how="left")
    df = df.join(global_features["mcc_currency_freq"], on=["mcc_code", "currency_iso_cd"], how="left")
    
    # Редкие значения категориальных признаков
    df = add_rare_flag(df, global_features, "mcc_code", "mcc_freq")
    df = add_rare_flag(df, global_features, "event_desc", "event_desc_freq")
    df = add_rare_flag(df, global_features, "event_type_nm", "event_type_freq")
    df = add_rare_flag(df, global_features, "channel_indicator_type", "channel_freq")
    df = add_rare_flag(df, global_features, "channel_indicator_sub_type", "subchannel_freq")
    df = add_rare_flag(df, global_features, "currency_iso_cd", "currency_freq")
    df = add_rare_flag(df, global_features, "pos_cd", "pos_freq")
    df = add_rare_flag(df, global_features, "accept_language", "accept_lang_freq")
    df = add_rare_flag(df, global_features, "browser_language", "browser_lang_freq")
    df = add_rare_flag(df, global_features, "timezone", "timezone_freq")
    df = add_rare_flag(df, global_features, "operating_system_type", "os_freq")
    
    df = add_rare_flag(df, global_features, ["event_type_nm", "hour"], "event_type_hour_freq")
    df = add_rare_flag(df, global_features, ["mcc_code", "channel_indicator_type"], "mcc_channel_freq")
    df = add_rare_flag(df, global_features, ["mcc_code", "timezone"], "mcc_timezone_freq")
    df = add_rare_flag(df, global_features, ["operating_system_type", "channel_indicator_type"], "os_channel_freq")
    df = add_rare_flag(df, global_features, ["mcc_code", "currency_iso_cd"], "mcc_currency_freq")
    
    df = df.sort(["customer_id", "event_dttm"])
    
    return df

# Rolling-фичи по количеству операций, общей сумме и среднему значению суммы операций 1 клиента за 1 день и за 7 дней
def add_time_rolling(df):
    df = df.sort(["customer_id", "event_dttm"])
    
    rolling_1d = (
        df.rolling(
            index_column="event_dttm",
            period="1d",
            by="customer_id"
        )
        .agg([
            pl.len().alias("cnt_1d"),
            pl.col("operaton_amt").sum().alias("sum_1d"),
            pl.col("operaton_amt").mean().alias("mean_1d")
        ])
    )
    
    rolling_7d = (
        df.rolling(
            index_column="event_dttm",
            period="7d",
            by="customer_id"
        )
        .agg([
            pl.len().alias("cnt_7d"),
            pl.col("operaton_amt").sum().alias("sum_7d"),
            pl.col("operaton_amt").mean().alias("mean_7d")
        ])
    )
    
    # Исключаем текущую строку, чтобы учесть только прошлые события
    rolling_1d = rolling_1d.with_columns([
        pl.col("cnt_1d").shift(1).over("customer_id"),
        pl.col("sum_1d").shift(1).over("customer_id"),
        pl.col("mean_1d").shift(1).over("customer_id")
    ])
    
    rolling_7d = rolling_7d.with_columns([
        pl.col("cnt_7d").shift(1).over("customer_id"),
        pl.col("sum_7d").shift(1).over("customer_id"),
        pl.col("mean_7d").shift(1).over("customer_id")
    ])
    
    df = df.join(
        rolling_1d,
        on=["customer_id", "event_dttm"],
        how="left"
    )
    
    df = df.join(
        rolling_7d,
        on=["customer_id", "event_dttm"],
        how="left"
    )
    
    return df

# Бинарные фичи для редких значений категориальных признаков
def add_rare_flag(df, global_features, col, freq_col):
    if isinstance(col, str):
        cols = [col]
    else:
        cols = col
    
    threshold = (
        global_features[freq_col]
        .select(pl.col(freq_col).quantile(0.05))
        .item()
    )
    
    rare_values = (
        global_features[freq_col]
        .filter(pl.col(freq_col) < threshold)
        .select(cols)
    )
    
    df = df.join(
        rare_values.with_columns(
            pl.lit(1).alias(f"{'_'.join(cols)}_is_rare")
        ),
        on=cols,
        how="left"
    )
    
    df = df.with_columns(
        pl.col(f"{'_'.join(cols)}_is_rare").fill_null(0)
    )
    
    return df