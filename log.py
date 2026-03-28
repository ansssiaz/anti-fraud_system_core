def log(message):
    print(f"[LOG] {message}")
    
def log_df(df, name):
    log(f"{name}: shape={df.shape}, cols={len(df.columns)}")