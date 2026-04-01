import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df.sort_values('date')


def append_row(df, date, value):
    new_row = {"date": date, "sales": value}
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)