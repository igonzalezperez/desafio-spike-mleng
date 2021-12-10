import pandas as pd


def convert_int(x: str) -> int:
    """
    Convert integer-like string to integer

    Args:
        x (str): String representing an integer.

    Returns:
        int: Integer.
    """
    return int(x.replace('.', ''))


def to_100(x: str) -> float:
    """
    Convert float-like string to float.

    Args:
        x (str): String representing an float.

    Returns:
        float: Float.
    """
    x = x.split('.')
    if x[0].startswith('1'):
        if len(x[0]) > 2:
            return float(x[0] + '.' + x[1])
        else:
            x = x[0]+x[1]
            return float(x[0:3] + '.' + x[3:])
    else:
        if len(x[0]) > 2:
            return float(x[0][0:2] + '.' + x[0][-1])
        else:
            x = x[0] + x[1]
            return float(x[0:2] + '.' + x[2:])


def datetime_to_unix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Receive a DataFrame column with datetimes and convert them to unix timestamps.

    Args:
        df (pd.DataFrame): Input DataFrame (column with datetime).

    Returns:
        pd.DataFrame: Output DataFrame (column with unix timestamp).
    """
    return (df - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
