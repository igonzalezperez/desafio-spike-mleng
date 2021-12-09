import pandas as pd


def convert_int(x):
    return int(x.replace('.', ''))


def to_100(x):
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


def datetime_to_unix(df):
    return (df - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
