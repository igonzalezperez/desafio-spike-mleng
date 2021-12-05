"""
DOCSTRING: TODO
"""
# %% Imports
import warnings
from typing import List
import numpy as np
import pandas as pd
import dateparser
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from loguru import logger
from logger_config import logger_config
from dotenv import load_dotenv

load_dotenv()
logger_config(level=os.getenv('LOG_LEVEL'))
pd.options.mode.chained_assignment = None  # default='warn'
plt.style.use('seaborn-notebook')
warnings.filterwarnings(
    "ignore", message="The localize method is no longer necessary, as this time zone supports the fold attribute",)


# %% Misc functions


def convert_int(x):
    return int(x.replace('.', ''))


def to_100(x):  # mirando datos del bc, pib existe entre ~85-120 - igual esto es cm (?)
    x = x.split('.')
    if x[0].startswith('1'):  # es 100+
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


# %% Data Ingestion
def ingest_data():
    # Rain data
    rain = pd.read_csv('data_science/data/precipitaciones.csv')  # [mm]
    central_bank = pd.read_csv('data_science/data/banco_central.csv')
    milk_price = pd.read_csv('data_science/data/precio_leche.csv')
    # rain['date'] = pd.to_datetime(rain['date'], format='%Y-%m-%d')
    # rain = rain.sort_values(by='date', ascending=True).reset_index(drop=True)
    # rain[rain.isna().any(axis=1)]  # no tiene nans
    # rain[rain.duplicated(subset='date', keep=False)]  # ni repetidos

    # rain[regions].describe()
    # rain.dtypes

    # Central bank data

    # central_bank['Periodo'] = central_bank['Periodo'].apply(lambda x: x[0:10])
    # central_bank['Periodo'] = pd.to_datetime(
    #     central_bank['Periodo'], format='%Y-%m-%d', errors='coerce')
    # central_bank[central_bank.duplicated(
    #     subset='Periodo', keep=False)]  # repetido se elimina
    # central_bank.drop_duplicates(subset='Periodo', inplace=True)
    # central_bank = central_bank[~central_bank.Periodo.isna()]
    # cols_pib = [x for x in list(central_bank.columns) if 'PIB' in x]
    # cols_pib.extend(['Periodo'])
    # central_bank_pib = central_bank[cols_pib]
    # central_bank_pib = central_bank_pib.dropna(how='any', axis=0)

    # for col in cols_pib:
    #     if col == 'Periodo':
    #         continue
    #     else:
    #         central_bank_pib[col] = central_bank_pib[col].apply(
    #             lambda x: convert_int(x))

    # central_bank_pib.sort_values(by='Periodo', ascending=True)

    return rain, central_bank, milk_price


def validate_data(data, cols: List[str] = None, metadata: bool = False, duplicate_col: str = None):
    if cols:
        for col in cols:
            assert col in data.columns, f'Input data is missing required column {col}.'
    nan_data = {}
    for col in data:
        nan_data[col] = data[col].isnull().sum()
    duplicates = data[data.duplicated(
        subset=duplicate_col, keep=False)] if duplicate_col else None
    return pd.DataFrame(nan_data, index=[0]), duplicates


logger.debug('Data ingestion - Rain, Central bank and milk price.')
rain, central_bank, milk_price = ingest_data()
data = {'rain': rain, 'central_bank': central_bank, 'milk_price': milk_price}
regions = ['Coquimbo', 'Valparaiso', 'Metropolitana_de_Santiago',
           'Libertador_Gral__Bernardo_O_Higgins', 'Maule', 'Biobio',
           'La_Araucania', 'Los_Rios']
required_cols = {'rain': regions, 'central_bank': None, 'milk_price': None}
duplicate_cols = {'rain': 'date',
                  'central_bank': 'Periodo', 'milk_price': None}

for d, c in zip(data, required_cols):
    logger.debug(f'{d} data validation.')
    nan_data, dups = validate_data(data[c], required_cols[c])
    nans = nan_data.sum().sum()
    if nans != 0:
        logger.debug(f'{nans} Nan datapoints found.')
        # for col in data[c]:
        #     logger.debug(f'\n{nan_data[col].name}\n{nan_data[col].values[0]}')
    else:
        logger.debug(f'No Nan values found.')
    if dups:
        logger.debug(f'{dups} duplicate rows found.')
    else:
        logger.debug(f'No duplicate rows found.')

cols_pib = [c for c in central_bank.columns if 'PIB' in c]
cols_imacec = [x for x in list(central_bank.columns) if 'Imacec' in x]
central_bank = central_bank.drop_duplicates(subset='Periodo')
central_bank = central_bank[~central_bank['Periodo'].isna()]
central_bank = central_bank[['Periodo', *cols_pib, *cols_imacec,
                             'Indice_de_ventas_comercio_real_no_durables_IVCM']]
central_bank = central_bank.dropna(how='any', axis=0)
for col in central_bank:
    if col in cols_pib:
        central_bank[col] = central_bank[col].apply(lambda x: convert_int(x))
    elif col in cols_imacec:
        central_bank[col] = central_bank[col].apply(lambda x: to_100(x))
        assert central_bank[col].max() > 100
        assert central_bank[col].min() > 30
central_bank['num'] = central_bank['Indice_de_ventas_comercio_real_no_durables_IVCM'].apply(
    lambda x: to_100(x))
central_bank = central_bank.sort_values(by='Periodo', ascending=True)
breakpoint()
# %% [markdown]
# #### Observaciones
# - Los datos de rain vienen limpios y completos.
# - Llueve más en el sur que en el norte.

# %% [markdown]
# ### Banco central

# %%


# %% [markdown]
# #### Observaciones
# - Datos no vienen como numero, si no, como string con puntos.
# - Hay varios datos mezclados: PIB, IMACEC, índices de producción industrial, índice de ventas, etc.. (abajo una gugleada de los términos)
# - Fechas no válidas se eliminan.
# - Para poder saber el valor 'real' de los datos, depende de cada una de los grupos de las categorías. Por ejemplo, PIB pareciera ser suficiente convertirlo a entero, mientras que con el IMACEC hay que jugar un poco más (https://www.bcentral.cl/web/banco-central/areas/estadisticas/imacec) -ideal sería preguntarle al cliente la fuente de sus datos, para no estar adivinando valores que pueden o no estar correctos-
# - Grupos que pueden ser interesantes, dado el objetivo de predecir $ leche
#     - Datos IMACEC fechas: 1996-2020
#     - Datos PIB fechas: 2013-2020
#     - Indice de ventas bienes no durables (leche) 2014-2020
#     - Varios otros: ocupación en areas relacionadas, generación energía eléctrica? (será relevante la electricidad q se usa en la producción de leche?, precios de algunas cosas relacionadas (petroleo/bencina = transporte leches? camiones?)

# %% [markdown]
# - IMACEC = El Índice Mensual de Actividad Económica (Imacec) es una estimación que resume la actividad de los distintos sectores de la economía en un determinado mes, a precios del año anterior; **su variación interanual constituye una aproximación de la evolución del PIB**. El cálculo del Imacec se basa en múltiples indicadores de oferta que son ponderados por la participación de las actividades económicas dentro del PIB en el año anterior. -- unidades = , en general se ve la variación del IMACEC de un periodo a otro
# - PIB = expresa el valor monetario de la producción de bienes y servicios de demanda final de un país o región durante un período determinado, normalmente de un año o trimestrales. (1960 fue de 4,1 millones de USD - 2019 fue de 282,3 miles de millones USD)
# - Derechos de importacion: También denominado derecho aduanero, por tratarse de un impuesto que cobra la aduana de un país para permitir el ingreso de mercancías al territorio nacional.
# - Impuesto al valor agregado (IVA)
# - Precio de ..
# - Tipo de cambio (USD a CLP)
# - Ocupados/Ocupacion en ../No responde-no sabe/: Tasa empleo - Encuesta Nacional de Empleo (ENE) ?
# - Indices de produccion industrial (x): Es un indicador analítico que tiene por fin mostrar la evolución de la producción física de tres sectores de la economía: Minería, Manufactura y Electricidad, Gas y Agua (EGA).
# - Generacion energia electrica
# - Indice de ventas (IVCM = indice ventas comercio al por menor) [$]
#  * un bien duradero se define como aquel producto que una vez adquirido puede ser utilizado un gran número de veces a lo largo del tiempo. Los bienes duraderos son aquellos bienes reutilizables y que, aunque pueden acabar gastándose, no se consumen rápidamente como los bienes no duraderos (**leche = no duradero**)
# - Venta autos nuevos


cols_imacec = [x for x in list(central_bank.columns) if 'Imacec' in x]
cols_imacec.extend(['Periodo'])
central_bank_imacec = central_bank[cols_imacec]
central_bank_imacec = central_bank_imacec.dropna(how='any', axis=0)

for col in cols_imacec:
    if col == 'Periodo':
        continue
    else:
        central_bank_imacec[col] = central_bank_imacec[col].apply(
            lambda x: to_100(x))
        assert(central_bank_imacec[col].max() > 100)
        assert(central_bank_imacec[col].min() > 30)

central_bank_imacec.sort_values(by='Periodo', ascending=True)
central_bank_imacec

# %%
central_bank_iv = central_bank[[
    'Indice_de_ventas_comercio_real_no_durables_IVCM', 'Periodo']]
central_bank_iv = central_bank_iv.dropna()  # -unidades? #parte
central_bank_iv = central_bank_iv.sort_values(by='Periodo', ascending=True)
breakpoint()
# %%
# unidades? https://si3.bcentral.cl/siete/ES/Siete/Canasta?idCanasta=M57TP1161519 porcentajes?
central_bank_iv.head()

# %%
central_bank_iv['num'] = central_bank_iv.Indice_de_ventas_comercio_real_no_durables_IVCM.apply(
    lambda x: to_100(x))

# %%
central_bank_iv.Periodo.min()

# %%
central_bank_iv.Periodo.max()

# %%
central_bank_num = pd.merge(
    central_bank_pib, central_bank_imacec, on='Periodo', how='inner')
central_bank_num = pd.merge(
    central_bank_num, central_bank_iv, on='Periodo', how='inner')

# %% [markdown]
# ## Visualización

# %% [markdown]
# ### Funciones

# %%


def visualizacion_ppes(df_ppes, region, fecha_i, fecha_f, col_fechas='date'):
    '''
    df_ppes = df con datos de preciptaciones
    region = region que se quiere graficar (string)
    fecha_i/fecha_f = fecha inicial y final de periodo a analizar. En un string de formato 'dd-mm-yyyy' 
    col_fechas = columna donde estan las fechas en el df de rain
    '''
    try:
        df_ppes_region = df_ppes[[col_fechas, region]]
    except:
        print('Región no en datos')
        return np.nan

    try:
        fecha_i = pd.to_datetime(fecha_i, format='%d-%m-%Y')
        fecha_f = pd.to_datetime(fecha_f, format='%d-%m-%Y')
        df_ppes_region_fechas = df_ppes_region[(df_ppes_region[col_fechas] >= fecha_i) & (
            df_ppes_region[col_fechas] <= fecha_f)]
    except:
        print('Revisar fechas')
        return np.nan

    # fig, ax = plt.subplots(figsize=(16, 8))
    # ax.plot(df_ppes_region_fechas[col_fechas], df_ppes_region_fechas[region])
    # ax.set_title(
    #     f"rain en región {' '.join(region.split('_')).title().strip()}")
    # ax.set_ylabel('rain [mm]')
    # ax.set_xlabel('Fecha')

    # ax.grid(True)
    # plt.show()


def rain_ano(df_ppes, anos, region, col_fechas='date'):
    df_ppes['ano'] = df_ppes[col_fechas].apply(lambda x: x.year)
    df_ppes['mes'] = df_ppes[col_fechas].apply(lambda x: x.strftime("%B"))

    try:
        rain_region = df_ppes[['ano', 'mes', col_fechas, region]]
    except:
        print('Región no en datos')
        return np.nan

    try:
        rain_region_ano = rain_region[rain_region.ano.isin(
            anos)]
    except:
        print('Revisar lista de años')
        return np.nan

    # fig, ax = plt.subplots(figsize=(16, 8))
    # for ano in anos:
    #     rain_ano = rain_region_ano[rain_region_ano.ano == ano]
    #     ax.plot(rain_ano['mes'],
    #             rain_ano[region], label=ano)
    # ax.set_title(
    #     f"rain en región {' '.join(region.split('_')).title().strip()}")
    # ax.set_ylabel('rain [mm]')
    # ax.set_xlabel('Mes')
    # ax.grid(True)
    # ax.legend()


def series_pib(df, serie1, serie2, fecha_i, fecha_f, col_fecha):
    central_bank_pib_ = df[(df[col_fecha] >= fecha_i)
                           & (df[col_fecha] <= fecha_f)]
    central_bank_pib_serie = central_bank_pib_[[serie1, serie2, col_fecha]]
    central_bank_pib_serie = central_bank_pib_serie.sort_values(
        by=col_fecha, ascending=True)
    series = [serie1, serie2]
    # fig, ax = plt.subplots(figsize=(16, 8))
    for serie in series:
        central_bank_pib_selec = central_bank_pib_serie[[serie, 'Periodo']]
        # ax.plot(central_bank_pib_selec['Periodo'], central_bank_pib_selec[serie], label=serie.replace(
        #     '_', ' ').replace('PIB', '').strip())
    # ax.set_title(
    #     f"PIB entre {fecha_i} y {max(central_bank_pib_selec.Periodo)}")
    # ax.set_ylabel('$')
    # ax.set_xlabel('Periodo')
    # ax.grid(True)
    # ax.legend()

# %% [markdown]
# ### Resultados


# %%
# rain en Ohiggins vs Metropolitana
fecha_i = '01-01-2000'
fecha_f = '01-01-2020'

visualizacion_ppes(
    rain, 'Libertador_Gral__Bernardo_O_Higgins', fecha_i, fecha_f)
visualizacion_ppes(
    rain, 'Metropolitana_de_Santiago', fecha_i, fecha_f)

# %% [markdown]
# Observaciones
# - En general las rain han disminuido en Chile (</3)
# - Región O'Higgins llueve en mayor proporción que en la región Metropolitana
# - Ambas regions
# tienen la misma estacionalidad de lluvia/no lluvia

# %%
# rain para la región del maule durante 1982, 1992, 2002, 2012, 2019
rain_ano(rain, [1982, 1992, 2002, 2012, 2019], 'Maule')

# %% [markdown]
# Observaciones
# - El mes más lluvioso varía entre mayo, junio, julio, agosto (invierno).A medida que pasa el tiempo, el mes más lluvioso se ha ido desplazando desde mayo-junio a junio-julio
# - Se observa que  medida que pasa el tiempo, llueve menos en Chile (</3) (volumen total como el maximo de mm/mes)
# - La lluvia en Chile ha cambiado con el paso del tiempo y las estaciones no están tan marcadas (esto igual es conocimiento por vivir en Chile jaja). Por ejemplo, el 2002 llovió lo mismo en agosto-octubre y en diciembre(!), mientras que en los años anteriores y los que le siguen, las lluvias en diciembre son bajas (verano). Lo mismo al mirar el 2012, es el único año donde llueve más de 100 mm en abril. En el 2019 el valor máximo no pasa de los 200+ mm.

# %%
# visualizar dos series hcas de PIB >=2013-01-01
serie1 = 'PIB_Agropecuario_silvicola'
serie2 = 'PIB_Servicios_financieros'
fecha_i = '2013-01-01'
fecha_f = '2021-04-01'
col_fecha = 'Periodo'

series_pib(central_bank_pib, serie1, serie2, fecha_i, fecha_f, col_fecha)


# %% [markdown]
# Observaciones
# - No sé si las caídas/subidas bruscas corresponden a outliers o si los datos son así (en particular de servicios financieros). Acá sería bueno poder ver la serie mensual para ver los valores (solo he encontrado la serie trimestral/anual) y si fuera el caso, eliminar outliers e interpolar la serie (Tomando como supuesto que sin son outliers, se podría observar que el PIB de servicios financieron ha ido creciendo a lo largo del tiempo)
# - Si no son outliers, pareciera ser que el PIB mensual varía bastante entre meses. Quizás por eso se presenta en semestral/trimestral?
# - La serie de agropecuario/silvicola tambén varía bastante mes a mes, pero es una variación más suave(?), podría pensarse que cumplen con algún tipo de periodicidad (estaciones del año y productos provenientes de seres vivos?).

# %% [markdown]
# ## Tratamiento y creación de varaiables

# %% [markdown]
# ### Variable a predecir: precio leche

# %%
milk_price = pd.read_csv('data_science/data/precio_leche.csv')
# precio = nominal, sin iva en clp/litro
milk_price.rename(columns={'Anio': 'ano', 'Mes': 'mes_pal'}, inplace=True)
milk_price['mes'] = milk_price['mes_pal'].apply(
    lambda x: dateparser.parse(x))
milk_price['mes'] = milk_price['mes'].apply(lambda x: x.month)
milk_price['mes-ano'] = milk_price.apply(
    lambda x: f'{x.mes}-{x.ano}', axis=1)
milk_price.head()

# %%
# milk_price.plot(x='mes-ano', y='milk_price')  # alza 2010-2011?

# %%
milk_price[milk_price.ano >= 2013].plot(x='mes-ano', y='milk_price')

# %%
rain['mes'] = rain.date.apply(lambda x: x.month)
rain['ano'] = rain.date.apply(lambda x: x.year)
milk_price_pp = pd.merge(milk_price, rain, on=[
    'mes', 'ano'], how='inner')
milk_price_pp.drop('date', axis=1, inplace=True)
milk_price_pp  # rain fecha_max = 2020-04-01

# %%
central_bank_num['mes'] = central_bank_num['Periodo'].apply(lambda x: x.month)
central_bank_num['ano'] = central_bank_num['Periodo'].apply(lambda x: x.year)
milk_price_pp_pib = pd.merge(milk_price_pp, central_bank_num, on=[
    'mes', 'ano'], how='inner')
milk_price_pp_pib.drop(['Periodo', 'Indice_de_ventas_comercio_real_no_durables_IVCM',
                        'mes-ano', 'mes_pal'], axis=1, inplace=True)

# %% [markdown]
# #### Correlación entre variables
# Se puede ver la matriz de correlación para observar la correlación entre variables (graficada más abajo). Intuitivamente, una creería que si dos variables están correlacionadas no aportan información distinta y, por lo tanto, convendría tener solo una de ellas..pero yo creo que todo depende de los datos y de que modelo/suposiciones estamos usando.
#

# %%

cc_cols = [x for x in milk_price_pp_pib.columns if x not in ['ano', 'mes']]
# Compute the correlation matrix
corr = milk_price_pp_pib[cc_cols].corr()

# Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr,  cmap=cmap,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})


# %% [markdown]
# - Se observa que las variables de lluvia están correlacionadas entre si y que las variables del imacec + indice de ventas están correlacionadas entre sí.
# - El precio de la leche tiene una correlación levemente positiva con la lluvia en las regions
# del sur (tiene sentido dado que allá están las vacas?), pero igual no es muy alta (probablemnte pqe el proceso de la leche está super industrializado), con el PIB de manufactura y de alimentos, PIB del petróleo, PIB de servicios financieros, IMACEC empalmado (la producción de leche es una entrada del IMACEC) y de IMACEC de algunos sectores (servicios, costo de factores y no minero). Sería interesante averiguar más sobre estos indicadores para saber si realmente se relacionan con el precio de la leche.
#

# %% [markdown]
# Con respecto al modelo a usar y la correlación de las variables, se construirán ciertos indicadores para el precio de la leche en el mes n y luego todas esas características serán pasadas a un selector de características. Esto para evitar el sesgo humano de elegir (más pqe yo no tengo experiencia ni conocimiento experto como para elegir las variables correctas)

# %% [markdown]
# #### Variables a usar: se eligen las variables relacionadas con el PIB del BC, el IMACEC empalmado, el índice de ventas de comercio real no durable, las rain y el precio de la leche (n-1) --> Para todas estas se utiliza el indicador del mes anterior y mean/std de 3 meses acumulados

# %%
cc_cols = [x for x in milk_price_pp_pib.columns if x not in ['ano', 'mes']]

# %%
milk_price_pp_pib_shift3_mean = milk_price_pp_pib[cc_cols].rolling(
    window=3, min_periods=1).mean().shift(1)

milk_price_pp_pib_shift3_mean.columns = [
    x+'_shift3_mean' for x in milk_price_pp_pib_shift3_mean.columns]

milk_price_pp_pib_shift3_std = milk_price_pp_pib[cc_cols].rolling(
    window=3, min_periods=1).std().shift(1)

milk_price_pp_pib_shift3_std.columns = [
    x+'_shift3_std' for x in milk_price_pp_pib_shift3_std.columns]

milk_price_pp_pib_shift1 = milk_price_pp_pib[cc_cols].shift(1)

milk_price_pp_pib_shift1.columns = [
    x + '_mes_anterior' for x in milk_price_pp_pib_shift1.columns]


# %%
milk_price_pp_pib = pd.concat([milk_price_pp_pib['milk_price'], milk_price_pp_pib_shift3_mean,
                               milk_price_pp_pib_shift3_std, milk_price_pp_pib_shift1], axis=1)
milk_price_pp_pib = milk_price_pp_pib.dropna(how='any', axis=0)
milk_price_pp_pib.head()

# %%
milk_price_pp_pib[['milk_price', 'milk_price_mes_anterior']]

# %% [markdown]
# ## Modelo

# %% [markdown]
# - Se elige una separación de 80-20 para el test/train

# %% [markdown]
# ### Regresión utilizando las variables del precio de la leche de periodos anteriores (y variables climatológicas y macroeconómicas)

# %%
X = milk_price_pp_pib.drop(['milk_price'], axis=1)
y = milk_price_pp_pib['milk_price']

# %%
y.mean()

# %%
y.std()

# %%
# imports

# generate random data-set
np.random.seed(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

pipe = Pipeline([('scale', StandardScaler()),
                 ('selector', SelectKBest(mutual_info_regression)),
                 ('poly', PolynomialFeatures()),
                 ('model', Ridge())])
k = [3, 4, 5, 6, 7, 10]
alpha = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
poly = [1, 2, 3, 5, 7]
grid = GridSearchCV(estimator=pipe,
                    param_grid=dict(selector__k=k,
                                    poly__degree=poly,
                                    model__alpha=alpha),
                    cv=3,
                    scoring='r2')
grid.fit(X_train, y_train)
y_predicted = grid.predict(X_test)

# evaluar modelo
rmse = mean_squared_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

# printing values
print('RMSE: ', rmse)
print('R2: ', r2)

# %%
grid.best_params_

# %%
X_train.columns[grid.best_estimator_.named_steps['selector'].get_support()]

# %%
predicted = pd.DataFrame(y_test).reset_index(drop=True)
predicted['predicc'] = y_predicted
predicted = predicted.reset_index()
# plt.scatter(predicted.index, predicted['milk_price'], label='real')
# plt.scatter(predicted.index, predicted['predicc'],
#             color='red', label='prediccion', alpha=0.5)
# plt.grid(axis='x')
# plt.legend()

# %%
predicted['residual'] = predicted.milk_price - predicted.predicc
# plt.hlines(0, xmin=predicted.predicc.min()-10, xmax=predicted.predicc.max() +
#            10, linestyle='--', color='black', linewidth=0.7)
# plt.scatter(predicted.predicc, predicted.residual)
# plt.xlabel('Predicción')
# plt.ylabel('Residuo (y_real - y_pred)')

# %% [markdown]
# ### Regresión utilizando solamente variables macroeconómicas y climatológicas

# %%
# generate random data-set
np.random.seed(0)
cols_no_leche = [x for x in list(X.columns) if not ('leche' in x)]
X_train = X_train[cols_no_leche]
X_test = X_test[cols_no_leche]

pipe = Pipeline([('scale', StandardScaler()),
                 ('selector', SelectKBest(mutual_info_regression)),
                 ('poly', PolynomialFeatures()),
                 ('model', Ridge())])
k = [3, 4, 5, 6, 7, 10]
alpha = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
poly = [1, 2, 3, 5, 7]
grid = GridSearchCV(estimator=pipe,
                    param_grid=dict(selector__k=k,
                                    poly__degree=poly,
                                    model__alpha=alpha),
                    cv=3,
                    scoring='r2')
grid.fit(X_train, y_train)
y_predicted_noleche = grid.predict(X_test)

# evaluar modelo
rmse = mean_squared_error(y_test, y_predicted_noleche)
r2 = r2_score(y_test, y_predicted_noleche)

# printing values
print('RMSE: ', rmse)
print('R2: ', r2)

# %%
grid.best_params_

# %%
X_train.columns[grid.best_estimator_.named_steps['selector'].get_support()]

# %%
predicted = pd.DataFrame(y_test).reset_index(drop=True)
predicted['predicc'] = y_predicted_noleche
predicted = predicted.reset_index()
# plt.scatter(predicted.index, predicted['milk_price'], label='real')
# plt.scatter(predicted.index, predicted['predicc'],
#             color='red', label='prediccion', alpha=0.5)
# plt.grid(axis='x')
# plt.legend()

# %%
predicted['residual'] = predicted.milk_price - predicted.predicc
# plt.hlines(0, xmin=predicted.predicc.min()-10, xmax=predicted.predicc.max() +
#            10, linestyle='--', color='black', linewidth=0.7)
# plt.scatter(predicted.predicc, predicted.residual)
# plt.xlabel('Predicción')
# plt.ylabel('Residuo (y_real - y_pred)')

# %% [markdown]
# ### Observaciones/Respuestas

# %% [markdown]
# - Métricas que tiene sentido mirar: creo que tiene sentido mirar el rmse para saber por cuanto se equivoca en promedio las predicciones de los datos reales y el r2 para ver que tal anda la regresión. También tiene sentido ver qué características fueron seleccionadas y su importancia para la predicción, así quizás se puede ir entendiendo más.
#
# - Al observar las variables seleccionadas en cada una de las regresiones, se observa que en la primera se selecciona: precio leche mes anterior, promedio precio leche 3 meses anteriores, pib precio del petróleo promedio de 3 meses (transporte leches y alimento/insumos para las vacas?) y pib admnistración pública (???). Para el segundo modelo, se observa que se eligen 10 variables: 9 que tienen que ver con el PIB promedio de 3 meses atrás y el valor del mes anterior y el que se relaciona con el promedio de 3 meses del índice de ventas de productos no duraderos (se llama `num`). Llama la atención que el PIB de la zona agropecuaria no está presente en ninguno de los dos (la leche será un % muy bajo?). Las rain de ninguna región aparecen. En ambas regresiones parece ser relevante el PIB que tiene que ver con el precio del petróleo, lo que tiene sentido dado que -supongo- las leches se transportan desde el sur de Chile hacia la zona centro y norte en camiones.
#
# - Evaluación del modelo: se obtiene una regresión con r2 de 0.85 y rmse de 88.8, utilizando como input el precio de la leche en periodos de tiempo pasado. Esto quizás va un poco en contra del objetivo del problema de 'estimar el precio de un producto usando variables climatológicas y macroeconómicas'. Al sacar estas variables, se observa que baja el r2 (a 0.37) y sube el rmse (a 389.6), por lo que realmente no se estaría logrando el objetivo. Quizás se podría tunear más esta regresión y mejorar un poco los resultados.
#
# - El valor promedio del precio de la leche es de 230.9 con una desviación estándar de 23.68. Claramente el modelo que utiliza precio de leches anteriores tiene un mejor fit que el que no usa, pero creo que determinar si es 'buena' o 'mala' depende del objetivo para el cual se haya construido esta regresión.
#
# - Datos adicionales:
#     - Me gustaría entender más como se pone el precio de la leche en Chile y tener variables de acuerdo a eso. No sé si las rain influyan tanto, dado que las vacas igual crecen si llueve? Además que no es como que el proceso se vea afectado si es que llueve o por las estaciones del año, google me dice que las vacas tienen un celo con un periodo corto (como las mujeres humanas) y la inseminisación es artificial (para que puedan producir leche), por lo que tiene sentido que las variables relacionadas con las rain no aparezcan en las regresiones. Quizás tendría sentido tener variables climatológicas que afecten realmente a las vacas (por ejemplo, la comida de las vacas se produce en Chile? Si es así, se vería afectada por alguna razón del clima? Hubo un desastre natural y muchas vacas se murieron?)
#     - No sé mucho de macroeconomiea, pero sería interesante estudiar más las variables del banco central, para agregar otras que puedan influir y para saber si las suposiciones que se realizaron al pasarlas de string a número son correctas o si le metí cualquier valor. También al tener en cuenta el PIB y el IMACEC habría que ver si estos indicadores se ven afectados por la venta de la leche y si la producción es relevante para estos indicadores. Leí un poco y el IMACEC tiene como input (entre varios otros -no sé si esto es lo actual https://si3.bcentral.cl/estadisticas/principal1/metodologias/ccnn/imacec/serieestudios48.pdf [p20-22] -) la producción de leche, pero dice que es solo el 4.2% por lo que quizás usar el IMACEC no es lo ideal, quizás se podría probar con el IMACEC solo de las actividades agropecuarias y agrícolas? O quizás añadir variables más locales que tengan que ver con las vacas (la electricidad que usa una planta lechera es relevante? El agua que toman las vacas? Los medicamentos que les dan? Qué gastos/insumos conlleva la producción de la leche?)
#
# - Me cuesta ver como este modelo ayudaría al calentamiento global la verdad. Quizás mostrando que el precio de la leche va en alza y que conviene cambiarse a algo más amable y no dependiente de vacas?
