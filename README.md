# Desafío ML Engineer - Spike - 2021/12

Este proyecto contiente el código necesario para abordar el desafío para el puesto de Machine Learning Engineer en Spike. Para ello se desarrolló una Web App en Flask para predecir el precio de la leche en base a indicadores climatológicos, macroeconómicos y precios anteriores de la leche.

# Cómo correr la Web App localmente
Primero que todo es necesario tener instalado [`docker`](https://docs.docker.com/get-docker/) y [`docker compose`](https://docs.docker.com/compose/install/).

Para correr la API en un server va a ser necesario abrir la consola. En Windows esto se puede hacer con `Win + R` lo cual abrira una ventana en la que se debe escribir `cmd` y dar `Enter`. En otros sistemas operativos se puede abrir de manera similar.

<p align="center">
    <img src=".etc/open_cmd.jpg" width="300"/>
</p>

Esto abre la consola. Luego se deben seguir los siguientes pasos:
## 1. Clonar repositorio
El proyecto se puede descargar directamente como un .zip y descomprimirlo localmente. Alterenativamente Si se tiene `git` instalado, se puede clonar usando la consola:

```
git clone https://github.com/igonzalezperez/desafio-spike-mleng.git
``` 

<p align="center">
    <img src=".etc/clone_repo.jpg" width="600"/>
</p>

## 2. Crear contenedor en Docker
Para correr la aplicación en un servidor local se debe ejecutar el siguiente comando en la carpeta en que esté contenido el proyecto:

```
docker compose up -d --build
```

<p align="center">
    <img src=".etc/docker_compose.jpg" width="600"/>
</p>

Con esto `docker` comenzará a crear las imágenes y el contenedor para disponibilizar la app. Esto podría tomar unos minutos ya que al iniciar se optimizan los parámetros del modelo de regresión y y estos se guardan para que luego la API pueda acceder a el.

## 3. Navegar a la App
El comando anterior tomará unos segundos (~30s) en crear el contenedor, una vez terminado el proceso abrir un navegador e ir a la dirección: 
```
http://localhost/
```
En esa dirección debería aparecer el sitio que soporta la API que se verá así:

<p align="center">
    <img src=".etc/api_landing.jpg" width="600"/>
</p>

# Cómo usar la web app
## Predicción - Inferencia
Para generar predicciones basta con introducir el mes a predecir o bien un intervalo de meses (batch) en la caja de texto, con lo cual se ejecutará el modelo para esa(s) fecha(s) y se mostrarán los resultados en pantalla.

<p align="center">
    <img src=".etc/pred_1.jpg" width="500"/>
</p>

<p align="center">
    <img src=".etc/pred_2.jpg" width="500"/>
</p>

Los meses se introducen en formato `YYYY-MM` y son separados por un espacio en caso de predicción por batch.

La app muestra tanto los valores predecidos como los reales si es que estos ya existen en la base de datos (el modelo de datos se explica en la siguiente sección).

Si el input no es correcto, la app tirará un error y no se ejecutará nada. Otro caso de error ocurre cuando se busca predecir un mes para el cual no existe data suficiente para predecir, ya que para predecir el tiempo T + 1, se necesita data de T, T-1 y T-2. En cuyo caso se indicará qué meses faltan.

<p align="center">
    <img src=".etc/pred_3.jpg" width="500"/>
</p>

## Base de datos
El contenedor genera una base de datos con los archivos que se entregaron para el desafío (`precipitaciones.csv`, `banco_central.csv` y `precio_leche.csv`) los cuales tienen data suficiente para predecir entre `2014-04` y `2020-05`. Los meses de datos existentes se pueden observar al ir a `DB INFO` en la parte superior en donde se mostrará una lista con todos los meses.

Para poder hacer más predicciones futuras es posible cargar nuevos datos, se espera que estos tengan la misma estructura que los originales de precipitación, banco central y precio. En particular se requiere que tengan todas las columnas necesarias para la predicción y que no contengan Nans.

En la carpeta `/app/data/csv/`, además de los datos originales hay otros 3 archivos con el sufijo `_dummy`, los cuales se pueden utilizar para testear la funcionalidad de carga de datos.

Esta revisa que los archivos tengan el formato correcto y luego inserta las filas si es que estas no se encuentran en la BBDD. Luego, la app informa qupe columnas fueron insertadas tanto en las features (precipitación + banco central) como targets (precio de la leche).

<p align="center">
    <img src=".etc/dummy.jpg" width="500"/>
</p>

<p align="center">
    <img src=".etc/insert_data.jpg" width="500"/>
</p>

Como se ve en el ejemplo, ahora se cuenta con datos para poder predecir el mes `2020-06` que antes no se podía.

<p align="center">
    <img src=".etc/pred_new.jpg" width="500"/>
</p>

Adicionalmente, la base de datos se puede resetear al hacer click en la parte superior en donde aparece `Reset DB`. Esto recrea la BBDD original.

## Monitoreo
En la parte superior derecha de la app se encuentran los logs que permiten monitorear la app, estos se dividen en entrenamiento y predicción.
### Train logs
Información de grid search y entrenamiento. Se genera al crear el contenedor y no es modificado por el uso de la API. El output debería ser el siguiente:

<p align="center">
    <img src=".etc/trainlog.jpg" width="500"/>
</p>

Se ven los parámetros de optimización así como los resultados de entrenamiento. También está la opción de descargarlo como archivo de texto.

### Pred logs
El log de predicción va guardando todas las interacciones con la app en sí. Predicciones, inputs incorrectos, data insuficiente, update de la BBDD, etc y debería verse algo como esto:

<p align="center">
    <img src=".etc/predlog.jpg" width="500"/>
</p>

Es importante mencionar que cuando se inserta data a la BBDD, automáticamente se vuelve a predecir con los nuevos datos para recalcular el RMSE y R2, y así poder identificar si el desempeño mejora o empeora en el tiempo, pudiendo detectar fenómenos como data drifting. Al igual que el caso anterior, es posible descargar los logs como archivos de texto.


# Planteamiento del desafío
En esta sección detallaré cómo aborde el problema a un nivel técnico, las decisiones que tomé y el porqué de ellas.
## Tech stack
- **Backend**:
  - ``sqlite``: Para la base de datos, se utiliza porque es sencillo y rápido de usar para un proyecto pequeño. Para un proyecto que vaya a ser implementado debería migrarse a otra opción como ``PostgreSQL``, ``MySQL`` o ``MongoDB``.
  - ``python``: Se refactoriza el notebook original a archivos de `python` comunes.
- **API**:
  - ``Flask``: Se utiliza por su facilidad y porque ya tengo experiencia con este framework. También podrían considerarse Django o FastAPI.
- **Frontend**:
  - html/css + bootstrap: Utilicé templates para generar un front-end agradable a la vista.
  
## Modelo de datos
### Fuente de datos
Se tienen 3 fuentes de datos de archivos `.csv` los cuales primero se preprocesan para limpiar los datos.
#### `precipitaciones.csv`
- Datos de precipitación en las regiones de Chile: 
  - 9 columnas: Fecha (str) y 8 regiones de Chile (float).
  - 496 filas: Data mensual de 1979-01 a 2020-04. 0 Nans, 0 duplicados.
- Preprocesamiento:
  - Convertir columna 'date' (timestamp-str) a unix timestamp (float). Renombrar a 'timestamp'.
#### `banco_central.csv`
- 85 columnas: 
  - Periodo (str): Timestamp mensual.
  - 9 Columnas Imacec (str).
  - 28 Columnas PIB (str).
  - 1 Columna Indice de ventas comercio real no durables IVCM (str).
  - 46 otras columnas descartables.
- 614 filas:
  - 531 filas con Nans en alguna de sus features, 2 filas duplicadas.
- Data mensual desde 2014-01 a 2020-09 (81 filas)
- Preprocesamiento:
  - Convertir columna 'Periodo' (timestamp-str) a unix timestamp (float). Renombrar a 'timestamp'
  - Eliminar filas con Nans y con timestamp duplicados.
  - Seleccionar solo columnas (features) relevantes: Columnas con 'PIB' e 'Imacec' como substring y columna 'Indice_de_ventas_comercio_real_no_durables_IVCM', el resto se eliminan.
  - Convertir valores de columnas (features) de string a float/int ('2.5' -> 2.5)
#### `precio_leche.csv`
- 3 Columnas:
  - Anio (int).
  - Mes (str), e.g. 'Ene', 'Feb', etc.
  - Precio_leche (float): Variable a predecir.
- 506 filas: Data mensual desde 1979-01 a 2021-02. 0 Nans, 0 duplicados.
- Preprocesamiento:
  - Combinar columnas 'Anio'(float) y 'Mes'(str) a unix timestamp (float). Nombrar nueva columna 'timestamp' y eliminar las otras dos.
  
### Base de datos
Una vez limpiados los datos estos se ingresan a una base de datos de `sqlite3`. En específico se divide el input del output.
- Tabla `features`: Inner join entre data de precipitación y banco central.
- Tabla `target`: Data del precio de la leche.

De este modo, la data puede ser consumida por algún modelo o continuar siendo procesada accediendo a la BBDD mediante queries. También es posible extender la cantidad de datos y dejar de depender de los `.csv`.

## Modelo de ML
### Procesamiento
Primero se hace un inner join entre la tabla `features` y `targets`, para que coincidan las fechas a predecir.

Luego, para generar el dataset que luego se usará para entrenar modelos, se realiza un procesamiento extra que toma en cuenta datos anteriores para predecir el siguiente, ya que el problema se plantea como una regresión.

En particular se calculan los promedios y desviaciones estándar de cada columna para los últimos 3 meses como medias móviles, incluyéndo el precio de la leche, por lo que si el precio de la leche en un tiempo `t` es `Y[t]` y las features son `X[t]`, la predicción se realiza mediante:

```
Y[t+1] = Modelo(X[t, t-1, t-2],
                mean(X[t, t-1, t-2]),
                std(X[t, t-1, t-2]),
                Y[t, t-1, t-2],
                mean(Y[t, t-1, t-2]),
                std(Y[t, t-1, t-2]),)
```
### Optimización
Luego del procesamiento se utiliza grid search para optimizar los parámetros del siguiente Pipeline:

| Pipeline step      | Grid                                              |
| ------------------ | ------------------------------------------------- |
| StandardScaler     | -                                                 |
| SelectKBest        | k = 3, 4, 5, 6, 7, 10                             |
| PolynomialFeatures | poly__degree = 1, 2, 3, 5, 7                      |
| Ridge              | model__alpha = 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01 |

En la práctica se tienen dos pipelines, uno con todos los transformadores/selectores de características y otro únicamente con el modelo.
#### Parámetros óptimos
- `model__alpha = .1`
- `poly__degree = 1`
- `selector__k = 7`
- `RMSE = 12.54`
- `R2 = 0.83`
  
El modelo y pipelines óptimos se guardan como archivos `.pkl` para luego ser utilizados en la API. Los procesos de limpieza y preprocesamiento (medias móviles) de secciones anteriores no necesitan guardarse ya que no tienen un método `.fit()`, solo transforman.

## API
La API se desarrolla usando `Flask`. Dado que el problema de regresión requiere de datos anteriores para poder predecir, el input para una sola predicción deberían ser 3 filas correspondientes a los meses inmediatamente anteriores al mes de la predicción. Sin embargo consideré que esto podía resultar engorroso, es por eso que separé la parte de carga de datos de la de predicción. Para predecir basta con escribir una fecha, mientras que la base de datos podría mantenerse periódicamente, sin que el usuario se preocupe de si el input es correcto o no.

### **/endpoints**
- **/predict**:
  - Recibe un string con las fechas a predecir. Puede ser una predicción o por batches. El formato es `YYYY-MM`. Dos fechas separadas por un espacio se procesan como batch incluyendo todos los meses intermedios.
  - Retorna un dataframe con las fechas, predicciones y valores reales si es que existen.
- **/insert_data**:
  - Recibe los archivos `.csv` para ser insertados en la BBDD. Se espera que sean 3 archivos con el mismo fromato que la fuente de datos original del desafío.
  - Inserta todas las filas que aun no existan en la BBDD existente.
  - Calcula el RMSE y R2 considerando los datos nuevos.
  - Actualiza el archivo `db_span.json` que contiene los meses de data que hay disponibles.
- **/db_info**:
  - Muestra el contenido de `db_span.json` como una tabla.
- **/reset-db**:
  - Recrea la BBDD con los datos originales. Esto lo incluí para testear funcionalidades, en general no sería muy buena idea dejar que el usuario borre datos directamente.
- **/logs/train**:
  - Muestra los logs de entrenamiento.
- **/logs/pred**:
  - Muestra los logs de predicción.
- **/get-logs/<log_name>**:
  - Descarga el archivo de log `<log_name>` (e.g. `/get-logs/train.log`).

## Contenedores
La aplicación se encapsula en imágenes/contenedores usando Docker. Para que la app sea escalable se genera una imagen de la app en sí, y otra con `nginx` que funciona como web server y puede ser utilizado como load balancer para escalar la app y manejar el tráfico en ella. En la sección se cómo correr el server mostré que el contenedor se puede montar con

```
docker compose up -d --build
```

para poder escalar bastaría con agregar

```
docker compose up -d --build --scale app=5
```

donde el último parámetro índica cuántas copias de la app se disponibilizarán y `nginx` luego se encarga de distribuir el tráfico.

La imágen de la app tiene acoplada la parte de entrenamiento y de la API, el entrenamiento se hace antes de disponibilizar el server y crea los archivos necesarios para que luego la app pueda predecir y acceder a la BBDD.

## Cosas que habría hecho en un proyecto más grande.
- Utilizar otro soporte para la base de datos como PostgreSQL o MySQL.
- Generar más imágenes que dependan entre sí, separando `nginx`, `API`, Optimización/Entrenamiento. En vez de tener solo dos imágenes (`nginx` y el resto).
- Generar una nueva imágen para la Base de Datos para generar persistencia de datos, como está ahora la data se resetea al detener el contenedor.
- Autenticación y seguridad en general, en específico para la base de datos.
- Tests unitarios para las funciones/clases en `python`.
- Explorar más los datos para definir rangos aceptables. Ya que el usuario puede ingresar datos, no sería bueno si los datos ingresados están alterados, ya que solo se chequea que las columnas existan y no sean Nan. Podría incluirse otra tabla con stats de cada columna para ver si nuevos datos son anómalos.
- Utilizar los logs para un reentrenamiento rutinario en base al drift en los datos.