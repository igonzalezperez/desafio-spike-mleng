import os

from dotenv import load_dotenv

from data_science.utils.logger_config import logger_config

load_dotenv()
logger_config(level=os.environ["LOG_LEVEL"])

MODEL_DIR = os.environ["MODEL_DIR"]
PIPELINE_DIR = os.environ["PIPELINE_DIR"]
PARAM_DIR = os.environ["PARAM_DIR"]

BEST_PARAMS_FILE = os.environ["BEST_PARAMS_FILE"]
FEATURE_PIPELINE_FILE = os.environ["FEATURE_PIPELINE_FILE"]
TARGET_PIPELINE_FILE = os.environ["TARGET_PIPELINE_FILE"]
MODEL_FILE = os.environ["MODEL_FILE"]

BEST_PARAMS_PATH = os.path.join(MODEL_DIR, PARAM_DIR, BEST_PARAMS_FILE)
FEATURE_PIPELINE_PATH = os.path.join(
    MODEL_DIR, PIPELINE_DIR, FEATURE_PIPELINE_FILE)
TARGET_PIPELINE_PATH = os.path.join(
    MODEL_DIR, PIPELINE_DIR, TARGET_PIPELINE_FILE)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

REQUIRED_COLUMNS = {'rain': ['date', 'Coquimbo', 'Valparaiso', 'Metropolitana_de_Santiago',
                             'Libertador_Gral__Bernardo_O_Higgins', 'Maule', 'Biobio', 'La_Araucania',	'Los_Rios',
                             ],
                    'central_bank': ['Imacec_empalmado', 'Imacec_produccion_de_bienes', 'Imacec_minero',
                                     'Imacec_industria', 'Imacec_resto_de_bienes', 'Imacec_comercio',	'Imacec_servicios',	'Imacec_a_costo_de_factores', 'Imacec_no_minero',	'PIB_Agropecuario_silvicola', 'PIB_Pesca', 'PIB_Mineria',	'PIB_Mineria_del_cobre', 'PIB_Otras_actividades_mineras', 'PIB_Industria_Manufacturera', 'PIB_Alimentos', 'PIB_Bebidas_y_tabaco', 'PIB_Textil', 'PIB_Maderas_y_muebles', 'PIB_Celulosa',	'PIB_Refinacion_de_petroleo', 'PIB_Quimica', 'PIB_Minerales_no_metalicos_y_metalica_basica', 'PIB_Productos_metalicos',	'PIB_Electricidad',	'PIB_Construccion',	'PIB_Comercio',	'PIB_Restaurantes_y_hoteles', 'PIB_Transporte', 'PIB_Comunicaciones',	'PIB_Servicios_financieros', 'PIB_Servicios_empresariales',	'PIB_Servicios_de_vivienda', 'PIB_Servicios_personales',	'PIB_Administracion_publica', 'PIB_a_costo_de_factores', 'PIB',	'Indice_de_ventas_comercio_real_no_durables_IVCM',
                                     ],
                    'milk_price': ['Anio', 'Mes',	'Precio_leche',
                                   ]
                    }
