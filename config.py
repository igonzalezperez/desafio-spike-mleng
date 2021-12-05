DATA_SOURCES = ['rain', 'central_bank', 'milk_price']
COLS = {'rain':
        {'name': 'all',
         'subset': None
         },
        'central_bank':
        {'name': ['Period', 'Indice_de_ventas_comercio_real_no_durables_IVCM'],
         'subset': ['PIB', 'IMACEC']
         },
        'milk_price': 'all'
        }
