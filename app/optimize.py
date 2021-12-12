import os
from loguru import logger

import config
from config.logger_config import logger_config
from database import database
from data_science import training

logger_config(filepath=os.path.join('logs', 'train.log'),
              level=os.environ["LOG_LEVEL"])

if __name__ == '__main__':
    logger.info('Create database')
    database.create_db(mode='replace')
    database.db_data_span()
    training.grid_search()
    training.train()
