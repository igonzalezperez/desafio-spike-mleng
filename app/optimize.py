from loguru import logger

import config
from database import database
from data_science import training

if __name__ == '__main__':
    logger.info('Create database')
    database.create_db(mode='replace')
    training.grid_search()
    training.train()
