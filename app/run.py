import os

import config
from flask_app.views import app

if __name__ == '__main__':
    app.run(debug=os.environ["DEBUG"])
