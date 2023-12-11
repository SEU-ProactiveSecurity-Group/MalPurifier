from __future__ import print_function

import os
import sys
import time
import logging

if sys.version_info[0] < 3:
    import ConfigParser as configparser
else:
    import configparser

config = configparser.ConfigParser()

get = config.get
config_dir = os.path.dirname(__file__)


def parser_config():
    config_file = os.path.join(config_dir, "conf")

    if not os.path.exists(config_file):
        sys.stderr.write("Error: Unable to find the config file!\n")
        sys.exit(1)

    # parse the configuration
    global config
    config.read_file(open(config_file))

parser_config()

logging.basicConfig(level=logging.INFO, filename=os.path.join(config_dir, time.strftime("%Y%m%d-%H%M%S") + ".log"), filemode="w",
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
ErrorHandler = logging.StreamHandler()
ErrorHandler.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'))

