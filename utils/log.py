import logging


class Logger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.DEBUG = logging.DEBUG
        self.INFO = logging.INFO
        self.WARNING = logging.WARNING
        self.ERROR = logging.ERROR
        self.level = dict(DEBUG=10, INFO=20, WARNING=30, ERROR=40)
        self.formatter = logging.Formatter('%(name)s (%(levelname)s) [%(asctime)s]: %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S')
        ch = logging.StreamHandler()
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

    def set_level(self, lv):
        self.logger.setLevel(lv)

    def write(self, filename, mode='w'):
        fh = logging.FileHandler(filename, mode=mode, encoding='utf-8')
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)
