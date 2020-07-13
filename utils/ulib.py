import logging 

class Logger:
    """Class to provide basic functions for producing log files.
    """
    __loggers = {}
    __path = 'log.txt'

    @staticmethod
    def set_log_file_path(path: str) -> None:
        Logger.__path = path

    @staticmethod
    def get_logger(name: str) -> logging:
        # return instance if already exist
        if Logger.__loggers.get(name):
            return Logger.__loggers.get(name)

        # create a new isstance
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # build handler
        f_handler = logging.FileHandler(Logger.__path)
        formatter = logging.Formatter("[%(asctime)s] %(message)s",
                                    "%Y-%m-%d %H:%M:%S")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

        # store logger to static arr
        Logger.__loggers[name] = logger

        return logger

class StdLogger:
    """Class to output stdouts to both console and log file.
    """
    
    def __init__(self) -> None:
        self.terminal_original = None
        self.terminal = None

    def open(self, log_file: str, flush_log_file: bool = False) -> None:
        self.terminal_original = sys.stdout
        if flush_log_file:
            self.terminal = open(log_file, "w")
        else:
            self.terminal = open(log_file, "a")
        sys.stdout = self.terminal

    def close(self) -> None:
        self.terminal.close()
        sys.stdout = self.terminal_original