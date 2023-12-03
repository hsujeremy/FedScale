from fedscale.cloud.fllibs import *
import fedscale.cloud.config_parser as parser

logDir = None

class Clientfilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    """
    def __init__(self, client_id):
        self.client_id = client_id
    def filter(self, record):
        record.client = self.client_id
        return True

def init_logging(client_id=-1):
    global logDir

    logDir = os.path.join(parser.args.log_path, "logs", parser.args.job_name,
                          parser.args.time_stamp, 'executor')
    logFile = os.path.join(logDir, 'log')
    if not os.path.isdir(logDir):
        os.makedirs(logDir, exist_ok=True)

    handler = logging.StreamHandler()
    file_handler = logging.FileHandler(logFile, mode='a')
    handler.addFilter(Clientfilter(client_id))
    file_handler.addFilter(Clientfilter(client_id))
    logging.basicConfig(
        format='E%(client)s %(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='(%m-%d) %H:%M:%S',
        level=logging.INFO,
        handlers=[
            file_handler,
            handler
        ])


def initiate_client_setting(client_id=-1):
    init_logging(client_id=client_id)
