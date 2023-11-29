# Define Basic Experiment Setup
from enum import Enum

SIMULATION_MODE = 'simulation'
DEPLOYMENT_MODE = 'deployment'

# Define Basic FL Events
UPDATE_MODEL = 'update_model'
MODEL_TEST = 'model_test'
SHUT_DOWN = 'terminate_executor'
START_ROUND = 'start_round'
CLIENT_CONNECT = 'client_connect'
CLIENT_TRAIN = 'client_train'
DUMMY_EVENT = 'dummy_event'
UPLOAD_MODEL = 'upload_model'

# Define Basic GL Events
GL_REQUEST_WEIGHTS = "request_weights"
GL_SEND_WEIGHTS = "send_weights"

# PLACEHOLD
DUMMY_RESPONSE = 'N'


TENSORFLOW = 'tensorflow'
PYTORCH = 'pytorch'
