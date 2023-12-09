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
GL_ACK = 'ack'

# Client A sends GL_REQUEST_WEIGHTS to client B to get weights back
GL_REQUEST_WEIGHTS = "request_weights"
GL_REQUEST_NEIGHBORS = "request_neighbors"

# PLACEHOLD
DUMMY_RESPONSE = 'N'
GL_ACK_RESPONSE = 'ACK'


TENSORFLOW = 'tensorflow'
PYTORCH = 'pytorch'
