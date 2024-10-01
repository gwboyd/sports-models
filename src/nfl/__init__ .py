import os
from dotenv import load_dotenv

load_dotenv()

IS_LOCALHOST = os.getenv("LOCALHOST") == "True"
VERBOSE = IS_LOCALHOST