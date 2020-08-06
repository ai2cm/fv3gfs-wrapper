import os
import json

DIR = os.path.dirname(os.path.abspath(__file__))
MM_PER_M = 1000

with open(os.path.join(DIR, "dynamics_properties.json"), "r") as f:
    DYNAMICS_PROPERTIES = json.load(f)

with open(os.path.join(DIR, "physics_properties.json"), "r") as f:
    PHYSICS_PROPERTIES = json.load(f)
