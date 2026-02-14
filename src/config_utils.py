import json
import os

CONFIG_FILE = "best_hyperparameters.json"

def save_params(model_name, params):
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    data[model_name] = params
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved {model_name} params to {CONFIG_FILE}")

def load_params(model_name):
    if not os.path.exists(CONFIG_FILE):
        return None
    with open(CONFIG_FILE, 'r') as f:
        data = json.load(f)
    return data.get(model_name)