import os, json
from typing import Any, Dict, Optional
from pymongo.collection import Collection
from .paths import ENV_FILE
from models.settings import default_env
from db.mongo import COL_ENV

def load_env_doc(db=None) -> Dict[str, Any]:
    if db is not None:
        doc = db[COL_ENV].find_one({"_id": "global"}) or {}
        return {**default_env(), **doc}
    if os.path.exists(ENV_FILE):
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
                return {**default_env(), **raw}
        except Exception:
            pass
    return default_env()

def save_env_doc(env_doc: Dict[str, Any], db=None):
    if db is not None:
        env_doc = {**default_env(), **(env_doc or {})}
        env_doc["_id"] = "global"
        db[COL_ENV].find_one_and_replace({"_id": "global"}, env_doc, upsert=True)
    else:
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            json.dump({**default_env(), **env_doc}, f, indent=2)
