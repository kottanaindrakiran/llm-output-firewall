import uvicorn
from fastapi import FastAPI
from openenv.core import OpenEnvApp

from main import app as fastapi_app

def main():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
