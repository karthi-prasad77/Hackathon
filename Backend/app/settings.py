from pydantic import BaseSettings

class photo(BaseSettings):
    mongo_url : str

    class Config:
        env_file = '/.env'