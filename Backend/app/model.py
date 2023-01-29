from pydantic import BaseModel
from typing import Optional

class Data(BaseModel):
    Contrast : float
    Brightness : float

class User(BaseModel):
    username : str
    password : str
    institute : str

class Login(BaseModel):
    username : str
    password : str

class Token(BaseModel):
    access_token : str
    token_type : str

class TokenData(BaseModel):
    username : Optional[str] = None