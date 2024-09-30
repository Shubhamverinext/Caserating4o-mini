from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from Main import flag_check
import pickle
import pandas as pd
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import yaml
from uvicorn import Config, Server
from pydantic import BaseModel

#logging.basicConfig(filename='auth_api.log', level=logging.INFO)
# Configure logging with a TimedRotatingFileHandler
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        TimedRotatingFileHandler(
            filename='app.log',
            when='W0',  # Rotate logs on a weekly basis, starting on Monday
            backupCount=1  # Retain one backup log file (the current week's log)
        )
    ],
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class InputData(BaseModel):
    msg:str

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# SECRET_KEY = "83daa0256a2289b0fb23693bf1f6034d44396675749244721a2b20e896e11662"
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Load the YAML file
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
SECRET_KEY = config['AuthConfig']['SECRET_KEY']
ALGORITHM = config['AuthConfig']['ALGORITHM']
ACCESS_TOKEN_EXPIRE_MINUTES = config['AuthConfig']['ACCESS_TOKEN_EXPIRE_MINUTES']

db = {
    "admin": {
        "username": "admin",
        "full_name": "admin",
        "email": "admin@gmail.com",
        "hashed_password": "$2b$12$MmntqWcewE7zMWgnHnWLQubk430mdycToxFSkGabA8Yu.GBryaBie",
        "disabled": False
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str or None = None


class User(BaseModel):
    username: str
    email: str or None = None
    full_name: str or None = None
    disabled: bool or None = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_data = db[username]
        return UserInDB(**user_data)


def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False

    return user


def create_access_token(data: dict, expires_delta: timedelta or None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credential_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                         detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credential_exception

        token_data = TokenData(username=username)
    except JWTError as error:
        logging.error('JWT Error occurred while decoding token: %s', error)
        raise credential_exception

    user = get_user(db, username=token_data.username)
    if user is None:
        logging.error('User not found for token: %s', token)
        raise credential_exception

    return user


async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")

    return current_user


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logging.warning("User login failed with username: %s", form_data.username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    logging.info("User login successful for username: %s", form_data.username)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires)
    logging.info("Access token generated for username: %s", form_data.username)
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@app.get("/users/me/items")
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return [{"item_id": 1, "owner": current_user}]

# Protected route that requires authentication
@app.post("/case-classifier")
async def case_classifier_endpoint(
    input_data: InputData,
    current_user: User = Depends(get_current_active_user),
):
    try:
        msg = input_data.msg
        logging.info(f"Received message: {msg}")
        result= flag_check(msg)
        logging.info(f"Processed result: {result}")
        # print(result)
        return result
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
   
    config = Config(app, host='0.0.0.0', port=8080, log_level='info')
    server = Server(config)
    server.run()


    

