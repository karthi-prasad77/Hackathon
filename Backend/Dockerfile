FROM python

WORKDIR /Backend

RUN apt-get update

RUN apt install -y libgl1-mesa-glx

COPY ./requirements.txt /Backend/requirements.txt

RUN pip install -r /Backend/requirements.txt

COPY /app /code/app

EXPOSE 8000


