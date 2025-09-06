FROM python:3.12.3-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# RUN apt-get update

# RUN apt-get install build-essential -y

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]