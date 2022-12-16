FROM python:3.8.11-buster

WORKDIR /app

ENV PORT 80

COPY ./requirements.txt /tmp/

RUN pip install --requirement /tmp/requirements.txt

COPY . .

CMD [ "morningstar.py" ]
ENTRYPOINT [ "python3" ]

# in command line cd to docker and run `docker build -t morningstar:1.0` to build a container

# to run the container use `docker run morningstar:1.0`

