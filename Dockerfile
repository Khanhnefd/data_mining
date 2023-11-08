FROM python:3.8-slim

# ARG MONGO_USER
# ARG MONGO_PASSWORD

# ENV MONGO_USER=${MONGO_USER}
# ENV MONGO_PASSWORD=${MONGO_PASSWORD}
ENV MONGO_USER=user
ENV MONGO_PASSWORD=pass_word
ENV WORKDIR=/rec_sys
ENV PYTHONPATH=${WORKDIR}

WORKDIR ${WORKDIR}

COPY ./app ./app
COPY ./data ./data
COPY ./model ./model
COPY ./recommendation ./recommendation
COPY requirements.txt requirements.txt

# SHELL ["/bin/bash", "-c"]

RUN pip install -r ./requirements.txt --no-cache-dir

CMD ["python", "app/main.py"]

# CMD [ "tail", "-f", "/dev/null" ]

EXPOSE 8001


#  docker build -f Dockerfile -t rec-sys:v1.0 --build-arg MONGO_USER= --build-arg MONGO_PASSWORD= .


# docker run --name khanh_rec -it -d -p 9001:8001 -e MONGO_USER= -e MONGO_PASSWORD= rec-sys:lastest
# docker run --name khanh_rec -it -d -p 9001:8001 --env-file .env rec-sys:latest
# docker build -f Dockerfile -t rec-sys:v1.0 .