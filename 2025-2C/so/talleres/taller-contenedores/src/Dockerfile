FROM alpine AS external

WORKDIR /app

COPY ./externalServer .

RUN ["./externalServer"]

EXPOSE 8080

FROM alpine AS local

WORKDIR /app

COPY ./localServer .

RUN ["./localServer"]

EXPOSE 8000
