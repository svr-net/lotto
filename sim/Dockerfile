# syntax=docker/dockerfile:1.7
# Builds sim/nested_mc.cpp. The toolchain layer is cached, so only the compile reruns when the source changes:
#
#   docker build --network=host --output type=local,dest=sim/bin sim
#
# Behind a TLS-inspecting proxy add: --secret id=ca,src=<CA bundle> (mounted only while packages install).
ARG BASE=mirror.gcr.io/library/alpine:3.20

FROM ${BASE} AS toolchain
RUN --mount=type=cache,target=/var/cache/apk,sharing=locked \
    --mount=type=secret,id=ca,target=/etc/ssl/certs/ca-certificates.crt,required=false \
    apk add g++

FROM toolchain AS build
WORKDIR /src
COPY nested_mc.cpp .
RUN g++ -std=c++20 -O3 -march=x86-64-v2 -Wall -Wextra -Werror -pthread -static -o nested_mc nested_mc.cpp

FROM scratch
COPY --from=build /src/nested_mc /nested_mc
