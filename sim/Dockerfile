# syntax=docker/dockerfile:1.7
# Builds the simulator on Alpine. The toolchain layer is cached, so only the compile steps rerun when a
# source changes.
#
#   native drivers:  docker build --network=host --target native --output type=local,dest=sim/bin sim
#   browser build:   docker build --network=host --target web --output type=local,dest=results/exports sim
#
# Behind a TLS-inspecting proxy add: --secret id=ca,src=<CA bundle> (mounted only while packages install).
ARG ALPINE=mirror.gcr.io/library/alpine:3.20

FROM ${ALPINE} AS toolchain
RUN --mount=type=cache,target=/var/cache/apk,sharing=locked \
    --mount=type=secret,id=ca,target=/etc/ssl/certs/ca-certificates.crt,required=false \
    apk add g++ clang lld wasi-libc wasi-libcxx wasi-compiler-rt
WORKDIR /src
COPY kernel.hpp .

FROM toolchain AS native-build
COPY data.hpp models.hpp threads.hpp assimilate.hpp nn.hpp nested_mc.cpp assimilate.cpp ./
RUN for b in nested_mc assimilate; do g++ -std=c++20 -O3 -march=x86-64-v2 -Wall -Wextra -Werror -pthread -static -o $b $b.cpp || exit 1; done

FROM scratch AS native
COPY --from=native-build /src/nested_mc /src/assimilate /

FROM toolchain AS web-build
COPY web.cpp .
RUN clang++ --target=wasm32-wasi --sysroot=/usr/share/wasi-sysroot -std=c++20 -O3 -Wall -Wextra -Werror \
      -fno-exceptions -fno-rtti -mexec-model=reactor -Wl,--strip-all -o lotto_sim.wasm web.cpp && ls -l lotto_sim.wasm

FROM scratch AS web
COPY --from=web-build /src/lotto_sim.wasm /lotto_sim.wasm
