################################################################################
# build solvers
################################################################################
FROM debian:bookworm-slim as build_solvers

ARG CRYPTOMINISAT_VERSION=5.11.21
ARG APPROXMC_VERSION=4.1.24
ARG ARJUN_VERSION=2.5.4


RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    git \
    libboost-program-options-dev \
    libboost-serialization-dev \
    libgmp3-dev \
    libmpfr-dev \
    libntl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# espresso logic
RUN git clone https://github.com/classabbyamp/espresso-logic.git /tmp/espresso-logic \
    && cd /tmp/espresso-logic/espresso-src \
    && make CFLAGS=-O3 \
    && install -m 755 ../bin/espresso /usr/bin/espresso \
    && rm -rf /tmp/espresso-logic

# CryptoMiniSat, Arjun, ApproxMC
RUN mkdir /tmp/sat && cd /tmp/sat \
    && git clone https://github.com/msoos/cryptominisat \
    && cd cryptominisat \
    && git checkout tags/${CRYPTOMINISAT_VERSION} \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make \
    && make install \
    && ldconfig \
    && rm -rf /tmp/sat

# Install Arjun
RUN mkdir /tmp/sat && cd /tmp/sat \
    && git clone https://github.com/meelgroup/arjun \
    && cd arjun \
    && git checkout tags/${ARJUN_VERSION} \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make \
    && make install \
    && ldconfig \
    && rm -rf /tmp/sat

# Install ApproxMC
RUN mkdir /tmp/sat && cd /tmp/sat \
    && git clone https://github.com/meelgroup/approxmc \
    && cd approxmc \
    && git checkout tags/${APPROXMC_VERSION} \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make \
    && make install \
    && ldconfig \
    && rm -rf /tmp/sat

################################################################################
# build venv
################################################################################
FROM debian:bookworm-slim as build_venv

RUN apt-get update && apt-get install -y \
    python-is-python3 \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1

RUN useradd -m -u 1000 user
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# speed up subsequent builds of autodiver by installing the dependencies in a separate layer
RUN pip install click cython galois icecream ipython pyapproxmc pycryptosat!=5.11.23 sat_toolkit>=0.4.6 galois setuptools tqdm numpy scipy

RUN mkdir /home/user/autodiver
WORKDIR /home/user/autodiver

RUN mkdir tests/
COPY --chown=user tests/pyproject.toml tests/setup.py tests/
COPY --chown=user tests/src tests/src


RUN pip install ./tests \
    && rm -rf ~/.cache/pip

COPY --chown=user src/ .
COPY --chown=user pyproject.toml .
RUN pip install . \
    && rm -rf ~/.cache/pip


################################################################################
# main image
################################################################################
FROM debian:bookworm-slim

ENV TERM=xterm-256color
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y \
    libboost-program-options-dev \
    libboost-serialization-dev \
    libgmp3-dev \
    libmpfr-dev \
    libntl-dev \
    sudo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python-is-python3 \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --link --from=build_solvers /usr/bin/espresso /usr/bin/espresso
COPY --link --from=build_solvers /usr/local/bin/cryptominisat5 /usr/bin/cryptominisat5
COPY --link --from=build_solvers /usr/local/bin/arjun /usr/bin/arjun
COPY --link --from=build_solvers /usr/local/bin/approxmc /usr/bin/approxmc

COPY --link --from=build_solvers /usr/local/lib/libcryptominisat5.so* /usr/lib/
COPY --link --from=build_solvers /usr/local/lib/libarjun.so* /usr/lib/
COPY --link --from=build_solvers /usr/local/lib/libapproxmc.so* /usr/lib/

RUN useradd -m -s /usr/bin/zsh -G sudo -u 1000 user \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER user
WORKDIR /home/user

COPY --link --from=build_venv /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ENTRYPOINT ["autodiver"]
CMD ["--help"]
