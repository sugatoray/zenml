FROM ubuntu:18.04

WORKDIR /zenml

# python
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_HOME=/root/.local

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update -y && \
  apt-get install --no-install-recommends -y -q software-properties-common && \
  # add-apt-repository ppa:deadsnakes/ppa && \
  add-apt-repository ppa:maarten-fonville/protobuf && \
  apt-get update -y && \
  apt-get install --no-install-recommends -y -q \
  build-essential \
  ca-certificates \
  libsnappy-dev \
  protobuf-compiler \
  libprotobuf-dev \
  python3.7 \
  python3.7-dev \
  python3-distutils \
  wget \
  unzip \
  git && \
  # add-apt-repository -r ppa:deadsnakes/ppa && \
  add-apt-repository -r ppa:maarten-fonville/protobuf && \
  # apt-get autoremove --purge python2.7-dev python2.7 libpython2.7 python2.7-minimal \
  # python3.5-dev python3.5 libpython3.5 python3.5-minimal -y && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
  update-alternatives --install /usr/bin/python-config python-config /usr/bin/python3.7-config 1 && \
  apt-get autoclean && \
  apt-get autoremove --purge && \
  wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
  pip install --no-cache-dir --upgrade --pre pip  && \
  wget https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py && python install-poetry.py

# copy project requirement files here to ensure they will be cached.
COPY pyproject.toml /zenml

# install runtime deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
RUN poetry config virtualenvs.create false && poetry install

# create an alias for zenml
RUN echo 'alias zenml="poetry run zenml"' >> ~/.bashrc

ADD . /zenml