FROM ghcr.io/commaai/openpilot-base:latest

ENV PYTHONUNBUFFERED 1

ENV OPENPILOT_PATH /home/batman/openpilot/
ENV PYTHONPATH ${OPENPILOT_PATH}:${PYTHONPATH}

RUN mkdir -p ${OPENPILOT_PATH}
WORKDIR ${OPENPILOT_PATH}

COPY Pipfile Pipfile.lock $OPENPILOT_PATH
RUN pip install --no-cache-dir pipenv==2020.8.13 && \
    pipenv install --system --deploy --dev --clear && \
    pip uninstall -y pipenv


COPY SConstruct ${OPENPILOT_PATH}

COPY ./pyextra ${OPENPILOT_PATH}/pyextra
COPY ./third_party ${OPENPILOT_PATH}/third_party
COPY ./site_scons ${OPENPILOT_PATH}/site_scons
COPY ./laika ${OPENPILOT_PATH}/laika
COPY ./laika_repo ${OPENPILOT_PATH}/laika_repo
COPY ./rednose ${OPENPILOT_PATH}/rednose
COPY ./rednose_repo ${OPENPILOT_PATH}/rednose_repo
COPY ./tools ${OPENPILOT_PATH}/tools
COPY ./release ${OPENPILOT_PATH}/release
COPY ./common ${OPENPILOT_PATH}/common
COPY ./opendbc ${OPENPILOT_PATH}/opendbc
COPY ./cereal ${OPENPILOT_PATH}/cereal
COPY ./panda ${OPENPILOT_PATH}/panda
COPY ./selfdrive ${OPENPILOT_PATH}/selfdrive
COPY ./*.md ${OPENPILOT_PATH}/

RUN scons -j$(nproc)

RUN apt-get update && apt-get install -y --no-install-recommends \
    doxygen \
    graphviz

COPY ./docs ${OPENPILOT_PATH}/docs
RUN mkdir build && cd build && mkdir docs/
RUN git init .
WORKDIR ${OPENPILOT_PATH}/docs
RUN mkdir build && cd build && cmake ../
RUN cd build && cmake --build . --target doxy_build -j$(nproc)
RUN cd build && cmake --build . --target sphinx_build -j$(nproc)
RUN cd build && cmake --install . -j$(nproc)

FROM nginx:1.21
COPY --from=0  /home/batman/openpilot/docs/build/sphinx/html /usr/share/nginx/html
COPY ./docs/docker/nginx.conf /etc/nginx/conf.d/default.conf
