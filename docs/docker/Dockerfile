FROM ghcr.io/commaai/openpilot-base:latest as openpilot-docs-base

ENV PYTHONUNBUFFERED 1

ENV OPENPILOT_PATH /home/batman/openpilot/
ENV PYTHONPATH ${OPENPILOT_PATH}:${PYTHONPATH}
ENV POETRY_VIRUALENVS_CREATE false

RUN mkdir -p ${OPENPILOT_PATH}
WORKDIR ${OPENPILOT_PATH}

COPY SConstruct ${OPENPILOT_PATH}

COPY ./body ${OPENPILOT_PATH}/body
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
COPY ./system ${OPENPILOT_PATH}/system
COPY ./*.md ${OPENPILOT_PATH}/

RUN scons -j$(nproc)

RUN apt update && apt install doxygen -y
COPY ./docs ${OPENPILOT_PATH}/docs
RUN git init .
WORKDIR ${OPENPILOT_PATH}/docs
RUN make html

FROM nginx:1.21
COPY --from=0  /home/batman/openpilot/build/docs/html /usr/share/nginx/html
COPY ./docs/docker/nginx.conf /etc/nginx/conf.d/default.conf
