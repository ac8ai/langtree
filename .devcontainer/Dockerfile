ARG UBUNTU_VERSION="24.04"
ARG UV_VERSION="0.8.22"
ARG USER="user"
ARG USER_UID="1000"
ARG USER_GID=${USER_UID}
ARG PYTHON_VERSION='3.12'
ARG PROJECT_NAME='project'
ARG NODE_VERSION='vR0.40.3'
ARG NVM_VERSION='22.0.0'
ARG PROJDIR

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_image

FROM ubuntu:${UBUNTU_VERSION} AS build-env
COPY --from=uv_image /uv /uvx /bin/

ARG UBUNTU_VERSION
ARG UV_VERSION
ARG USER
ARG USER_UID
ARG USER_GID
ARG PROJECT_NAME
ARG NODE_VERSION
ARG NVM_VERSION

ENV UBUNTU_VERSION=${UBUNTU_VERSION}
ENV UV_VERSION=${UV_VERSION}

ARG PROJDIR="/workspace/${PROJECT_NAME}"
# Make non-interactive environment.
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# Don't enable bytecode compilation
ENV UV_COMPILE_BYTECODE=0
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

    ## Installing dependencies.
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        openssh-client gnupg2 gnupg-agent \
        wget \
        git \
        curl \
        # non-root user: \
        sudo \
    # non-root user: \
    && if ! getent group ${USER_GID}; then \
        groupadd --gid ${USER_GID} ${USER}; \
    fi \
    && if id -u ${USER_UID} >/dev/null 2>&1; then \
        existing_user=$(getent passwd ${USER_UID} | cut -d: -f1) \
        && usermod -l ${USER} -d /home/${USER} -m ${existing_user}; \
    else \
        useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USER}; \
    fi \
    && echo "${USER}:100000:65536" > /etc/subuid \
    && echo "${USER}:100000:65536" > /etc/subgid \
    && echo ${USER} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER} \
    ## preparing env for uv.
    && echo 'eval "$(uv generate-shell-completion bash)"' >> /home/${USER}/.bashrc \
    && echo 'eval "$(uvx --generate-shell-completion bash)"' >> /home/${USER}/.bashrc \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt \
    && apt-get clean

USER $USER
# ENV PATH="/app/.venv/bin:$PATH"
ENV USER=${USER}
ENV HOME=/home/${USER}

# Node 
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/${NODE_VERSION}/install.sh | bash \
    && export NVM_DIR="$HOME/.nvm" \
    && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" \
    && nvm install ${NVM_VERSION}
ENV NVM_DIR="/home/${USER}/.nvm"

# Claude Code
RUN curl -fsSL claude.ai/install.sh | bash
ENV PATH="${HOME}/.local/bin:${PROJDIR}/.venv/bin:${PATH}"

RUN mkdir -p ${HOME}/.local/share/containers /${HOME}/.config/containers

WORKDIR ${PROJDIR}

ENV DEVCONTAINER=true