FROM python:3.11-slim

# System deps. gosu (drop-priv entrypoint), git (agent autonomy), curl +
# ca-certificates. Plus the UNIFORM T4 privilege-contract leg (sudo +
# util-linux/nsenter + docker.io CLI): every template must satisfy the live
# T4 conformance gate (RFC internal#456 §11), even though google-adk defaults
# to tier 2 — the gate boots the image under tier-4 provisioner flags and
# asserts host-root reach. No node/npm (ADK is pure Python).
# docker-cli is explicit: docker.io 26.1.5+ split the `/usr/bin/docker` client
# into the `docker-cli` package (a Recommends), which --no-install-recommends
# drops — the T4 conformance probe needs the client for docker_socket_reachable.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gosu ca-certificates git sudo util-linux docker.io docker-cli \
    && rm -rf /var/lib/apt/lists/*

# Agent user — runs as uid-1000; /configs/.auth_token must stay agent-owned
# (Hermes list_peers-401 class). entrypoint.sh enforces ownership.
# /agent-home is the agent-writable home the T4 contract probes.
RUN useradd -u 1000 -m -s /bin/bash agent && \
    mkdir -p /agent-home && chown agent:agent /agent-home

# T4 escalation leg (RFC internal#456 §9/§11): wired path uid-1000 agent →
# host root inside the provisioner's --privileged --pid=host -v /:/host
# -v docker.sock container. NOPASSWD sudoers (visudo-validated) + docker group.
# Additive: does NOT change the agent uid or /configs token ownership.
RUN set -eux; \
    printf 'agent ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/agent-t4; \
    chmod 0440 /etc/sudoers.d/agent-t4; \
    visudo -cf /etc/sudoers.d/agent-t4; \
    groupadd -f docker; \
    usermod -aG docker agent; \
    id agent

WORKDIR /app

# RUNTIME_VERSION forwarded from the publish workflow busts the pip cache key
# on cascade builds. PIP_INDEX_URL = the Gitea PyPI registry (runtime wheel).
ARG RUNTIME_VERSION=
# Gitea registry is an EXTRA index (not the primary): it serves the private
# molecule-ai-workspace-runtime wheel; its transitive deps (pyyaml, starlette,
# python-multipart, …) and google-adk resolve from PyPI. Using it as
# --index-url alone fails (the registry is not a PyPI proxy).
ARG PIP_EXTRA_INDEX_URL=https://git.moleculesai.app/api/packages/molecule-ai/pypi/simple/

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url "${PIP_EXTRA_INDEX_URL}" -r requirements.txt && \
    if [ -n "${RUNTIME_VERSION}" ]; then \
      pip install --no-cache-dir --extra-index-url "${PIP_EXTRA_INDEX_URL}" --upgrade "molecule-ai-workspace-runtime==${RUNTIME_VERSION}"; \
    fi

# Google ADK from PyPI. [mcp] pulls the MCP client for McpToolset; we NEVER
# install the [a2a] extra (it pins a2a-sdk<0.4, incompatible with the
# platform's a2a-sdk>=1.0).
# FOLLOW-UP (RFC #730 Phase 0): switch to the molecule-ai/adk-python fork.
RUN pip install --no-cache-dir "google-adk[mcp]==2.1.0"

# Adapter code (top-level modules; /app is on sys.path, ADAPTER_MODULE=adapter)
COPY adapter.py google_adk_executor.py _routing.py __init__.py config.yaml ./
COPY internal/ ./internal/

# Generic GIT_ASKPASS helper (reused from the platform image contract).
COPY scripts/molecule-askpass /usr/local/bin/molecule-askpass
RUN chmod +x /usr/local/bin/molecule-askpass

ENV ADAPTER_MODULE=adapter

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
