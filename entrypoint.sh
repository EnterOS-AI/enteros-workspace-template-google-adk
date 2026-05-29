#!/bin/sh
# Drop privileges to the agent user (uid 1000) before exec'ing molecule-runtime,
# fixing /configs ownership first so /configs/.auth_token is agent-owned (the
# Hermes list_peers-401 contract, RFC internal#456). Pattern matches the proven
# template-claude-code / template-langgraph entrypoints.

# Source persistent workspace secrets BEFORE anything that needs them.
if [ -f /configs/secrets.d/load.sh ]; then
  . /configs/secrets.d/load.sh
fi

# Boot-context snapshot on every start — NAMES of auth env vars only, never values.
log_boot_context() {
    echo "----- entrypoint boot $(date -u +%Y-%m-%dT%H:%M:%SZ) -----"
    echo "uid=$(id -u) gid=$(id -g) user=$(id -un 2>/dev/null || echo unknown)"
    echo "hostname=$(hostname) workspace_id=${WORKSPACE_ID:-<unset>} platform_url=${PLATFORM_URL:-<unset>}"
    echo "configs: $(ls -ld /configs 2>/dev/null || echo MISSING)"
    for var in GOOGLE_API_KEY GOOGLE_GENAI_USE_VERTEXAI GOOGLE_CLOUD_PROJECT GOOGLE_CLOUD_LOCATION; do
        eval "val=\$$var"
        [ -n "$val" ] && echo "env $var=set" || echo "env $var=unset"
    done
    echo "------------------------------------------------"
}
log_boot_context

if [ "$(id -u)" = "0" ]; then
    # /configs is created root by Docker; the uid-1000 agent needs to read +
    # rewrite /configs/.auth_token. chown BEFORE the gosu re-exec so the
    # runtime's first token write is agent-owned.
    chown -R agent:agent /configs 2>/dev/null
    chown agent:agent /workspace 2>/dev/null || true
    if [ -d /workspace ]; then
        first=$(find /workspace -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)
        if [ -n "$first" ] && [ "$(stat -c '%u' "$first" 2>/dev/null)" = "0" ]; then
            chown -R agent:agent /workspace 2>/dev/null
        fi
    fi
    exec gosu agent "$0" "$@"
fi

# Now running as agent (uid 1000).
exec molecule-runtime "$@"
