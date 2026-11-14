# syntax=docker/dockerfile:1
# CPU image: Python, pinned dependencies, and Tectonic with a pre-filled TeX
# cache, so tests, the pipeline and the PDF build all run inside it.
FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ARG TECTONIC_VERSION=0.16.9
ARG TECTONIC_SHA256_AARCH64=f9aa39017dbd51f111fdb93dda222178cbe51c8193508fc567b523cc74fff9c1
ARG TECTONIC_SHA256_X86_64=60b13a0826ae7ad9ce34b4a2df06bff2cfcfa6dda8a915477c0cbb84e1a4a902

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    HOME=/opt/home \
    XDG_CACHE_HOME=/opt/cache \
    MPLCONFIGDIR=/opt/cache/matplotlib \
    FEDUNLEARN_DEVICE=cpu

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    rm -rf /var/lib/apt/lists/*; \
    arch="$(uname -m)"; \
    case "$arch" in \
      aarch64) sum="$TECTONIC_SHA256_AARCH64" ;; \
      x86_64)  sum="$TECTONIC_SHA256_X86_64" ;; \
      *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    file="tectonic-${TECTONIC_VERSION}-${arch}-unknown-linux-musl.tar.gz"; \
    curl -fsSL -o "/tmp/$file" \
      "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${TECTONIC_VERSION}/$file"; \
    echo "$sum  /tmp/$file" | sha256sum -c -; \
    tar -xzf "/tmp/$file" -C /usr/local/bin tectonic; \
    rm "/tmp/$file"; \
    mkdir -p /opt/home /opt/cache

WORKDIR /app

COPY requirements.lock ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock

COPY run_pipeline.py ./
COPY fedunlearn ./fedunlearn
COPY report_template ./report_template
COPY scripts ./scripts
COPY tests ./tests

# Fetch every TeX package and font the report uses, then forbid network access
# for later builds. The container may run as any user, so caches are shared.
#
# Tectonic is a static musl binary and cannot use some container DNS resolvers
# (Colima's, for one). If the first attempt fails, resolve its two hosts with
# glibc for this step only and try again.
RUN set -eu; \
    if ! python scripts/warm_tex_cache.py; then \
      relay=relay.fullyjustified.net; \
      data="$(curl -fsSI "https://$relay/default_bundle_v33.tar.index.gz" \
              | tr -d '\r' | sed -n 's|^[Ll]ocation: https://\([^/]*\)/.*|\1|p')"; \
      for host in $relay $data; do \
        echo "$(getent ahostsv4 "$host" | awk '{print $1; exit}') $host" >> /etc/hosts; \
      done; \
      python scripts/warm_tex_cache.py; \
    fi; \
    chmod -R a+rwX /opt/home /opt/cache
ENV FEDUNLEARN_TEX_OFFLINE=1

CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
