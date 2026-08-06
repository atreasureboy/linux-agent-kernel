# ── LAK (Linux Agent Kernel) Dockerfile ────────────────────────────
# Multi-stage build for minimal production image
#
# Build:
#   docker build -t lak:latest .
#
# Run:
#   docker run -p 9191:9191 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY lak:latest

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM rust:1.85-slim-bookworm AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/lak-core/Cargo.toml crates/lak-core/
COPY crates/lak-are/Cargo.toml crates/lak-are/
COPY crates/lak-tal/Cargo.toml crates/lak-tal/
COPY crates/lak-proto/Cargo.toml crates/lak-proto/
COPY crates/lak-proto/build.rs crates/lak-proto/
COPY crates/lak-proto/proto crates/lak-proto/proto
COPY crates/lak-services/Cargo.toml crates/lak-services/
COPY crates/lakd/Cargo.toml crates/lakd/

# Pre-build dependencies with empty src stubs (populates the dep cache)
RUN mkdir -p crates/lak-core/src crates/lak-are/src crates/lak-tal/src \
    crates/lak-proto/src crates/lak-services/src crates/lakd/src && \
    for d in lak-core lak-are lak-tal lak-proto lak-services; do \
        echo '' > crates/$d/src/lib.rs; \
    done && \
    echo 'fn main() {}' > crates/lakd/src/main.rs && \
    cargo build --release || true

# Copy actual source
COPY crates/ crates/

# Build release binary (touch stubs away so cargo recompiles them)
RUN touch crates/*/src/*.rs crates/lak-proto/src/lib.rs && \
    cargo build --release -p lakd

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        bash \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash lak

COPY --from=builder /build/target/release/lakd /usr/local/bin/lakd

USER lak
WORKDIR /home/lak

EXPOSE 9191

# TCP connect probe (no extra binaries needed in the slim image)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/9191' || exit 1

ENTRYPOINT ["lakd"]
