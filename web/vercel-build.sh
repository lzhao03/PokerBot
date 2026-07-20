#!/usr/bin/env bash
set -euo pipefail

npx --yes --package=@bazel/bazelisk \
  bazelisk build -c opt //web:policy_decoder_wasm

npm ci --prefix web/app
(cd web/app && ./node_modules/.bin/vite build)
