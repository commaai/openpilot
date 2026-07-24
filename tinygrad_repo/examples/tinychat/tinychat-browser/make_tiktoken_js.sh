#!/usr/bin/env bash
cd "$(dirname "$0")"
npm init -y && \
npm install --save-dev webpack webpack-cli && \
npm install tiktoken && \
jq '.scripts.build = "webpack"' package.json > package.tmp.json && \
mv package.tmp.json package.json && \
npm run build && \
mv dist/*.wasm ./tiktoken_bg.wasm && \
mv dist/* ./ && \
rm -rf dist node_modules package-lock.json package.json