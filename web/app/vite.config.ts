import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { brotliDecompressSync } from "node:zlib";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig, type Plugin } from "vite";

const repository = fileURLToPath(new URL("../..", import.meta.url));
const decoder = resolve(repository, "bazel-bin/web/policy_decoder.js");
const policy = brotliDecompressSync(readFileSync(resolve(repository, "models/compact_texture_16-16-64_small_betting_h36_current_100bb_550k_100m.policy.br")));
const neuralPolicy = readFileSync(resolve(repository, "models/deep_cfr_compact_texture_h36_history_small_100bb_i300.pnn"));

function pokerPolicy(): Plugin {
  let output = resolve("dist/pokerbot.policy");
  let neuralOutput = resolve("dist/deep-cfr.pnn");
  return {
    name: "poker-policy",
    configResolved(config) {
      output = resolve(config.root, config.build.outDir, "pokerbot.policy");
      neuralOutput = resolve(config.root, config.build.outDir, "deep-cfr.pnn");
    },
    configureServer(server) {
      server.middlewares.use("/pokerbot.policy", (_request, response) => {
        response.setHeader("Content-Type", "application/octet-stream");
        response.end(policy);
      });
      server.middlewares.use("/deep-cfr.pnn", (_request, response) => {
        response.setHeader("Content-Type", "application/octet-stream");
        response.end(neuralPolicy);
      });
    },
    writeBundle() {
      writeFileSync(output, policy);
      writeFileSync(neuralOutput, neuralPolicy);
    }
  };
}

export default defineConfig({
  plugins: [svelte(), pokerPolicy()],
  resolve: { alias: { "@poker/policy_decoder": decoder } },
  server: { fs: { allow: [repository] } }
});
