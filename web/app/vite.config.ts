import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { brotliDecompressSync } from "node:zlib";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig, type Plugin } from "vite";

const repository = fileURLToPath(new URL("../..", import.meta.url));
const decoder = resolve(repository, "bazel-bin/web/policy_decoder.js");
const policy = brotliDecompressSync(readFileSync(resolve(repository, "models/compact_texture_16-16-64_small_betting_h36_current_100bb_550k_100m.policy.br")));

function pokerPolicy(): Plugin {
  let output = resolve("dist/pokerbot.policy");
  return {
    name: "poker-policy",
    configResolved(config) {
      output = resolve(config.root, config.build.outDir, "pokerbot.policy");
    },
    configureServer(server) {
      server.middlewares.use("/pokerbot.policy", (_request, response) => {
        response.setHeader("Content-Type", "application/octet-stream");
        response.end(policy);
      });
    },
    writeBundle: () => writeFileSync(output, policy)
  };
}

export default defineConfig({
  plugins: [svelte(), pokerPolicy()],
  resolve: { alias: { "@poker/policy_decoder": decoder } },
  server: { fs: { allow: [repository] } }
});
