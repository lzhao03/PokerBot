import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { brotliDecompressSync } from "node:zlib";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig, type Plugin } from "vite";

const policy = brotliDecompressSync(readFileSync(new URL("./models/pokerbot.policy.br", import.meta.url)));

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
  plugins: [svelte(), pokerPolicy()]
});
