/**
 * dekho-local-inference — local model discovery for opencode.
 *
 * Registers one provider per running local inference server, so whatever
 * `llmctl start` launched is immediately pickable in /models with no config
 * editing. Several servers can run at once (a 30B on :8000 and an 8B on
 * :8001, say); a provider has exactly one baseURL, so each server becomes its
 * own provider. The lowest port keeps the stable id "dekho-local-inference".
 *
 * Sources, in order of authority:
 *   1. <endpoint>/models        the live server: what is reachable now.
 *   2. manifest.json            written by llmctl on every start/stop:
 *                               context window, generation ceiling, and
 *                               whether mlx-lm has a tool parser for a model
 *                               (resolved by mlx-lm itself, not guessed).
 *
 * The manifest matters because these are local repo ids that models.dev has
 * no entry for, so opencode would otherwise know neither the context window
 * (which drives compaction) nor whether a model can be driven agentically.
 *
 * Never throws and probes with a short timeout: a plugin that throws or hangs
 * blocks opencode from starting.
 */

import { readFile } from "node:fs/promises"
import { homedir } from "node:os"
import { join } from "node:path"

const PRIMARY_ID = "dekho-local-inference"
const DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
const MANIFEST =
  process.env.DEKHO_MANIFEST ||
  join(homedir(), ".cache", "dekho-local-inference", "manifest.json")
const PROBE_TIMEOUT_MS = 1500

async function readManifest() {
  try {
    return JSON.parse(await readFile(MANIFEST, "utf8"))
  } catch {
    return null
  }
}

async function probe(endpoint) {
  const ac = new AbortController()
  const timer = setTimeout(() => ac.abort(), PROBE_TIMEOUT_MS)
  try {
    const res = await fetch(`${endpoint}/models`, { signal: ac.signal })
    if (!res.ok) return null
    const body = await res.json()
    return (body?.data ?? []).map((m) => m.id).filter(Boolean)
  } catch {
    return null
  } finally {
    clearTimeout(timer)
  }
}

/** Normalise v1 (single endpoint) and v2 (servers array) manifests. */
function serversOf(manifest) {
  if (Array.isArray(manifest?.servers) && manifest.servers.length > 0) {
    return manifest.servers
  }
  if (manifest?.endpoint) {
    return [
      {
        provider_id: manifest.provider_id ?? PRIMARY_ID,
        provider_name: manifest.provider_name ?? "Dekho Local Inference (MLX)",
        endpoint: manifest.endpoint,
        model: manifest.preloaded,
        context: manifest.max_context_cap,
      },
    ]
  }
  return []
}

function buildModels({ ids, meta, liveIds, pinnedModel, serverContext }) {
  const models = {}
  for (const id of ids) {
    const m = meta[id] ?? {}
    const up = liveIds?.includes(id) ?? false
    const label = m.label ?? id.split("/").pop()
    // A server pins one model at boot; others load on demand via hot-swap,
    // which costs throughput — worth signalling in the picker.
    const flags = [
      up ? null : "offline",
      m.tool_call === false ? "no tools" : null,
      pinnedModel && id !== pinnedModel ? "swaps in" : null,
    ].filter(Boolean)
    // The server advertises one context cap; clamp it to the model's own
    // trained window so a smaller model is not over-filled.
    const native = m.context_native ?? m.context ?? 32768
    const context = Math.min(native, serverContext ?? native)
    models[id] = {
      name: flags.length ? `${label} [${flags.join(", ")}]` : label,
      tool_call: m.tool_call !== false,
      limit: { context, output: m.output ?? 32768 },
      ...(m.arch ? { family: m.arch } : {}),
    }
  }
  return models
}

export const DekhoLocalInference = async ({ client }) => {
  const log = async (level, message, extra) => {
    try {
      await client?.app?.log({
        body: { service: PRIMARY_ID, level, message, extra },
      })
    } catch {
      /* logging must never break startup */
    }
  }

  return {
    config: async (config) => {
      const manifest = await readManifest()
      const meta = manifest?.models ?? {}
      let servers = serversOf(manifest)

      // No manifest at all: still try the conventional endpoint, so the
      // plugin is useful before llmctl has ever run.
      if (servers.length === 0) {
        const ids = await probe(DEFAULT_ENDPOINT)
        if (!ids || ids.length === 0) {
          await log("info", "no local servers found; nothing registered", {
            manifest: MANIFEST,
          })
          return
        }
        servers = [
          {
            provider_id: PRIMARY_ID,
            provider_name: "Dekho Local Inference (MLX)",
            endpoint: DEFAULT_ENDPOINT,
          },
        ]
      }

      config.provider = config.provider ?? {}
      let registered = 0

      for (const srv of servers) {
        const liveIds = await probe(srv.endpoint)
        // Intersect, don't union. The manifest is the vetted list: it holds
        // only text models mlx-lm can serve, with a real context window and a
        // known tool-parser verdict. A server's own /v1/models lists whatever
        // it happens to hold — for a generative server that is TTS and
        // embedding models, which would otherwise be registered as chat models
        // with guessed metadata. Live ids confirm availability; they never add.
        const ids = new Set(Object.keys(meta))
        if (ids.size === 0) continue

        const providerId = srv.provider_id ?? PRIMARY_ID
        const existing = config.provider[providerId] ?? {}
        const models = buildModels({
          ids,
          meta,
          liveIds,
          pinnedModel: srv.model,
          serverContext: srv.context,
        })

        config.provider[providerId] = {
          ...existing,
          npm: existing.npm ?? "@ai-sdk/openai-compatible",
          name:
            existing.name ??
            srv.provider_name ??
            (srv.model
              ? `Dekho Local :${srv.port} (${srv.model.split("/").pop()})`
              : "Dekho Local Inference (MLX)"),
          options: {
            baseURL: srv.endpoint,
            apiKey: "not-needed", // mlx_lm.server has no auth
            ...(existing.options ?? {}),
          },
          // Discovered first, then anything hand-pinned in opencode.json, so
          // a manual entry always wins over discovery.
          models: { ...models, ...(existing.models ?? {}) },
        }
        registered++
      }

      await log("info", "registered local providers", {
        providers: registered,
        servers: servers.length,
        models: Object.keys(meta).length,
      })
    },
  }
}

export default DekhoLocalInference
