// Agency policy bridge for DeepSeek Harness (dsh), loaded as a Cordis plugin.
//
// dsh ships no command-hook bridge in its standalone runtime, so this plugin
// does what `_harness_permission_hook.py` does for the other CLIs, from inside
// the harness: before every tool call it asks Agency's `/agpolicy/check_tool`
// (allow or deny, fail closed), and after the call it reports the result to
// `/agpolicy/complete_tool` (best effort). Dependency-free on purpose: it is
// imported from disk by the bundled runtime, which cannot resolve npm packages
// for it. `AGPOLICY_BASE_URL` and `AGPOLICY_TOKEN` come from the harness env.
//
// dsh runs with its own approvals off, so this plugin is the only tool
// admission there is. When dsh applies it, it creates `AGPOLICY_LOADED_FILE`;
// the adapter fails an attempt that ends without that file.

import fs from 'node:fs'
import http from 'node:http'
import https from 'node:https'

export const name = 'agency-agpolicy'

const POLICY_TIMEOUT_MS = 5000
const COMPLETION_TIMEOUT_MS = 2000
const MAX_RESULT_CHARS = 64 * 1024

// node:http with no connection pool: a pooled keep-alive socket (as global
// fetch keeps) would hold the headless runtime open after its final answer.
function post(path, body, timeoutMs) {
  const base = (process.env.AGPOLICY_BASE_URL || '').replace(/\/+$/, '')
  const token = process.env.AGPOLICY_TOKEN || ''
  if (!base || !token) return Promise.reject(new Error('agpolicy gateway not configured'))
  const url = new URL(base + path)
  const data = Buffer.from(JSON.stringify(body))
  const transport = url.protocol === 'https:' ? https : http
  return new Promise((resolve, reject) => {
    const request = transport.request(
      url,
      {
        method: 'POST',
        agent: false,
        timeout: timeoutMs,
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': data.length,
          Authorization: `Bearer ${token}`,
          Connection: 'close',
        },
      },
      (response) => {
        const chunks = []
        response.on('data', (chunk) => chunks.push(chunk))
        response.on('end', () => {
          if (response.statusCode < 200 || response.statusCode >= 300) {
            reject(new Error(`agpolicy ${path} returned HTTP ${response.statusCode}`))
            return
          }
          try {
            resolve(JSON.parse(Buffer.concat(chunks).toString('utf8') || 'null'))
          } catch (error) {
            reject(error)
          }
        })
        response.on('error', reject)
      },
    )
    request.on('timeout', () => request.destroy(new Error(`agpolicy ${path} timed out`)))
    request.on('error', reject)
    request.end(data)
  })
}

function resultPayload(result) {
  if (!result) return { result: null, error: 'Tool returned no result' }
  const content = Array.isArray(result.content) ? result.content : []
  let text = content.map((block) => (block && block.type === 'text' ? block.text : '')).join('')
  // A result with no text (an image, structured content) is still a result.
  if (!text && content.length) text = JSON.stringify(content)
  let error = null
  if (result.isError) {
    error = (result.error && (result.error.message || result.error.code)) || text || 'Tool reported failure'
  }
  const value = text.length > MAX_RESULT_CHARS ? text.slice(0, MAX_RESULT_CHARS) + '…[truncated]' : text
  return { result: value, error: typeof error === 'string' ? error.slice(0, MAX_RESULT_CHARS) : error }
}

export function apply(ctx) {
  if (process.env.AGPOLICY_LOADED_FILE) fs.writeFileSync(process.env.AGPOLICY_LOADED_FILE, '')

  // Pre-call admission hands its span id to the post-call report. Calls that
  // come without an id are keyed by their own object, so they never share one.
  const callIds = new Map()
  const keyOf = (exec) => exec.callId ?? exec

  async function complete(exec, payload) {
    const callId = callIds.get(keyOf(exec))
    if (!callId) return
    callIds.delete(keyOf(exec))
    try {
      await post('/agpolicy/complete_tool', { call_id: callId, ...payload }, COMPLETION_TIMEOUT_MS)
    } catch {
      // Completion telemetry never revokes an admitted call.
    }
  }

  ctx.on('tools/pre-execute', async (exec, next) => {
    let decision
    try {
      decision = await post(
        '/agpolicy/check_tool',
        {
          tool_name: exec.name,
          tool_input: exec.arguments && typeof exec.arguments === 'object' ? exec.arguments : {},
          model_tool_call_id: exec.callId,
        },
        POLICY_TIMEOUT_MS,
      )
    } catch (error) {
      return { kind: 'deny', reason: `Cannot check invocation admission: ${error.message}; return to the model` }
    }
    if (!decision || decision.decision !== 'allow') {
      const reason = decision && typeof decision.reason === 'string' ? decision.reason : 'denied by agpolicy'
      return { kind: 'deny', reason }
    }
    if (decision.call_id) callIds.set(keyOf(exec), decision.call_id)
    try {
      return await next()
    } catch (error) {
      // A tool that throws never reaches post-execute; close its span here.
      await complete(exec, { result: null, error: String((error && error.message) || error) })
      throw error
    }
  })

  ctx.on('tools/post-execute', async (exec, result, next) => {
    await complete(exec, resultPayload(result))
    return next()
  })
}
