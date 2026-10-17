/** Snapshot / data / report export. Works offline; falls back to copy-to-clipboard where downloads are blocked. */
import { X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useColors, type Colors } from '../sim/appState'

export interface ExportPayload { json: object; markdown: string; filenameBase: string }

const cssVarMap = (c: Colors): Record<string, string> => ({
  '--cyan': c.cyan, '--violet': c.violet, '--amber': c.amber, '--teal': c.teal, '--rose': c.rose, '--fg': c.fg, '--muted': c.muted,
  '--faint': c.faint, '--line': c.line, '--panel': c.panel, '--panel-2': c.panel2, '--bg': c.bg,
})

/** Render the largest SVG or canvas inside the stage to a PNG data URL. */
export async function snapshotStage(root: HTMLElement, c: Colors): Promise<string | null> {
  const candidates = [...root.querySelectorAll<SVGSVGElement | HTMLCanvasElement>('svg[role="img"]:not([data-snapshot="skip"]), canvas')]
  if (!candidates.length) return null
  const el = candidates.map((e) => ({ e, a: e.getBoundingClientRect().width * e.getBoundingClientRect().height })).sort((x, y) => y.a - x.a)[0].e
  if (el instanceof HTMLCanvasElement) {
    try { return el.toDataURL('image/png') } catch { return null }
  }
  const r = el.getBoundingClientRect()
  const clone = el.cloneNode(true) as SVGSVGElement
  clone.setAttribute('width', String(r.width))
  clone.setAttribute('height', String(r.height))
  clone.removeAttribute('style')
  // capture animated arcs in their finished state
  clone.querySelectorAll('path[pathLength]').forEach((p) => { p.removeAttribute('stroke-dasharray'); p.removeAttribute('stroke-dashoffset'); p.removeAttribute('pathLength') })
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  let src = new XMLSerializer().serializeToString(clone)
  for (const [k, v] of Object.entries(cssVarMap(c))) src = src.replaceAll(`var(${k})`, v)
  const img = new Image()
  const url = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(src)
  await new Promise<void>((res, rej) => { img.onload = () => res(); img.onerror = () => rej(new Error('svg render failed')); img.src = url })
  const scale = 2
  const cv = document.createElement('canvas')
  cv.width = r.width * scale; cv.height = r.height * scale
  const ctx = cv.getContext('2d')!
  ctx.fillStyle = c.bg
  ctx.fillRect(0, 0, cv.width, cv.height)
  ctx.scale(scale, scale)
  ctx.drawImage(img, 0, 0, r.width, r.height)
  return cv.toDataURL('image/png')
}

type Saver = { save: (r: { filename: string; data: Blob | string }) => Promise<unknown> }
type Host = { use?: (n: string) => Promise<Saver | null> }

/**
 * Save a file. Inside a hosted claude.ai page the viewer's frame blocks plain downloads,
 * so we ask the host's `downloads` capability; locally we fall back to an <a download>.
 */
async function download(name: string, data: Blob): Promise<string> {
  const host = (window as unknown as { claude?: Host }).claude
  if (host?.use) {
    const saver = await host.use('downloads').catch(() => null)
    if (saver) {
      try { await saver.save({ filename: name, data }); return 'Saved' }
      catch (e) { const code = (e as { code?: string }).code; return code === 'declined' ? 'Save cancelled' : 'Saving is unavailable here: use Copy instead' }
    }
  }
  const a = document.createElement('a')
  a.href = URL.createObjectURL(data); a.download = name
  document.body.appendChild(a); a.click(); a.remove()
  return 'Downloaded'
}
const dataUrlToBlob = (u: string) => {
  const [, b64] = u.split(',')
  const bin = atob(b64)
  return new Blob([Uint8Array.from(bin, (c) => c.charCodeAt(0))], { type: 'image/png' })
}

export function ExportDialog({ open, onClose, payload, stageEl }: { open: boolean; onClose: () => void; payload: () => ExportPayload; stageEl: HTMLElement | null }) {
  const c = useColors()
  const [tab, setTab] = useState<'image' | 'json' | 'report'>('image')
  const [png, setPng] = useState<string | null>(null)
  const [pngFailed, setPngFailed] = useState(false)
  const [data, setData] = useState<ExportPayload | null>(null)
  const [copied, setCopied] = useState('')
  useEffect(() => {
    if (!open) return
    setData(payload())
    setPng(null); setPngFailed(false)
    if (stageEl) snapshotStage(stageEl, c).then((u) => { setPng(u); setPngFailed(!u) }).catch(() => setPngFailed(true))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])
  if (!open || !data) return null
  const json = JSON.stringify(data.json, null, 2)
  const copy = async (text: string) => {
    try { await navigator.clipboard.writeText(text); setCopied('Copied to clipboard') } catch { setCopied('Copy blocked here: select the text and copy it manually') }
  }
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-bg/70 p-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="Export" onClick={onClose}>
      <div className="panel flex max-h-[88vh] w-[min(820px,100%)] flex-col rounded-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between border-b border-line px-4 py-3">
          <div className="flex gap-1">
            {(['image', 'json', 'report'] as const).map((t) => (
              <button key={t} type="button" className="btn" aria-pressed={tab === t} onClick={() => { setTab(t); setCopied('') }}>
                {t === 'image' ? 'Snapshot PNG' : t === 'json' ? 'Step data JSON' : 'Classroom report'}
              </button>
            ))}
          </div>
          <button type="button" className="btn icon-btn" aria-label="Close export" onClick={onClose}><X size={16} /></button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto p-4 scroll-thin">
          {tab === 'image' && (png ? <img src={png} alt="Snapshot of the current visualization" className="mx-auto max-h-[60vh] rounded-lg border border-line" /> : <p className="text-muted">{pngFailed ? 'This view has no image to capture. Switch to a step with a chart, or use the JSON and report tabs.' : 'Rendering the largest visualization on the stage…'}</p>)}
          {tab === 'json' && <pre className="whitespace-pre-wrap break-all font-mono text-[11.5px] text-muted">{json.slice(0, 60000)}</pre>}
          {tab === 'report' && <pre className="whitespace-pre-wrap font-mono text-[12.5px] leading-relaxed">{data.markdown}</pre>}
        </div>
        <div className="flex flex-wrap items-center gap-2 border-t border-line px-4 py-3">
          {tab === 'image' && png && <button type="button" className="btn btn-primary" onClick={async () => setCopied(await download(`${data.filenameBase}.png`, dataUrlToBlob(png)))}>Save PNG</button>}
          {tab === 'json' && <>
            <button type="button" className="btn btn-primary" onClick={async () => setCopied(await download(`${data.filenameBase}.json`, new Blob([json], { type: 'application/json' })))}>Save JSON</button>
            <button type="button" className="btn" onClick={() => copy(json)}>Copy</button></>}
          {tab === 'report' && <>
            <button type="button" className="btn btn-primary" onClick={async () => setCopied(await download(`${data.filenameBase}.md`, new Blob([data.markdown], { type: 'text/markdown' })))}>Save Markdown</button>
            <button type="button" className="btn" onClick={() => copy(data.markdown)}>Copy</button></>}
          <span className="text-[12px] text-muted" role="status">{copied || (tab === 'image' ? '' : '')}</span>
        </div>
      </div>
    </div>
  )
}
