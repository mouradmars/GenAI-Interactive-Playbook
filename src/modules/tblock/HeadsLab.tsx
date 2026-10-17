/** Multi-head attention explorer: head chambers, overlay, isolation and the ablation experiment. */
import { useMemo, useState } from 'react'
import { useColors } from '../../sim/appState'
import { runBlock, vnorm } from '../../engine/block'
import { Card, Tex } from '../../components/viz'
import { MatrixView, f3, useRoles } from './common'
import { cfgOf, useTb } from './store'

export function HeadsLab() {
  const c = useColors(), r = useRoles()
  const s = useTb((x) => x)
  const cfg = cfgOf(s)
  const trace = useMemo(() => runBlock(cfg, s.params, s.X, s.iv), [cfg.T, cfg.d, cfg.H, cfg.dff, cfg.act, cfg.arch, cfg.causal, cfg.eps, s.params, s.X, s.iv]) // eslint-disable-line react-hooks/exhaustive-deps
  const allOn = useMemo(() => runBlock(cfg, s.params, s.X, { ...s.iv, heads: s.iv.heads.map(() => true) }), [cfg.T, cfg.d, cfg.H, cfg.dff, cfg.act, cfg.arch, cfg.causal, cfg.eps, s.params, s.X, s.iv]) // eslint-disable-line react-hooks/exhaustive-deps
  const hc = [r.score, r.q, r.v, r.k, r.ffn, r.res, r.tok, r.ln]
  const [overlay, setOverlay] = useState<boolean[]>(() => s.iv.heads.map(() => true))
  const ov = overlay.length === s.H ? overlay : new Array(s.H).fill(true)
  // ablation: Y with each single head disabled, compared with all heads on
  const ablate = useMemo(() => Array.from({ length: s.H }, (_, h) => {
    const t = runBlock(cfg, s.params, s.X, { ...s.iv, heads: s.iv.heads.map((_, k) => k !== h) })
    return t.Y.map((y, i) => vnorm(y.map((v, k) => v - allOn.Y[i][k])))
  }), [allOn]) // eslint-disable-line react-hooks/exhaustive-deps
  const { T, d, H } = cfg, dh = d / H
  const isolate = (h: number) => s.setIv({ heads: s.iv.heads.map((_, k) => k === h) })
  return (
    <div className="grid gap-3">
      <Card title={`${H} head chambers · each with its own Q, K, V columns (dₖ = ${dh}) and its own attention pattern`}>
        <div className="flex flex-wrap gap-3">
          {trace.heads.map((hd, k) => (
            <div key={k} className={`rounded-lg border p-2 ${hd.enabled ? '' : 'opacity-50'}`} style={{ borderColor: hc[k] }}>
              <div className="mb-1 flex items-center justify-between gap-2 text-[12.5px]">
                <b style={{ color: hc[k] }}>head {k + 1}</b>
                <span className="flex items-center gap-1">
                  <label className="flex items-center gap-1 text-[11.5px]"><input type="checkbox" checked={s.iv.heads[k] !== false} onChange={() => s.toggleHead(k)} aria-label={`Enable head ${k + 1}`} />on</label>
                  <button type="button" className="btn !px-1.5 !py-0 text-[11px]" onClick={() => isolate(k)}>isolate</button>
                </span>
              </div>
              <MatrixView m={hd.A} mode="prob" maxAbs={1} rowLabels={s.tokens} colLabels={s.tokens.map((_, n) => `k${n + 1}`)} color={hc[k]} ariaLabel={`Head ${k + 1} weights`} testid={`head-A${k}`} cell={40} digits={2} onCell={(i, j) => s.select({ h: k, i, j })} />
              <div className="mt-1 font-mono text-[10.5px] text-faint">O_{k + 1}: [{T}, {dh}] · columns {k * dh + 1}–{(k + 1) * dh} of Concat</div>
            </div>
          ))}
        </div>
        <button type="button" className="btn mt-2" onClick={() => s.setIv({ heads: s.iv.heads.map(() => true) })}>enable all heads</button>
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title="overlay: attention from the selected query, per head">
          <div className="mb-1 flex flex-wrap gap-2 text-[12px]">{trace.heads.map((_, k) => <label key={k} className="flex items-center gap-1"><input type="checkbox" checked={ov[k]} onChange={() => setOverlay(ov.map((x, q) => (q === k ? !x : x)))} /><span style={{ color: hc[k] }}>head {k + 1}</span></label>)}</div>
          <svg viewBox={`0 0 ${T * 80} 130`} className="block w-full" role="img" aria-label="Overlaid attention">
            {trace.heads.map((hd, k) => ov[k] && hd.A[s.sel.i].map((w, j) => {
              if (w <= 0) return null
              const x1 = s.sel.i * 80 + 40, x2 = j * 80 + 40, off = (k - (H - 1) / 2) * 4
              return j === s.sel.i ? <circle key={`${k}.${j}`} cx={x1 + off} cy={80} r={3 + 10 * w} fill="none" stroke={hc[k]} strokeWidth={1 + 4 * w} opacity={0.4 + 0.6 * w} />
                : <path key={`${k}.${j}`} d={`M${x1 + off},90 Q${(x1 + x2) / 2},${90 - 20 - Math.abs(x2 - x1) * 0.35} ${x2 + off},90`} fill="none" stroke={hc[k]} strokeWidth={1 + 6 * w} opacity={0.35 + 0.65 * w} strokeLinecap="round" />
            }))}
            {s.tokens.map((t, i) => <g key={i}><rect x={i * 80 + 6} y={96} width={68} height={26} rx={6} fill={c.panel2} stroke={i === s.sel.i ? r.tok : c.line} /><text x={i * 80 + 40} y={113} textAnchor="middle" fontSize={11} fill={c.fg} fontFamily="var(--font-mono)">{t.slice(0, 8)}</text></g>)}
          </svg>
          <p className="text-[11.5px] text-muted">Query: “{s.tokens[s.sel.i]}” (click a cell in a head chamber to change it). Line width = attention weight.</p>
        </Card>
        <Card title="concatenation and output projection">
          <div className="overflow-x-auto scroll-thin text-[13px]"><Tex block tex={`\\mathrm{MultiHead} = \\underbrace{\\mathrm{Concat}(${trace.heads.map((_, k) => `O_{${k + 1}}`).join(',')})}_{[${T}\\times ${d}]}\\;\\underbrace{W_O}_{[${d}\\times ${d}]} + b_O \\in \\mathbb{R}^{${T}\\times ${d}}`} /></div>
          <div className="mt-1 flex h-6 overflow-hidden rounded">{trace.heads.map((hd, k) => <div key={k} className="flex flex-1 items-center justify-center font-mono text-[10.5px]" style={{ background: hd.enabled ? hc[k] : c.panel2, color: hd.enabled ? c.bg : c.faint }}>O{k + 1}{hd.enabled ? '' : ' = 0'}</div>)}</div>
          <p className="mt-2 text-[12px] text-muted">Different heads use different learned projections and can attend differently. Trained heads sometimes show recognisable patterns, but a head is not guaranteed to correspond to one linguistic function such as syntax or coreference, and these toy heads are untrained.</p>
        </Card>
      </div>
      <Card title="challenge: what happens if we disable one head?">
        <p className="mb-2 text-[12.5px] text-muted">For each head, the block is recomputed with only that head switched off (its O replaced by zeros), and the change in every output vector is measured: ‖Y_without − Y_all‖.</p>
        <div className="overflow-x-auto scroll-thin">
          <table className="font-mono text-[12px]" data-testid="ablation-table">
            <thead className="text-faint"><tr><th className="px-2 text-left">disabled</th>{s.tokens.map((t, i) => <th key={i} className="px-2 text-right">{t}</th>)}<th className="px-2 text-right">mean</th></tr></thead>
            <tbody>{ablate.map((row, h) => <tr key={h} className="border-t border-line"><td className="px-2" style={{ color: hc[h] }}>head {h + 1}</td>{row.map((v, i) => <td key={i} className="px-2 text-right">{f3(v)}</td>)}<td className="px-2 text-right">{f3(row.reduce((a, b) => a + b, 0) / row.length)}</td></tr>)}</tbody>
          </table>
        </div>
        <p className="mt-1 text-[11.5px] text-muted">A change of 0 would mean the output does not depend on that head at all. The first token in a causal block attends only to itself, so a head still changes it through its own value vector.</p>
      </Card>
    </div>
  )
}
