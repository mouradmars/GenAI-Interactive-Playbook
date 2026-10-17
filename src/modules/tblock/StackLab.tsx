/** Stacked blocks: per-block attention, hidden-state norms, feature values, a PCA trajectory view and block ablation. */
import { Microscope } from 'lucide-react'
import { useMemo } from 'react'
import { useColors } from '../../sim/appState'
import { runStack, toyParams, vdot, vnorm } from '../../engine/block'
import { pca } from '../../engine/linalg'
import { Card } from '../../components/viz'
import { MatrixView, f3, useRoles } from './common'
import { cfgOf, tbStore, useTb } from './store'

export const stackParams = (d: number, dff: number, seed: number, n: number) => Array.from({ length: n }, (_, k) => toyParams({ d, dff }, seed + 101 * k))

export function StackLab() {
  const c = useColors(), r = useRoles()
  const s = useTb((x) => x)
  const cfg = cfgOf(s)
  const params = useMemo(() => stackParams(s.d, s.dff, s.seed, s.stackN), [s.d, s.dff, s.seed, s.stackN])
  const on = s.stackOn.slice(0, s.stackN)
  const st = useMemo(() => runStack(cfg, params, s.X, on), [cfg.T, cfg.d, cfg.H, cfg.dff, cfg.act, cfg.arch, cfg.causal, cfg.eps, params, s.X, on.join()]) // eslint-disable-line react-hooks/exhaustive-deps
  const b = Math.min(s.stackBlock, s.stackN - 1), tb = st.traces[b]
  const h = Math.min(s.sel.h, s.H - 1), i = s.sel.i
  const states = [s.X, ...st.outputs] // hidden states after 0 … N blocks
  const tokCol = [r.tok, r.q, r.v, r.k, r.ffn, r.res, r.score, r.ln, c.rose, c.teal]
  const proj = useMemo(() => { const rows = states.flatMap((m) => m); const p = pca(rows, 2); return { coords: p.coords, explained: p.explained } }, [states]) // eslint-disable-line react-hooks/exhaustive-deps
  const T = s.T, N = s.stackN
  const at = (layer: number, tok: number) => proj.coords[layer * T + tok]
  const xs = proj.coords.map((q) => q[0]), ys = proj.coords.map((q) => q[1])
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  const W = 420, Hh = 260, sx = (v: number) => 20 + ((v - x0) / (x1 - x0 || 1)) * (W - 80), sy = (v: number) => Hh - 20 - ((v - y0) / (y1 - y0 || 1)) * (Hh - 40)
  const maxNorm = Math.max(...states.flatMap((m) => m.map(vnorm)))
  const open = () => {
    if (!tb) return
    tbStore.setState({ X: st.inputs[b].map((row) => row.slice()), params: params[b], xEdited: true, paramsEdited: true, tab: 'block' })
  }
  return (
    <div className="grid gap-3">
      <Card title="Block 1 → Block 2 → … → Block N · each block has its own (seeded toy) weights; the output of one is the input of the next">
        <div className="flex flex-wrap items-center gap-3 text-[12.5px] text-muted">
          <label className="flex items-center gap-2">blocks <span className="w-4 font-mono text-fg">{N}</span><input type="range" min={1} max={6} value={N} onChange={(e) => s.setStack({ stackN: +e.target.value })} aria-label="Number of blocks" /></label>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-1" data-testid="stack-strip">
          <span className="rounded border px-2 py-1 font-mono text-[12px]" style={{ borderColor: r.tok, color: r.tok }}>X</span>
          {Array.from({ length: N }, (_, k) => (
            <span key={k} className="flex items-center gap-1">
              <span className="text-faint">→</span>
              <button type="button" onClick={() => s.setStack({ stackBlock: k })} className={`rounded-lg border px-2 py-1 text-[12px] ${k === b ? 'border-cyan text-cyan shadow-[var(--glow)]' : 'border-line'} ${on[k] === false ? 'line-through opacity-50' : ''}`}>block {k + 1}</button>
              <input type="checkbox" checked={on[k] !== false} onChange={() => s.setStack({ stackOn: s.stackOn.map((x, q) => (q === k ? !x : x)) })} aria-label={`Enable block ${k + 1}`} title="disable (toy experiment)" />
            </span>
          ))}
          <span className="text-faint">→</span><span className="rounded border px-2 py-1 font-mono text-[12px]" style={{ borderColor: r.tok, color: r.tok }}>Y</span>
        </div>
        <p className="mt-1 text-[11.5px] text-faint">A disabled block passes its input through unchanged (identity), an experiment you can only do in a toy.</p>
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title={`block ${b + 1} · attention weights, head ${h + 1}`} right={tb && <button type="button" className="btn !py-0.5 text-[12px]" onClick={open} data-testid="open-block"><Microscope size={12} />open in the microscope</button>}>
          {tb ? <MatrixView m={tb.heads[h].A} mode="prob" maxAbs={1} rowLabels={s.tokens} colLabels={s.tokens.map((_, n) => `k${n + 1}`)} color={r.score} ariaLabel="Block attention" cell={44} onCell={(qi, kj) => s.select({ i: qi, j: kj })} />
            : <p className="text-[12.5px] text-muted">This block is disabled.</p>}
          <div className="mt-1 flex gap-1">{Array.from({ length: s.H }, (_, k) => <button key={k} type="button" className={`btn !py-0.5 text-[11.5px] ${k === h ? 'active' : ''}`} onClick={() => s.select({ h: k })}>head {k + 1}</button>)}</div>
        </Card>
        <Card title="hidden-state norm ‖h‖ of every token after each block">
          <svg viewBox={`0 0 ${W} 200`} className="block w-full" role="img" aria-label="Norms across blocks">
            {states.map((_, L) => <text key={L} x={30 + (L / Math.max(1, N)) * (W - 60)} y={196} textAnchor="middle" fontSize={10} fill={c.faint} fontFamily="var(--font-mono)">{L === 0 ? 'X' : `B${L}`}</text>)}
            {s.tokens.map((_, k) => <path key={k} d={states.map((m, L) => `${L ? 'L' : 'M'}${30 + (L / Math.max(1, N)) * (W - 60)},${180 - (vnorm(m[k]) / maxNorm) * 160}`).join('')} fill="none" stroke={tokCol[k % 10]} strokeWidth={k === i ? 3 : 1.5} opacity={k === i ? 1 : 0.6} />)}
          </svg>
          <div className="flex flex-wrap gap-2 text-[11px]">{s.tokens.map((t, k) => <button key={k} type="button" onClick={() => s.select({ i: k })} className="flex items-center gap-1" style={{ color: tokCol[k % 10] }}>● {t}</button>)}</div>
          <p className="mt-1 text-[11.5px] text-muted">In this pre-LN toy nothing renormalizes the stream between blocks, so norms can drift; real pre-LN models end with a final LayerNorm.</p>
        </Card>
      </div>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title={`“${s.tokens[i]}” feature by feature across the stack`}>
          <table className="font-mono text-[12px]" data-testid="stack-features">
            <thead className="text-faint"><tr><th className="px-2 text-left">after</th>{s.X[0].map((_, f) => <th key={f} className="px-2 text-right">x{f + 1}</th>)}<th className="px-2 text-right">‖·‖</th><th className="px-2 text-right">cos to prev.</th></tr></thead>
            <tbody>{states.map((m, L) => <tr key={L} className="border-t border-line"><td className="px-2">{L === 0 ? 'input' : `block ${L}`}</td>{m[i].map((v, f) => <td key={f} className="px-2 text-right">{f3(v)}</td>)}<td className="px-2 text-right">{f3(vnorm(m[i]))}</td><td className="px-2 text-right text-muted">{L ? f3(vdot(m[i], states[L - 1][i]) / (vnorm(m[i]) * vnorm(states[L - 1][i]) || 1)) : '—'}</td></tr>)}</tbody>
          </table>
        </Card>
        <Card title={`token trajectories through the stack · PCA to 2-D (${(proj.explained.reduce((a, q) => a + q, 0) * 100).toFixed(0)} % of variance)`}>
          <svg viewBox={`0 0 ${W} ${Hh}`} className="block w-full" role="img" aria-label="PCA trajectories" data-testid="stack-pca">
            {s.tokens.map((t, k) => (
              <g key={k}>
                <path d={states.map((_, L) => `${L ? 'L' : 'M'}${sx(at(L, k)[0])},${sy(at(L, k)[1])}`).join('')} fill="none" stroke={tokCol[k % 10]} strokeWidth={k === i ? 2.5 : 1.3} opacity={0.85} />
                {states.map((_, L) => <circle key={L} cx={sx(at(L, k)[0])} cy={sy(at(L, k)[1])} r={L === 0 ? 3 : L === N ? 5 : 3.5} fill={L === N ? tokCol[k % 10] : c.panel} stroke={tokCol[k % 10]} />)}
                <text x={sx(at(N, k)[0]) + 6} y={sy(at(N, k)[1]) - 6} fontSize={10.5} fill={tokCol[k % 10]} fontFamily="var(--font-mono)">{t}</text>
              </g>
            ))}
          </svg>
          <p className="text-[11.5px] text-muted">Hollow start = input, filled end = after the last block. A 2-D projection of {s.d}-dimensional vectors: distances and angles are only approximate, and the share above says how much of the variation the picture keeps.</p>
        </Card>
      </div>
      <p className="text-[12px] text-muted">Each block applies attention (mixing positions) and an MLP (per position) to the representations it receives, so deeper blocks repeatedly transform already contextualized vectors.</p>
    </div>
  )
}
