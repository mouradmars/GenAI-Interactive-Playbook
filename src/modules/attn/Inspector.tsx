/**
 * Right panel · the mathematical microscope: the active equation, its numerical substitution for the current
 * selection, the selected pair's intermediate values, tensor shapes, an explanation at the chosen depth and
 * common misconceptions. All numbers are read from the engine trace.
 */
import { AlertTriangle } from 'lucide-react'
import { Tex } from '../../components/viz'
import { useApp } from '../../sim/appState'
import { aggregationTerms, matmulCell, shapeList, type AttnTrace } from '../../engine/attention'
import { GLYPH, fx, tx, useRoles, useTokens } from './common'
import type { Stage } from './stages'
import { useAttn } from './store'

const SHAPES_FOR: Record<Stage['id'], string[]> = {
  tokens: ['X'], embed: ['X'], qkv: ['X', 'W_Q^(h)', 'Q_h', 'K_h', 'V_h'], dot: ['Q_h', 'K_h', 'S_h'], scale: ['S_h'], mask: ['S_h', 'M'], softmax: ['M', 'A_h'], heatmap: ['A_h'],
  aggregate: ['A_h', 'V_h', 'head_h'], heads: ['Q_h', 'K_h', 'V_h', 'A_h', 'head_h'], concat: ['head_h', 'H'], proj: ['H', 'W_O', 'O'], output: ['X', 'O'],
}

export function substitution(stage: Stage['id'], t: AttnTrace, sel: { i: number; j: number; h: number; cell: { m: string; r: number; c: number } | null }, Wo: number[][]): string[] {
  const h = Math.min(sel.h, t.cfg.H - 1), hd = t.heads[h], i = Math.min(sel.i, t.Tq - 1), j = Math.min(sel.j, t.Tk - 1), dk = t.dk
  const vec = (v: number[], d = 2) => `[${v.map((x) => tx(x, d)).join(',\\,')}]`
  const sum = (a: number[], b: number[]) => a.map((x, k) => `${tx(x, 2)}\\cdot${tx(b[k], 2)}`).join(' + ')
  const row = hd.rows[i]
  switch (stage) {
    case 'tokens': case 'embed': return [`x_{${i + 1}} = ${vec(t.X[i])}`]
    case 'qkv': {
      const m = sel.cell && 'QKV'.includes(sel.cell.m) && sel.cell.m.length === 1 ? sel.cell : { m: 'Q', r: i, c: 0 }
      const W = m.m === 'Q' ? hd.Wq : m.m === 'K' ? hd.Wk : hd.Wv
      const cc = matmulCell(t.X, W, m.r, m.c)
      const lc = m.m.toLowerCase()
      return [`${lc}_{${m.r + 1},${m.c + 1}} = x_{${m.r + 1}} \\cdot W_${m.m}^{(${h + 1})}[:,${m.c + 1}]`, `= ${sum(cc.a, cc.b)}`, `= ${tx(cc.sum, 4)}`, `q_{${i + 1}} = ${vec(hd.Q[i], 3)},\\; k_{${i + 1}} = ${vec(hd.K[i], 3)},\\; v_{${i + 1}} = ${vec(hd.V[i], 3)}`]
    }
    case 'dot': return [`s_{${i + 1}${j + 1}} = q_{${i + 1}} \\cdot k_{${j + 1}} = ${sum(hd.Q[i], hd.K[j])}`, `= ${tx(hd.S[i][j], 4)}`]
    case 'scale': return t.cfg.scale ? [`\\tilde s_{${i + 1}${j + 1}} = \\frac{${tx(hd.S[i][j], 4)}}{\\sqrt{${dk}}} = \\frac{${tx(hd.S[i][j], 4)}}{${tx(Math.sqrt(dk), 4)}} = ${tx(hd.scaled[i][j], 4)}`] : [`\\text{scaling off: } \\tilde s_{${i + 1}${j + 1}} = s_{${i + 1}${j + 1}} = ${tx(hd.S[i][j], 4)}`]
    case 'mask': return [`M_{${i + 1}${j + 1}} = ${t.M[i][j] === 0 ? '0' : '-\\infty'}${t.reasons[i][j].length ? `\\;(\\text{${t.reasons[i][j].join(' + ')}})` : ''}`, `z_{${i + 1}${j + 1}} = ${tx(hd.scaled[i][j], 4)} + ${t.M[i][j] === 0 ? '0' : '(-\\infty)'} = ${tx(hd.logits[i][j], 4)}`]
    case 'softmax': case 'heatmap': {
      if (row.fullyMasked) return [`\\text{row ${i + 1} is fully masked: no permitted key}`, `a_{${i + 1}j} = 0 \\text{ for all } j \\text{ (flagged)}`]
      const ex = row.allowed[j] ? `e^{${tx(row.shifted[j], 3)}}` : '0'
      return [`m_{${i + 1}} = ${tx(row.max, 4)}`, `a_{${i + 1}${j + 1}} = \\frac{${ex}}{\\sum_k e^{z_{${i + 1}k} - m}} = \\frac{${tx(row.exps[j], 4)}}{${tx(row.sum, 4)}} = ${tx(hd.A[i][j], 4)}`, `\\textstyle\\sum_j a_{${i + 1}j} = ${tx(hd.A[i].reduce((a, b) => a + b, 0), 6)}`]
    }
    case 'aggregate': {
      const g = aggregationTerms(hd, i)
      return [`o_{${i + 1}} = ${g.a.map((a, k) => `${tx(a, 3)}\\,v_{${k + 1}}`).join(' + ')}`, `= ${vec(g.o, 4)}`]
    }
    case 'heads': return t.heads.map((x) => `\\text{head ${x.h + 1}: } a_{${i + 1}${j + 1}} = ${tx(x.A[i][j], 4)},\\; H(a_{${i + 1}}) = ${tx(x.entropy[i], 3)}\\text{ nats}${x.enabled ? '' : '\\;(\\text{disabled})'}`)
    case 'concat': return [`H_{${i + 1}} = [${t.heads.map((x) => (x.enabled ? x.O[i] : x.O[i].map(() => 0)).map((v) => tx(v, 3)).join(',\\,')).join('\\;|\\;')}]`]
    case 'proj': {
      const m = sel.cell?.m === 'O' ? sel.cell : { r: i, c: 0 }
      const cc = matmulCell(t.concat, Wo, m.r, m.c)
      return [`o_{${m.r + 1},${m.c + 1}} = H_{${m.r + 1}} \\cdot W_O[:,${m.c + 1}]`, `= ${sum(cc.a, cc.b)}`, `= ${tx(cc.sum, 4)}`]
    }
    case 'output': return [`\\mathrm{MultiHead}(X)_{${i + 1}} = ${vec(t.out[i], 4)}`]
  }
}

export function Inspector({ trace, stage }: { trace: AttnTrace; stage: Stage }) {
  const r = useRoles()
  const { depth } = useApp()
  const tokens = useTokens(), sel = useAttn((s) => s.sel), params = useAttn((s) => s.params)
  const h = Math.min(sel.h, trace.cfg.H - 1), hd = trace.heads[h], i = Math.min(sel.i, trace.Tq - 1), j = Math.min(sel.j, trace.Tk - 1)
  const lines = substitution(stage.id, trace, sel, params.Wo)
  const shapes = shapeList(trace)
  const active = SHAPES_FOR[stage.id]
  const g = aggregationTerms(hd, i)
  const explain = depth === 'beginner' ? stage.beginner : depth === 'advanced' ? stage.advanced : stage.intermediate
  return (
    <div className="grid gap-4 text-[13px]" data-testid="attn-inspector">
      <section>
        <div className="eyebrow mb-1">active equation · {stage.title}</div>
        <div className="overflow-x-auto rounded-lg border border-cyan/40 bg-cyan/5 px-2 py-2 scroll-thin" data-testid="active-equation"><Tex tex={stage.tex} block /></div>
        <p className="mt-1 text-[12.5px] text-muted">{stage.purpose}</p>
      </section>
      <section>
        <div className="eyebrow mb-1">numerical substitution · query {i + 1} “{tokens[i]}”, key {j + 1} “{tokens[j]}”, head {h + 1}</div>
        <div className="grid gap-1 overflow-x-auto scroll-thin" data-testid="substitution">{lines.map((l, k) => <Tex key={k} tex={l} block />)}</div>
      </section>
      <section>
        <div className="eyebrow mb-1">the selected pair through the pipeline</div>
        <table className="w-full font-mono text-[12px]" data-testid="pair-table">
          <tbody>
            <tr><td className="text-muted"><span style={{ color: r.q }}>{GLYPH.q}</span> q{i + 1}</td><td>[{hd.Q[i].map((v) => fx(v, 3)).join(', ')}]</td></tr>
            <tr><td className="text-muted"><span style={{ color: r.k }}>{GLYPH.k}</span> k{j + 1}</td><td>[{hd.K[j].map((v) => fx(v, 3)).join(', ')}]</td></tr>
            <tr><td className="text-muted"><span style={{ color: r.score }}>{GLYPH.score}</span> s = q·k</td><td>{fx(hd.S[i][j], 4)}</td></tr>
            <tr><td className="text-muted">s̃ {trace.cfg.scale ? `= s/√${trace.dk}` : '(unscaled)'}</td><td>{fx(hd.scaled[i][j], 4)}</td></tr>
            <tr><td className="text-muted"><span style={{ color: r.mask }}>{GLYPH.mask}</span> mask</td><td style={{ color: trace.M[i][j] === 0 ? undefined : r.mask }}>{trace.M[i][j] === 0 ? 'allowed (0)' : `−∞ (${trace.reasons[i][j].join(' + ')})`}</td></tr>
            <tr><td className="text-muted"><span style={{ color: r.prob }}>{GLYPH.prob}</span> a = softmax</td><td>{fx(hd.A[i][j], 4)}</td></tr>
            <tr><td className="text-muted"><span style={{ color: r.v }}>{GLYPH.v}</span> a·v{j + 1}</td><td>[{g.contrib[j].map((v) => fx(v, 4)).join(', ')}]</td></tr>
            <tr><td className="text-muted"><span style={{ color: r.out }}>{GLYPH.out}</span> o{i + 1} (head {h + 1})</td><td>[{hd.O[i].map((v) => fx(v, 4)).join(', ')}]</td></tr>
          </tbody>
        </table>
        {trace.fullyMaskedRows.includes(i) && <p className="mt-1 flex items-center gap-1 text-[12px] text-rose"><AlertTriangle size={13} />Row {i + 1} is fully masked: zero weights, zero output.</p>}
      </section>
      <section>
        <div className="eyebrow mb-1">tensor shapes (n = {trace.Tq}, d_model = {trace.dModel}, h = {trace.cfg.H}, d_k = d_v = {trace.dk})</div>
        <table className="w-full font-mono text-[12px]" data-testid="shape-table">
          <tbody>
            {shapes.map((s) => {
              const on = active.includes(s.name)
              return <tr key={s.name} className={on ? 'text-fg' : 'text-faint'}><td className="py-0.5 pr-2">{on ? '▸ ' : ''}<Tex tex={s.tex} /></td><td className="pr-2">{s.shape[0]}×{s.shape[1]}</td><td className="text-[11px]">{s.note}</td></tr>
            })}
          </tbody>
        </table>
      </section>
      <section>
        <div className="eyebrow mb-1">explanation · {depth}</div>
        <p className="leading-relaxed" data-testid="explain">{explain}</p>
      </section>
      <section>
        <div className="eyebrow mb-1">common misconceptions</div>
        <ul className="grid gap-1 text-[12.5px]">{stage.misconceptions.map((m) => <li key={m} className="flex gap-1.5"><AlertTriangle size={13} className="mt-0.5 shrink-0 text-amber" /><span>{m}</span></li>)}</ul>
      </section>
    </div>
  )
}
