/** Stages 1–2: tokens, input vectors, and what X is (and is not). Views only display store/trace values. */
import { Minus, Plus } from 'lucide-react'
import { useMemo } from 'react'
import { Card } from '../../components/viz'
import type { AttnTrace } from '../../engine/attention'
import { EditableMatrix, GLYPH, Grid, RoleTag, Seg, fx, tokLabel, useRoles, useTokens } from './common'
import { D_MODEL, MAX_T, PAD, allTokens, attnStore, embedInput, useAttn } from './store'

const feat = Array.from({ length: D_MODEL }, (_, k) => `f${k + 1}`)

/** token cards for the current window: click = query (and key with the second button) */
export function TokenCards({ showKey = true }: { showKey?: boolean }) {
  const r = useRoles()
  const tokens = useTokens(), sel = useAttn((s) => s.sel), select = useAttn((s) => s.select), pad = useAttn((s) => s.pad)
  return (
    <div className="flex flex-wrap gap-1.5" role="group" aria-label="Tokens in the attention window">
      {tokens.map((t, i) => {
        const q = i === sel.i, k = i === sel.j
        return (
          <div key={i} className="flex items-stretch overflow-hidden rounded-lg border" style={{ borderColor: q ? r.q : k && showKey ? r.k : 'var(--line)', boxShadow: q ? `0 0 0 1px ${r.q}` : undefined }}>
            <button type="button" onClick={() => select({ i, cell: null })} className="flex items-baseline gap-1.5 bg-panel-2 px-2.5 py-1.5 font-mono text-[13px]" aria-pressed={q} aria-label={`Select token ${i + 1} (${t}) as the query`} data-testid={`tok-${i}`}>
              <span className="text-[10px] text-faint">{i + 1}</span>
              <span className={t === PAD || pad[i] ? 'text-faint line-through decoration-rose/60' : 'text-fg'}>{t}</span>
              {q && <span style={{ color: r.q }} aria-hidden="true">{GLYPH.q}</span>}
            </button>
            {showKey && (
              <button type="button" onClick={() => select({ j: i, cell: null })} className={`border-l border-line px-1.5 font-mono text-[10px] ${k ? '' : 'text-faint hover:text-fg'}`} style={k ? { color: r.k } : undefined} aria-pressed={k} aria-label={`Select token ${i + 1} as the key`} title="use as the key">
                {GLYPH.k}
              </button>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function TokensView({ trace }: { trace: AttnTrace }) {
  const sentence = useAttn((s) => s.sentence), start = useAttn((s) => s.windowStart), T = useAttn((s) => s.T), sel = useAttn((s) => s.sel)
  const st = attnStore.getState()
  const all = allTokens(sentence)
  const tokens = useTokens()
  const vocab = useMemo(() => [...new Set(all.map((w) => w.toLowerCase()))], [all])
  const i = sel.i
  return (
    <div className="grid gap-3">
      <Card title="1 · text → tokens (toy word-level tokenizer)">
        <label className="grid gap-1 text-[12.5px] text-muted">sentence
          <input value={sentence} onChange={(e) => st.setSentence(e.target.value)} className="rounded-md border border-line bg-panel-2 px-2 py-1.5 font-mono text-[14px] text-fg" aria-label="Sentence" data-testid="sentence-input" />
        </label>
        <div className="mt-2 flex flex-wrap items-center gap-1" aria-label="All tokens of the sentence">
          {all.map((t, k) => {
            const inWin = k >= start && k < start + T
            return <span key={k} className={`rounded-md border px-2 py-1 font-mono text-[12.5px] ${inWin ? 'border-cyan/60 bg-cyan/10 text-fg' : 'border-line text-faint'}`} title={inWin ? `position ${k - start + 1} of the window` : 'outside the window'}>{t}</span>
          })}
          {all.length === 0 && <span className="text-[12px] text-rose">Type at least one word.</span>}
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-3 text-[12.5px] text-muted">
          <span className="flex items-center gap-1">window starts at token
            <button type="button" className="btn icon-btn !h-6 !w-6" onClick={() => st.setWindow(start - 1)} disabled={start <= 0} aria-label="Move window left"><Minus size={12} /></button>
            <span className="w-5 text-center font-mono text-fg" data-testid="window-start">{start + 1}</span>
            <button type="button" className="btn icon-btn !h-6 !w-6" onClick={() => st.setWindow(start + 1)} disabled={start >= Math.max(0, all.length - 1)} aria-label="Move window right"><Plus size={12} /></button>
          </span>
          <span className="flex items-center gap-1">sequence length n
            <button type="button" className="btn icon-btn !h-6 !w-6" onClick={() => st.setWindow(start, T - 1)} disabled={T <= 2} aria-label="Fewer tokens"><Minus size={12} /></button>
            <span className="w-5 text-center font-mono text-fg" data-testid="seq-len">{T}</span>
            <button type="button" className="btn icon-btn !h-6 !w-6" onClick={() => st.setWindow(start, T + 1)} disabled={T >= MAX_T} aria-label="More tokens"><Plus size={12} /></button>
          </span>
        </div>
        <p className="mt-2 text-[12px] leading-relaxed text-muted">
          The default preset attends over a 4-token window, <span className="font-mono text-fg">cat · chased · the · mouse</span>: the leading “The” and the period fall outside it. Positions past the end of the text become <span className="font-mono">{PAD}</span> tokens, which the padding mask hides. Real models use subword tokenizers and much longer windows.
        </p>
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card title="2 · the window the attention layer sees (click a token)">
          <TokenCards />
          <div className="mt-3 grid gap-1 text-[12.5px]">
            <div><span className="text-faint">selected:</span> token <b>{i + 1}</b> “<span className="font-mono">{tokens[i]}</span>”, position {i + 1} in the window ({start + i + 1} in the sentence){trace.cfg.pad[i] ? ' · padding' : ''}</div>
            <div><span className="text-faint">toy vocabulary id:</span> <span className="font-mono">{tokens[i] === PAD ? '— (pad)' : vocab.indexOf(tokens[i].toLowerCase())}</span> <span className="text-faint">(index of the lower-cased word in this sentence's vocabulary)</span></div>
            <div><span className="text-faint">its row of X:</span> <RoleTag role="x">x{i + 1} = [{trace.X[i].map((v) => fx(v, 2)).join(', ')}]</RoleTag></div>
          </div>
        </Card>
        <Card title={`X ∈ ℝ^${trace.Tq}×${trace.dModel} · the selected token's row is highlighted everywhere`}>
          <Grid m={trace.X} role="x" rowLabels={tokens.map((_, k) => tokLabel(tokens, k))} colLabels={feat} hlRows={[i]} onCell={(rr) => st.select({ i: rr, cell: null })} ariaLabel="Input matrix X" testid="g-X0" />
        </Card>
      </div>
      <FourThings />
    </div>
  )
}

function FourThings() {
  const r = useRoles()
  const items: [string, string, string][] = [
    ['text token', 'a piece of text, e.g. “cat”', r.w],
    ['token embedding', 'a learned row looked up by token id', r.x],
    ['positional information', 'added (or applied) so order can matter', r.k],
    ['hidden state', 'what actually enters an attention layer; in deep layers it already mixes context', r.out],
  ]
  return (
    <Card title="four different things that are easy to confuse">
      <div className="grid gap-2 @2xl:grid-cols-4">
        {items.map(([t, d, col], k) => (
          <div key={t} className="rounded-lg border border-line bg-panel-2/50 p-2 text-[12.5px]">
            <div className="font-mono text-[11px]" style={{ color: col }}>{k + 1} · {t}</div>
            <div className="text-muted">{d}</div>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[12px] text-muted">In this lab the attention layer receives <b className="text-fg">X</b>. By default X is configured by hand for easy arithmetic: renaming a token does <b className="text-fg">not</b> change its vector. Switch the source in stage 2 to build X from a toy embedding table plus sinusoidal positions instead.</p>
    </Card>
  )
}

export function EmbedView({ trace }: { trace: AttnTrace }) {
  const r = useRoles()
  const tokens = useTokens(), sel = useAttn((s) => s.sel), xSource = useAttn((s) => s.xSource), xEdited = useAttn((s) => s.xEdited), start = useAttn((s) => s.windowStart)
  const st = attnStore.getState()
  const up = useMemo(() => embedInput(tokens, start), [tokens, start])
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  const i = sel.i
  const ma = Math.max(1e-9, ...trace.X[i].map(Math.abs))
  return (
    <div className="grid gap-3">
      <Card title="where does X come from?" right={
        <Seg label="Input source" value={xSource} onChange={(v) => st.setXSource(v)} options={[{ v: 'hand', label: 'hand-configured hidden states' }, { v: 'embed', label: 'toy embedding + position' }]} />
      }>
        {xSource === 'hand'
          ? <p className="text-[12.5px] leading-relaxed text-muted">These rows are <b className="text-fg">hidden states entering an attention layer</b>, chosen by hand (one decimal) so every product is easy to check. They do not encode the meaning of the words. Edit any value below; everything downstream recomputes.</p>
          : <p className="text-[12.5px] leading-relaxed text-muted">First-layer style input: <span className="font-mono">x_i = e(token_i) + PE(position_i)</span>, with a seeded toy embedding table (not trained) and the sinusoidal encoding of Vaswani et al. (2017). The same word at two positions gets the same e but a different PE.</p>}
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,0.8fr)]">
        <Card title={<span><RoleTag role="x">X</RoleTag> ∈ ℝ^{trace.Tq}×{trace.dModel} · rows = positions, columns = features{xEdited ? ' · edited' : ''}</span>}>
          {xSource === 'hand'
            ? <EditableMatrix m={trace.X} onSet={(rr, c, v) => st.setX(rr, c, v)} rowLabels={rl} colLabels={feat} role="x" label="X" testid="edit-X" />
            : <Grid m={trace.X} role="x" rowLabels={rl} colLabels={feat} hlRows={[i]} onCell={(rr) => st.select({ i: rr })} ariaLabel="Input matrix X" digits={3} />}
        </Card>
        <Card title={`token ${i + 1} “${tokens[i]}” · its input vector`}>
          <div className="grid gap-1">
            {trace.X[i].map((v, k) => (
              <div key={k} className="grid grid-cols-[28px_1fr_58px] items-center gap-2 font-mono text-[12px]">
                <span className="text-faint">f{k + 1}</span>
                <div className="relative h-3 rounded bg-panel-2"><div className="absolute top-0 h-3 rounded" style={{ left: v >= 0 ? '50%' : `${50 - (Math.abs(v) / ma) * 50}%`, width: `${(Math.abs(v) / ma) * 50}%`, background: v >= 0 ? r.x : 'var(--muted)' }} /></div>
                <span className="text-right">{fx(v, 2)}</span>
              </div>
            ))}
          </div>
          <p className="mt-2 text-[11.5px] text-faint">‖x{i + 1}‖ = {fx(Math.hypot(...trace.X[i]), 3)}</p>
        </Card>
      </div>
      {xSource === 'embed' && (
        <Card title="X = E + PE">
          <div className="flex flex-wrap items-start gap-3">
            <Grid m={up.E} role="x" rowLabels={rl} colLabels={feat} title="E · toy token embeddings" ariaLabel="Token embeddings" cell={46} hlRows={[i]} />
            <span className="self-center font-mono text-lg">+</span>
            <Grid m={up.P} role="k" rowLabels={tokens.map((_, k) => `pos ${start + k}`)} colLabels={feat} title="PE · sinusoidal positions" ariaLabel="Positional encodings" cell={46} digits={3} hlRows={[i]} />
            <span className="self-center font-mono text-lg">=</span>
            <Grid m={up.X} role="x" rowLabels={rl} colLabels={feat} title="X" ariaLabel="Constructed input" cell={46} digits={3} hlRows={[i]} />
          </div>
        </Card>
      )}
    </div>
  )
}
