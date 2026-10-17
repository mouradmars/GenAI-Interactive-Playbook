/**
 * Microscope C · computational pipeline view: tensors as nodes with their shapes, operations as edges, the
 * current operation's inputs and outputs highlighted, and the selected (query, key, head) element of each tensor.
 */
import { useColors } from '../../sim/appState'
import type { AttnTrace } from '../../engine/attention'
import { fx, useRoles, type Role } from './common'
import { STAGES, stageIndex, type NodeId, type Stage, type StageId } from './stages'
import { useAttn } from './store'

const WRITER: Record<NodeId, StageId> = { X: 'embed', W: 'qkv', Q: 'qkv', K: 'qkv', V: 'qkv', S: 'dot', Ss: 'scale', M: 'mask', A: 'softmax', Oh: 'aggregate', H: 'concat', Wo: 'proj', O: 'proj' }

export function PipelineView({ trace, stage, onSeek, compact = false }: { trace: AttnTrace; stage: Stage; onSeek?: (i: number) => void; compact?: boolean }) {
  const c = useColors(), r = useRoles()
  const sel = useAttn((s) => s.sel)
  const { Tq: n, Tk: m, dModel: d, dk, cfg } = trace
  const h = Math.min(sel.h, cfg.H - 1), hd = trace.heads[h], i = Math.min(sel.i, n - 1), j = Math.min(sel.j, m - 1)
  const sub = (k: number) => String(k + 1)
  const nodes: { id: NodeId; label: string; shape: string; col: number; row: number; role: Role; detail: string }[] = [
    { id: 'X', label: 'X', shape: `${n}×${d}`, col: 0, row: 1, role: 'x', detail: `x${sub(i)}` },
    { id: 'W', label: `W_Q,K,V^(${h + 1})`, shape: `${d}×${dk} each`, col: 1, row: 0, role: 'w', detail: 'learned' },
    { id: 'Q', label: `Q${sub(h)}`, shape: `${n}×${dk}`, col: 2, row: 0, role: 'q', detail: `q${sub(i)}` },
    { id: 'K', label: `K${sub(h)}`, shape: `${m}×${dk}`, col: 2, row: 1, role: 'k', detail: `k${sub(j)}` },
    { id: 'V', label: `V${sub(h)}`, shape: `${m}×${dk}`, col: 2, row: 2, role: 'v', detail: `v${sub(j)}` },
    { id: 'S', label: 'S = QKᵀ', shape: `${n}×${m}`, col: 3, row: 0, role: 'score', detail: `s=${fx(hd.S[i][j], 2)}` },
    { id: 'Ss', label: cfg.scale ? 'S/√dₖ' : 'S (unscaled)', shape: `${n}×${m}`, col: 4, row: 0, role: 'score', detail: `${fx(hd.scaled[i][j], 2)}` },
    { id: 'M', label: 'mask M', shape: `${n}×${m}`, col: 4, row: 1, role: 'mask', detail: trace.M[i][j] === 0 ? 'allowed' : '−∞' },
    { id: 'A', label: 'A = softmax', shape: `${n}×${m}`, col: 5, row: 0, role: 'prob', detail: `a=${fx(hd.A[i][j], 3)}` },
    { id: 'Oh', label: `head${sub(h)} = AV`, shape: `${n}×${dk}`, col: 6, row: 1, role: 'out', detail: `o${sub(i)}` },
    { id: 'H', label: 'Concat', shape: `${n}×${cfg.H * dk}`, col: 7, row: 1, role: 'out', detail: `${cfg.H} head${cfg.H > 1 ? 's' : ''}` },
    { id: 'Wo', label: 'W_O', shape: `${cfg.H * dk}×${d}`, col: 7, row: 2, role: 'w', detail: 'learned' },
    { id: 'O', label: 'O = HW_O', shape: `${n}×${d}`, col: 8, row: 1, role: 'out', detail: `row ${i + 1}` },
  ]
  const links: [NodeId, NodeId][] = [['X', 'Q'], ['X', 'K'], ['X', 'V'], ['W', 'Q'], ['W', 'K'], ['W', 'V'], ['Q', 'S'], ['K', 'S'], ['S', 'Ss'], ['Ss', 'A'], ['M', 'A'], ['A', 'Oh'], ['V', 'Oh'], ['Oh', 'H'], ['H', 'O'], ['Wo', 'O']]
  const colW = compact ? 100 : 112, rowH = compact ? 54 : 62, bw = compact ? 90 : 100, bh = compact ? 42 : 48
  const W = 9 * colW + 8, H = 3 * rowH + 8
  const pos = (id: NodeId) => { const nd = nodes.find((x) => x.id === id)!; return { x: 4 + nd.col * colW, y: 4 + nd.row * rowH } }
  const active = new Set([...stage.reads, ...stage.writes])
  return (
    <div className="overflow-x-auto scroll-thin">
      <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" style={{ minWidth: W * 0.72 }} role="img" aria-label="Computational pipeline with tensor shapes" data-testid="pipeline-view">
        <defs><marker id="pv-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill={c.lineStrong} /></marker></defs>
        {links.map(([a, b]) => {
          const p = pos(a), q = pos(b)
          const x1 = p.x + bw, y1 = p.y + bh / 2, x2 = q.x - 2, y2 = q.y + bh / 2
          const on = stage.writes.includes(b) && stage.reads.includes(a)
          return <path key={`${a}${b}`} d={`M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}`} fill="none" stroke={on ? r.score : c.lineStrong} strokeWidth={on ? 2.2 : 1.1} markerEnd="url(#pv-arrow)" opacity={on ? 1 : 0.7} />
        })}
        {nodes.map((nd) => {
          const p = pos(nd.id), isW = stage.writes.includes(nd.id), isR = stage.reads.includes(nd.id)
          const col = r[nd.role]
          return (
            <g key={nd.id} transform={`translate(${p.x}, ${p.y})`} style={onSeek ? { cursor: 'pointer' } : undefined} onClick={onSeek ? () => onSeek(stageIndex(WRITER[nd.id])) : undefined}
              data-testid={`pv-${nd.id}`} data-active={active.has(nd.id) ? (isW ? 'write' : 'read') : 'no'}>
              <title>{`${nd.label} · ${nd.shape}${isW ? ' · produced by this operation' : isR ? ' · read by this operation' : ''}`}</title>
              <rect width={bw} height={bh} rx={8} fill={isW ? `color-mix(in oklab, ${col} 18%, ${c.panel})` : c.panel} stroke={col} strokeWidth={isW ? 2.6 : isR ? 1.8 : 0.9} strokeDasharray={isR && !isW ? '4 3' : undefined} opacity={active.size && !active.has(nd.id) ? 0.55 : 1} />
              <text x={bw / 2} y={15} textAnchor="middle" fontSize={10.5} fill={col} fontFamily="var(--font-mono)">{nd.label}</text>
              <text x={bw / 2} y={29} textAnchor="middle" fontSize={10} fill={c.fg} fontFamily="var(--font-mono)">[{nd.shape}]</text>
              {!compact && <text x={bw / 2} y={42} textAnchor="middle" fontSize={9.5} fill={c.muted} fontFamily="var(--font-mono)">{nd.detail}</text>}
            </g>
          )
        })}
      </svg>
      <p className="mt-1 text-[11px] text-faint">Solid outline: produced by the current operation ({STAGES[stageIndex(stage.id)].short}) · dashed: read by it · click a tensor to jump to the operation that produces it.</p>
    </div>
  )
}
