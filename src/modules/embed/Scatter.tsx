/** The main embedding map: SVG scatter with zoom/pan, selection, neighbour links and arrows. */
import { select, zoom, zoomIdentity, type ZoomBehavior } from 'd3'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useColors, type Colors } from '../../sim/appState'
import { euclid, type Vec } from '../../engine/linalg'
import { nearest, type Space } from '../../engine/embeddings'
import { GROUP_ORDER } from './context'

export function groupColor(c: Colors, groups: string[]) {
  const palette = [c.cyan, c.amber, c.violet, c.teal, c.rose, '#7fb4ff', '#e8a0ff', '#9be37a', '#ffd479', '#5fd1e8', '#ff9f6b', '#b8c4d6', c.faint, c.lineStrong, c.muted]
  const present = GROUP_ORDER.filter((g) => groups.includes(g)).concat([...new Set(groups)].filter((g) => !GROUP_ORDER.includes(g)))
  const m = new Map(present.map((g, i) => [g, palette[i % palette.length]]))
  return { color: (g: string) => m.get(g) ?? c.muted, legend: present.map((g) => ({ g, col: m.get(g)! })) }
}

export interface Arrow { from: Vec; to: Vec; label: string; color: string; dashed?: boolean }

interface Props {
  space: Space
  coords: Vec[]
  focus: number
  selected: number[]
  onFocus: (i: number) => void
  onToggle?: (i: number) => void
  /** draw lines to the focus word's k true (full-dimensional) neighbours */
  neighbourK?: number
  arrows?: Arrow[]
  extraPoints?: { at: Vec; label: string; color: string }[]
  highlightPair?: [number, number] | null
  height?: number
  labelAll?: boolean
  ariaLabel?: string
  /** fit the initial view to these word indices (plus arrows and extra points) instead of all words */
  fitTo?: number[]
}

export function Scatter2D({ space, coords, focus, selected, onFocus, onToggle, neighbourK = 0, arrows = [], extraPoints = [], highlightPair, height = 460, labelAll, ariaLabel, fitTo }: Props) {
  const c = useColors()
  const svgRef = useRef<SVGSVGElement>(null)
  const gRef = useRef<SVGGElement>(null)
  const zoomRef = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const [k, setK] = useState(1)
  const [hover, setHover] = useState<number | null>(null)
  const W = 800, H = height, pad = 36
  const all = useMemo(() => [...coords, ...extraPoints.map((p) => p.at), ...arrows.flatMap((a) => [a.from, a.to])], [coords, extraPoints, arrows])
  // Initial view: with many points, fit the 2nd–98th percentile box so a few far-away words cannot squash
  // everything else. Equal aspect keeps distances honest; points outside the box are counted and reachable by zooming out.
  const robust = coords.length > 40
  const sc = useMemo(() => {
    const q = (v: number[], t: number) => { const s2 = v.slice().sort((a, b) => a - b); return s2[Math.min(s2.length - 1, Math.max(0, Math.round(t * (s2.length - 1))))] }
    const base = fitTo?.length ? fitTo.map((i) => coords[i]).filter(Boolean) : coords
    const xs = base.map((p) => p[0]), ys = base.map((p) => p[1])
    let [x0, x1, y0, y1] = robust && !fitTo?.length ? [q(xs, 0.02), q(xs, 0.98), q(ys, 0.02), q(ys, 0.98)] : [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
    for (const p of all.slice(coords.length)) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]) }
    if (fitTo?.length) { const mx = (x1 - x0) * 0.35 || 0.5, my = (y1 - y0) * 0.35 || 0.5; x0 -= mx; x1 += mx; y0 -= my; y1 += my }
    const s = Math.min((W - 2 * pad) / (x1 - x0 || 1), (H - 2 * pad) / (y1 - y0 || 1))
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2
    return (p: Vec): [number, number] => [W / 2 + (p[0] - cx) * s, H / 2 - (p[1] - cy) * s]
  }, [all, coords, H, robust, fitTo])
  const outside = coords.filter((p) => { const [x, y] = sc(p); return x < 0 || x > W || y < 0 || y > H }).length
  const { color, legend } = groupColor(c, space.groups)
  const sel = new Set(selected)
  const nn = useMemo(() => (neighbourK > 0 && focus >= 0 ? nearest(space, space.vecs[focus], neighbourK, 'cosine', [focus]) : []), [space, focus, neighbourK])
  const nn2d = useMemo(() => {
    if (neighbourK <= 0 || focus < 0) return new Set<number>()
    return new Set(coords.map((p, i) => [euclid(p, coords[focus]), i] as const).filter(([, i]) => i !== focus).sort((a, b) => a[0] - b[0]).slice(0, neighbourK).map(([, i]) => i))
  }, [coords, focus, neighbourK])

  useEffect(() => {
    if (!svgRef.current) return
    const z = zoom<SVGSVGElement, unknown>().scaleExtent([0.15, 12]).on('zoom', (e) => {
      gRef.current?.setAttribute('transform', e.transform.toString())
      setK(e.transform.k)
    })
    zoomRef.current = z
    select(svgRef.current).call(z)
    return () => { select(svgRef.current!).on('.zoom', null) }
  }, [])
  // reset the view when the data changes shape
  useEffect(() => { if (svgRef.current && zoomRef.current) select(svgRef.current).call(zoomRef.current.transform, zoomIdentity) }, [space.id, coords.length])

  // points drawn on top of each other (projection collapsed them) share one combined label
  const labelGroups = useMemo(() => {
    const leader = new Map<number, number>(), text = new Map<number, string>(), membersOf = new Map<number, number[]>()
    const pos = coords.map(sc)
    const order = coords.map((_, i) => i).sort((x, y) => (x === focus ? -1 : y === focus ? 1 : x - y))
    for (const i of order) {
      if (leader.has(i)) continue
      const members = [i]
      for (const j of order) if (j !== i && !leader.has(j) && Math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]) < 2.5) members.push(j)
      members.forEach((m) => leader.set(m, i))
      text.set(i, members.map((m) => space.words[m]).join(' · '))
      membersOf.set(i, members)
    }
    return { leader, text, members: membersOf }
  }, [coords, sc, space.words, focus])
  const showLabel = (i: number) => labelAll || space.words.length <= 40 || i === focus || sel.has(i) || i === hover || nn.some((x) => x.i === i) || (highlightPair?.includes(i) ?? false) || k > 2.2
  const r = 5 / Math.sqrt(k)
  const fs = 12 / Math.sqrt(k)
  return (
    <div className="relative">
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={ariaLabel ?? 'embedding map'} style={{ touchAction: 'none', cursor: 'grab', display: 'block', maxHeight: '70vh', overflow: 'hidden' }}>
        <defs>
          {[...new Set(arrows.map((a) => a.color))].map((col, i) => (
            <marker key={col} id={`emb-arrow-${i}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill={col} /></marker>
          ))}
        </defs>
        <rect x={0} y={0} width={W} height={H} fill="transparent" />
        <g ref={gRef}>
          {highlightPair && (() => { const [a, b] = highlightPair; const [x1, y1] = sc(coords[a]), [x2, y2] = sc(coords[b]); return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={c.rose} strokeWidth={2.5 / Math.sqrt(k)} /> })()}
          {nn.map((x) => {
            const [x1, y1] = sc(coords[focus]), [x2, y2] = sc(coords[x.i])
            const agree = nn2d.has(x.i)
            return <line key={x.i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={agree ? c.cyan : c.amber} strokeOpacity={0.85} strokeWidth={1.6 / Math.sqrt(k)} strokeDasharray={agree ? undefined : `${5 / k} ${4 / k}`} />
          })}
          {arrows.map((a, i) => {
            const [x1, y1] = sc(a.from), [x2, y2] = sc(a.to)
            const mi = [...new Set(arrows.map((q) => q.color))].indexOf(a.color)
            return (
              <g key={i}>
                <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={a.color} strokeWidth={2.2 / Math.sqrt(k)} strokeDasharray={a.dashed ? `${6 / k} ${4 / k}` : undefined} markerEnd={`url(#emb-arrow-${mi})`} />
                <text x={(x1 + x2) / 2 + 6} y={(y1 + y2) / 2 - 6} fontSize={fs} fill={a.color} fontFamily="JetBrains Mono, monospace">{a.label}</text>
              </g>
            )
          })}
          {coords.map((p, i) => {
            const [x, y] = sc(p)
            const isF = i === focus, isS = sel.has(i)
            return (
              <g key={i} style={{ cursor: 'pointer' }} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
                onClick={(e) => { e.stopPropagation(); if ((e.shiftKey || e.metaKey) && onToggle) onToggle(i); else onFocus(i) }}>
                <circle cx={x} cy={y} r={isF ? r * 1.9 : isS ? r * 1.4 : r} fill={color(space.groups[i])} fillOpacity={isF || isS ? 1 : 0.8}
                  stroke={isF ? c.fg : isS ? c.amber : c.bg} strokeWidth={(isF ? 2 : 1) / Math.sqrt(k)} />
                {labelGroups.leader.get(i) === i && (labelGroups.members.get(i) ?? [i]).some(showLabel) && <text x={x + r + 3} y={y + fs / 3} fontSize={isF ? fs * 1.15 : fs} fill={isF ? c.fg : c.muted} fontWeight={isF ? 600 : 400}
                  fontFamily="JetBrains Mono, monospace" style={{ pointerEvents: 'none', paintOrder: 'stroke' }} stroke={c.bg} strokeWidth={3 / Math.sqrt(k)}>{labelGroups.text.get(i)}</text>}
              </g>
            )
          })}
          {extraPoints.map((pt, i) => {
            const [x, y] = sc(pt.at)
            return (
              <g key={`x${i}`}>
                <path d={`M${x - 7 / Math.sqrt(k)},${y} L${x},${y - 7 / Math.sqrt(k)} L${x + 7 / Math.sqrt(k)},${y} L${x},${y + 7 / Math.sqrt(k)} Z`} fill={pt.color} stroke={c.bg} strokeWidth={1} />
                <text x={x + 10} y={y - 8} fontSize={fs} fill={pt.color} fontFamily="JetBrains Mono, monospace" fontWeight={600}>{pt.label}</text>
              </g>
            )
          })}
        </g>
      </svg>
      <div className="pointer-events-none absolute left-2 top-2 flex max-w-[70%] flex-wrap gap-x-3 gap-y-1 font-mono text-[10.5px] text-muted">
        {legend.map(({ g, col }) => <span key={g} className="flex items-center gap-1"><i className="inline-block h-2 w-2 rounded-full" style={{ background: col }} />{g}</span>)}
      </div>
      <div className="absolute bottom-2 right-2 flex items-center gap-1">
        <button type="button" className="btn h-7 px-2 text-[11px]" onClick={() => svgRef.current && zoomRef.current && select(svgRef.current).call(zoomRef.current.scaleBy, 1.5)} aria-label="Zoom in">+</button>
        <button type="button" className="btn h-7 px-2 text-[11px]" onClick={() => svgRef.current && zoomRef.current && select(svgRef.current).call(zoomRef.current.scaleBy, 1 / 1.5)} aria-label="Zoom out">−</button>
        <button type="button" className="btn h-7 px-2 text-[11px]" onClick={() => svgRef.current && zoomRef.current && select(svgRef.current).call(zoomRef.current.transform, zoomIdentity)}>Reset view</button>
      </div>
      {outside > 0 && k <= 1.01 && (
        <div className="pointer-events-none absolute right-2 top-2 rounded border border-line bg-panel/90 px-2 py-0.5 font-mono text-[10.5px] text-amber" role="note">
          {outside} outlying {outside === 1 ? 'word lies' : 'words lie'} outside this view · zoom out (−) to see {outside === 1 ? 'it' : 'them'}
        </div>
      )}
      {neighbourK > 0 && (
        <div className="pointer-events-none absolute bottom-2 left-2 font-mono text-[10.5px] text-muted">
          <span className="text-cyan">—</span> true neighbour, also close on screen · <span className="text-amber">- -</span> true neighbour the picture puts far away
        </div>
      )}
    </div>
  )
}
