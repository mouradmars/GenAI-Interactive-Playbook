import { createContext, useContext } from 'react'
import type { ForwardTrace, NanoGPT } from '../../engine/model'
import type { DecodeConfig, DecodeStep } from '../../engine/decoding'
import type { JourneyStep } from './journey'

export interface Prediction { choice: number | null; revealed: boolean }

export interface LabCtx {
  model: NanoGPT
  trace: ForwardTrace
  step: JourneyStep
  prompt: string
  setPrompt: (p: string) => void
  generated: number[]
  appendToken: (id: number) => void
  undoToken: () => void
  sel: number
  setSel: (i: number) => void
  head: number
  setHead: (h: number) => void
  cell: [number, number] | null
  setCell: (c: [number, number] | null) => void
  decode: DecodeConfig
  setDecode: (d: DecodeConfig) => void
  decodeResult: DecodeStep
  sampleU: number
  predictMode: boolean
  prediction: Prediction
  setPrediction: (p: Prediction) => void
  unknown: string[]
}

export const Lab = createContext<LabCtx | null>(null)
export const useLab = () => {
  const c = useContext(Lab)
  if (!c) throw new Error('useLab outside provider')
  return c
}

/** Last-position slice of a [n, V] matrix. */
export const lastRow = (t: ForwardTrace) => t.logits.data.subarray((t.ids.length - 1) * t.logits.cols, t.ids.length * t.logits.cols)
