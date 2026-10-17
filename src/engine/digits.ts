/**
 * The trained 64 → 16 → 16 → 10 digit classifier (training/train_digits.py) and its data:
 * UCI 8×8 handwritten digits, pixel values 0..16, inputs scaled by 1/16, stratified 1,347 / 450 split.
 */
import raw from '../data/digits.json'
import { cloneParams, type NetConfig, type Params, type Sample } from './nn'

interface DigitsFile {
  source: string; sizes: number[]; acts: NetConfig['acts']; loss: NetConfig['loss']
  training: { optimizer: string; lr: number; batch: number; epochs: number; init: string; seed: number; inputScale: string }
  X: number[][]; y: number[]; train: number[]; test: number[]; W: number[][][]; b: number[][]
  metrics: { trainLoss: number; trainAcc: number; testLoss: number; testAcc: number; confusion: number[][] }
  curve: { epoch: number; trainLoss: number; trainAcc: number; testLoss: number; testAcc: number }[]
  reference: { ids: number[]; probs: number[][]; logits: number[][] }
}
export const DIGITS = raw as unknown as DigitsFile
export const DIGITS_CFG: NetConfig = { sizes: DIGITS.sizes, acts: DIGITS.acts, loss: DIGITS.loss }
export const trainedDigitParams = (): Params => cloneParams({ W: DIGITS.W, b: DIGITS.b })
export const digitSample = (i: number): Sample => ({ x: DIGITS.X[i].map((v) => v / 16), y: Array.from({ length: 10 }, (_, k) => (k === DIGITS.y[i] ? 1 : 0)), label: DIGITS.y[i] })
export const fromPixels = (px: number[], label?: number): Sample => ({ x: px.map((v) => v / 16), y: Array.from({ length: 10 }, (_, k) => (k === (label ?? 0) ? 1 : 0)), label })
let trainCache: Sample[] | null = null, testCache: Sample[] | null = null
export const digitTrain = () => (trainCache ??= DIGITS.train.map(digitSample))
export const digitTest = () => (testCache ??= DIGITS.test.map(digitSample))
/** are these exactly the exported trained weights? */
export const isTrainedDigits = (cfg: NetConfig, p: Params) =>
  cfg.sizes.join() === DIGITS.sizes.join() && cfg.acts.join() === DIGITS.acts.join() && p.W.every((L, l) => L.every((r, j) => r.every((w, i) => w === DIGITS.W[l][j][i]))) && p.b.every((v, l) => v.every((x, j) => x === DIGITS.b[l][j]))
