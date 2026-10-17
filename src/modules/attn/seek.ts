import type { StageId } from './stages'
/** lets lessons, challenges and experiments move the guided timeline (registered by AttentionLab) */
export const seekRef: { current: (id: StageId) => void } = { current: () => {} }
