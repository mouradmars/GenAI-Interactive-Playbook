import { useState } from 'react'
import { AppStateProvider } from './sim/appState'
import { Shell } from './components/Shell'
import InsideLLM from './modules/llm/InsideLLM'
import NGramLab from './modules/ngram/NGramLab'
import EmbedLab from './modules/embed/EmbedLab'
import SeqLab from './modules/seq/SeqLab'
import MlpLab from './modules/mlp/MlpLab'
import TransformerLab from './modules/tblock/TransformerLab'
import RagLab from './modules/rag/RagLab'
import AttentionLab from './modules/attn/AttentionLab'
import { PlannedModule } from './modules/PlannedModule'
import { MODULES } from './modules/registry'

export default function App() {
  const [active, setActive] = useState(() => {
    const h = typeof location !== 'undefined' ? location.hash.slice(1) : ''
    return MODULES.some((m) => m.id === h) ? h : 'llm'
  })
  const open = (id: string) => { setActive(id); try { history.replaceState(null, '', `#${id}`) } catch { /* sandboxed */ } }
  return (
    <AppStateProvider>
      <Shell active={active} onSelect={open}>
        {active === 'llm' ? <InsideLLM /> : active === 'ngram' ? <NGramLab /> : active === 'embeddings' ? <EmbedLab /> : active === 'rnn' ? <SeqLab /> : active === 'mlp' ? <MlpLab /> : active === 'transformer' ? <TransformerLab /> : active === 'rag' ? <RagLab /> : active === 'attention' ? <AttentionLab /> : <PlannedModule id={active} onOpen={open} />}
      </Shell>
    </AppStateProvider>
  )
}
