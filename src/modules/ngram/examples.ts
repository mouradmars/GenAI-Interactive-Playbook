/** Preset experiments for the n-gram lab. All corpora are original text written for this course. */
import nanoCorpus from '../../data/corpus-sample.txt?raw'

export interface NGramExample {
  id: string
  title: string
  corpus: string
  test: string
  heldOut: string
  why: string
  /** guided-mode stage this example is best opened at */
  stage: string
}

const nanoSentences = nanoCorpus.split('.').map((s) => s.trim()).filter(Boolean)

export const NGRAM_EXAMPLES: NGramExample[] = [
  {
    id: 'textbook',
    title: 'The textbook bigram example',
    corpus: 'I am Lina. Lina I am. I do not like cold tea and figs.',
    test: 'I am Lina. I like tea.',
    heldOut: 'Lina I am. I do not like tea.',
    why: 'Three sentences, small enough to verify every count by hand. It has the same structure as the classic example in Jurafsky & Martin, chapter 3: P(i | <s>) = 2/3, P(am | i) = 2/3. The second test sentence contains “i like”, a bigram the corpus never saw, so maximum likelihood gives the whole test text probability 0.',
    stage: 'mle',
  },
  {
    id: 'weather',
    title: 'Weather reports: when trigrams help',
    corpus: [
      'tomorrow will be sunny and warm.', 'tomorrow will be cloudy and cool.', 'today was sunny and warm.', 'today was rainy and cold.',
      'the morning will be foggy and cool.', 'the evening was clear and cold.', 'tonight will be clear and cold.', 'the weekend will be sunny and hot.',
      'tomorrow morning will be rainy and cool.', 'today was cloudy and warm.', 'the night was rainy and cold.', 'tomorrow will be windy and cool.',
      'the weekend was cloudy and cool.', 'tonight will be foggy and cold.', 'the afternoon will be sunny and hot.', 'today was windy and cold.',
    ].join(' '),
    test: 'tomorrow will be sunny and hot. the night will be clear and cold.',
    heldOut: 'today was clear and warm. the weekend will be rainy and cool.',
    why: 'Short, formulaic sentences. A trigram model knows that “will be” is followed by a weather word, but it has never seen “night will”. Compare bigram and trigram perplexity, and see how interpolation and backoff cope with the unseen trigram.',
    stage: 'interp',
  },
  {
    id: 'francisco',
    title: 'San Francisco versus glasses (Kneser-Ney)',
    corpus: [
      'we flew to san francisco.', 'san francisco is foggy.', 'she lives in san francisco.', 'san francisco has hills.', 'they love san francisco.',
      'san francisco is far.', 'i cleaned my glasses.', 'he lost his glasses.', 'new glasses help.', 'reading glasses are cheap.', 'she wears glasses.',
      'i need my reading lamp.', 'we need new shoes.', 'he lost his keys.', 'i cleaned my desk.',
    ].join(' '),
    test: 'i lost my glasses. she cleaned her glasses.',
    heldOut: 'he needs new glasses. we visited san francisco.',
    why: '“francisco” and “glasses” occur about equally often, but “francisco” only ever follows “san”. For an unseen context like “her ___”, Kneser-Ney backs off to continuation counts and prefers “glasses”, while plain backoff to unigram counts slightly prefers “francisco”, the more frequent word.',
    stage: 'kn',
  },
  {
    id: 'nano',
    title: 'The Observatory-Nano training corpus',
    corpus: nanoSentences.slice(0, 160).join('. ') + '.',
    test: 'the cat likes to eat fish. paris is the capital of france.',
    heldOut: nanoSentences.slice(160, 200).join('. ') + '.',
    why: 'The first 160 sentences of the corpus that trained the transformer in module 01. The grammar is regular, so a trigram model reaches a low perplexity here. It cannot generalize across words, though: it has no idea that “cat” and “dog” behave alike.',
    stage: 'compare',
  },
]
