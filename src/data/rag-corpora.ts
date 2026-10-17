/**
 * Example knowledge bases for module 11 (RAG). Two are written for this lab (the course handbook is fictional);
 * the solar-system notes use standard astronomy facts and deliberately include one outdated document.
 * Each example comes with questions whose gold evidence is an exact substring of one document.
 */
export interface RagDoc { id: string; title: string; text: string; note?: string }
export interface RagQuestion {
  id: string; q: string
  /** exact substring of a document that answers the question ('' for unanswerable questions) */
  gold: string
  /** short key that a correct answer must contain ('' → the right behaviour is to abstain) */
  key: string
  /** the failure mode this question is designed to expose */
  tag: 'baseline' | 'lexical-gap' | 'boundary' | 'conflict' | 'no-answer' | 'distractor' | 'multi-hop'
  /** educational simulation of a closed-book model (no retrieval): fluent, sometimes wrong */
  closedBook: string
  closedBookCorrect: boolean
}
export interface RagCorpus { id: string; title: string; blurb: string; docs: RagDoc[]; questions: RagQuestion[] }

export const CORPORA: RagCorpus[] = [
  {
    id: 'course', title: 'Course handbook (fictional CS 410)', blurb: 'Policies of a fictional machine-learning course: grading, late work, AI tools, exams, office hours.',
    docs: [
      { id: 'grading', title: 'Grading', text: 'The final grade in CS 410 combines four components. Weekly problem sets count for 30 percent of the grade. The course project counts for 25 percent. The midterm exam counts for 15 percent and the final exam counts for 30 percent. Grades are curved only upward; nobody receives a lower letter grade because of the curve. Regrade requests must be submitted within seven days after grades are released.' },
      { id: 'late', title: 'Late work', text: 'Problem sets are due on Fridays at 17:00. Every student receives one extension token that moves a single deadline by 48 hours without any penalty. Apart from the token, late submissions lose five percentage points for each day they are late. Submissions that are more than four days late are not accepted and receive zero points. The project deadline cannot be extended with the token.' },
      { id: 'ai', title: 'Generative AI assistants', text: 'You may use generative AI assistants such as chatbots and code completion tools to explain concepts and to debug code. You may not paste problem set questions into an assistant and submit its output as your own work. Every submission must include a short statement describing which assistants were used and for what. Undeclared use is treated like any other academic integrity violation.' },
      { id: 'exams', title: 'Exams', text: 'The midterm exam takes place in week eight and covers linear models, optimization and neural networks. The final exam is cumulative and takes place during the official examination period. Both exams are closed book, but you may bring one handwritten sheet of notes, both sides, A4 size. Calculators are not needed and not allowed. Students who miss an exam for a documented medical reason take an oral make-up exam.' },
      { id: 'office', title: 'Office hours', text: 'The instructor holds office hours on Tuesdays from 14:00 to 16:00 in room B214. Teaching assistants hold additional office hours on Thursdays from 10:00 to 12:00 online. Questions about grading should be sent to the course email address rather than to individual assistants. During the examination period office hours are replaced by two review sessions announced on the course page.' },
      { id: 'project', title: 'Course project', text: 'The project is completed in teams of two or three students. Each team proposes a research question in week four, submits a progress report in week nine and presents a poster in the last week of the semester. The final report is limited to eight pages. Teams must publish their code in a repository that the course staff can access. All team members receive the same project grade unless the peer evaluation shows a clear imbalance.' },
      { id: 'collab', title: 'Collaboration', text: 'Students are encouraged to discuss problem sets in small groups. However, each student must write up their own solutions independently and list the names of their discussion partners. Copying solutions from previous years or from online sources is not allowed. Sharing your written solutions with other students is also a violation, even after the deadline.' },
    ],
    questions: [
      { id: 'c1', q: 'How many points do I lose for a late problem set?', gold: 'late submissions lose five percentage points for each day they are late', key: 'five percentage points', tag: 'boundary', closedBook: 'Late assignments typically lose 10% per day.', closedBookCorrect: false },
      { id: 'c2', q: 'Can I use ChatGPT for my homework?', gold: 'You may use generative AI assistants such as chatbots and code completion tools to explain concepts and to debug code', key: 'explain concepts', tag: 'lexical-gap', closedBook: 'Most universities ban ChatGPT completely for graded work.', closedBookCorrect: false },
      { id: 'c3', q: 'What happens to a submission that is five days late?', gold: 'Submissions that are more than four days late are not accepted and receive zero points', key: 'not accepted', tag: 'multi-hop', closedBook: 'It would lose 50% of the points.', closedBookCorrect: false },
      { id: 'c4', q: 'How much is the final exam worth?', gold: 'the final exam counts for 30 percent', key: '30 percent', tag: 'distractor', closedBook: 'Final exams are usually worth about 40% of the grade.', closedBookCorrect: false },
      { id: 'c5', q: 'When are the office hours of the instructor?', gold: 'The instructor holds office hours on Tuesdays from 14:00 to 16:00 in room B214', key: 'Tuesdays', tag: 'baseline', closedBook: 'Office hours are usually held twice a week; check the syllabus.', closedBookCorrect: false },
      { id: 'c6', q: 'Which textbook is required for the course?', gold: '', key: '', tag: 'no-answer', closedBook: 'The required textbook is "Pattern Recognition and Machine Learning" by Bishop.', closedBookCorrect: false },
      { id: 'c7', q: 'Can I bring notes to the exam?', gold: 'you may bring one handwritten sheet of notes, both sides, A4 size', key: 'handwritten sheet', tag: 'baseline', closedBook: 'Exams are closed book, so no notes are allowed.', closedBookCorrect: false },
    ],
  },
  {
    id: 'solar', title: 'Solar system notes', blurb: 'Short notes on the planets, including one outdated textbook page that contradicts a newer one.',
    docs: [
      { id: 'mercury', title: 'Mercury', text: 'Mercury is the smallest planet and the closest to the Sun. It has almost no atmosphere, so its surface swings from about 430 degrees Celsius during the day to minus 180 degrees Celsius at night. A year on Mercury lasts 88 Earth days. Mercury has no moons.' },
      { id: 'venus', title: 'Venus', text: 'Venus is wrapped in a thick carbon dioxide atmosphere that traps heat. Its average surface temperature is about 465 degrees Celsius, the highest of any planet, even though Mercury is closer to the Sun. Venus rotates very slowly and backwards: one rotation takes 243 Earth days, which is longer than its year of 225 Earth days. Venus has no moons.' },
      { id: 'earth', title: 'Earth and the Moon', text: 'Earth is the only planet known to support life. About 71 percent of its surface is covered by water. Earth has one natural satellite, the Moon, which is slowly moving away from Earth by about 3.8 centimetres per year. The Moon always shows the same face to Earth because its rotation is synchronised with its orbit.' },
      { id: 'mars', title: 'Mars', text: 'Mars is a cold desert world with a thin atmosphere. It hosts Olympus Mons, the tallest volcano in the solar system, about 22 kilometres high. Mars has two small moons, Phobos and Deimos, which may be captured asteroids. Robotic rovers have explored its surface since 1997, but no human has visited Mars.' },
      { id: 'jupiter', title: 'Jupiter', text: 'Jupiter is the largest planet; more than 1300 Earths would fit inside it. It is a gas giant made mostly of hydrogen and helium. The Great Red Spot is a storm larger than Earth that has been observed for more than 350 years. Jupiter has dozens of known moons, including Ganymede, the largest moon in the solar system.' },
      { id: 'saturn', title: 'Saturn', text: 'Saturn is famous for its bright rings made of ice and rock. It is the least dense planet: its average density is lower than that of water. Its largest moon, Titan, has a thick nitrogen atmosphere and lakes of liquid methane.' },
      { id: 'pluto2006', title: 'Pluto (IAU 2006 decision)', text: 'In 2006 the International Astronomical Union defined a planet as a body that orbits the Sun, is nearly round and has cleared its orbital neighbourhood. Pluto does not clear its neighbourhood, so it was reclassified as a dwarf planet. Since then the solar system has eight planets.' },
      { id: 'pluto1999', title: 'Old textbook page (1999)', text: 'Our solar system has nine planets. Pluto, discovered in 1930, is the ninth and most distant planet from the Sun.', note: 'outdated source' },
    ],
    questions: [
      { id: 's1', q: 'How many moons does Mars have?', gold: 'Mars has two small moons, Phobos and Deimos', key: 'two', tag: 'baseline', closedBook: 'Mars has two moons, Phobos and Deimos.', closedBookCorrect: true },
      { id: 's2', q: 'Which planet is the hottest?', gold: 'Its average surface temperature is about 465 degrees Celsius, the highest of any planet', key: '465', tag: 'lexical-gap', closedBook: 'Mercury is the hottest planet because it is closest to the Sun.', closedBookCorrect: false },
      { id: 's3', q: 'Is Pluto a planet?', gold: 'it was reclassified as a dwarf planet', key: 'dwarf planet', tag: 'conflict', closedBook: 'Pluto is the ninth planet of the solar system.', closedBookCorrect: false },
      { id: 's4', q: 'Which planet has a day longer than its year?', gold: 'one rotation takes 243 Earth days, which is longer than its year of 225 Earth days', key: '243', tag: 'multi-hop', closedBook: 'Mercury has a day longer than its year.', closedBookCorrect: false },
      { id: 's5', q: 'Who was the first astronaut to walk on Mars?', gold: '', key: '', tag: 'no-answer', closedBook: 'The first person on Mars was an American astronaut in 2031.', closedBookCorrect: false },
      { id: 's6', q: 'What is the largest moon in the solar system?', gold: 'Ganymede, the largest moon in the solar system', key: 'Ganymede', tag: 'distractor', closedBook: 'Titan is the largest moon in the solar system.', closedBookCorrect: false },
    ],
  },
  {
    id: 'glossary', title: 'Retrieval glossary', blurb: 'Definitions of the retrieval techniques used in this very lab: chunking, BM25, embeddings, vector indexes, reranking.',
    docs: [
      { id: 'chunking', title: 'Chunking', text: 'Chunking splits long documents into smaller passages before indexing. Small chunks give precise matches but may cut a sentence or an argument in half. Large chunks keep context together but dilute the similarity score with unrelated text. An overlap between consecutive chunks repeats a few words at each boundary so that information near a boundary appears complete in at least one chunk.' },
      { id: 'bm25', title: 'BM25', text: 'BM25, short for Best Matching 25, is a lexical ranking function. It scores a passage by the query terms it contains, weighting rare terms more through inverse document frequency. Term frequency saturates, controlled by the parameter k1, and long passages are penalised through the length normalisation parameter b. BM25 cannot match synonyms that share no words with the query.' },
      { id: 'embeddings', title: 'Dense embeddings', text: 'A dense embedding maps a text to a vector so that texts with similar meaning lie close together. Similarity between a query vector and a passage vector is usually measured with cosine similarity, the dot product of the two vectors divided by the product of their lengths. Dense retrieval can match paraphrases that share few exact words.' },
      { id: 'hybrid', title: 'Hybrid search', text: 'Hybrid search combines lexical and dense retrieval. A common fusion method is reciprocal rank fusion, which adds one divided by sixty plus the rank of a passage in each result list. Fusion rewards passages that rank well in both lists without having to calibrate the raw scores against each other.' },
      { id: 'index', title: 'Vector indexes', text: 'An exact index compares the query with every stored vector. Approximate indexes trade a little recall for speed. An inverted file index clusters the vectors and searches only the clusters nearest to the query, controlled by the number of probes. HNSW, short for Hierarchical Navigable Small World, builds a layered proximity graph and searches it greedily.' },
      { id: 'rerank', title: 'Reranking', text: 'A reranker reorders the first retrieved candidates with a more expensive model. A cross-encoder reads the query and a passage together and outputs a relevance score, which is usually more accurate than comparing separate embeddings. Maximal marginal relevance instead reorders results to balance relevance against redundancy with passages already selected.' },
      { id: 'grounding', title: 'Grounding and hallucination', text: 'A generated answer is grounded when every claim can be traced to the retrieved context. Hallucination is fluent text that is not supported by any source. Asking the model to cite passage numbers and to answer that it does not know when the context is insufficient reduces hallucination but does not eliminate it.' },
    ],
    questions: [
      { id: 'g1', q: 'What does BM25 stand for?', gold: 'BM25, short for Best Matching 25', key: 'Best Matching 25', tag: 'lexical-gap', closedBook: 'BM25 stands for "Best Match 2005", after the year it was published.', closedBookCorrect: false },
      { id: 'g2', q: 'Why do chunks overlap?', gold: 'so that information near a boundary appears complete in at least one chunk', key: 'boundary', tag: 'baseline', closedBook: 'Overlap makes the index faster.', closedBookCorrect: false },
      { id: 'g3', q: 'What is HNSW?', gold: 'HNSW, short for Hierarchical Navigable Small World, builds a layered proximity graph', key: 'Hierarchical Navigable Small World', tag: 'baseline', closedBook: 'HNSW stands for Hierarchical Navigable Small World, a graph-based index.', closedBookCorrect: true },
      { id: 'g4', q: 'How does a cross-encoder score relevance?', gold: 'A cross-encoder reads the query and a passage together and outputs a relevance score', key: 'together', tag: 'baseline', closedBook: 'A cross-encoder embeds the query and the passage separately and compares them.', closedBookCorrect: false },
      { id: 'g5', q: 'How can paraphrases be found when they share no words with the question?', gold: 'Dense retrieval can match paraphrases that share few exact words', key: 'Dense retrieval', tag: 'lexical-gap', closedBook: 'By using a larger keyword list.', closedBookCorrect: false },
      { id: 'g7', q: 'Who invented the transformer architecture?', gold: '', key: '', tag: 'no-answer', closedBook: 'The transformer was invented by Geoffrey Hinton in 2012.', closedBookCorrect: false },
      { id: 'g6', q: 'What is the formula of reciprocal rank fusion?', gold: 'adds one divided by sixty plus the rank of a passage in each result list', key: 'sixty', tag: 'baseline', closedBook: 'Reciprocal rank fusion averages the raw similarity scores.', closedBookCorrect: false },
    ],
  },
]
export const corpusById = (id: string) => CORPORA.find((c) => c.id === id) ?? CORPORA[0]
