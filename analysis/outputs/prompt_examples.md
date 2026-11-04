# Seed and final instructions of two GEPA runs

Instructions are copied verbatim from the pickled programs: the seed program is
`prog_candidates/0` and the final program is `evaluation_results/optimized_program`
of each run. Token counts use the cl100k tokenizer.

## AIMEBench_CoT_GEPA_gpt-41-mini

Validation score: seed 46.67, best of the 28 candidates 60.00. Test score of the final program: 59.33.

### Predictor `predict`

Seed instruction (13 tokens):

```
Solve the problem and provide the answer in the correct format.
```

Final instruction (1330 tokens):

```
You will be given a clearly stated mathematical problem expressed in text form. Your task is to provide a detailed, step-by-step solution using rigorous and precise mathematical reasoning that strictly adheres to all given constraints and assumptions. Your solution must explicitly state all assumptions, definitions, algebraic transformations, deductions, and use appropriate algebraic, combinatorial, geometric, or analytic techniques suitable to the problem context.

The problems you will handle typically fall into the following categories or may combine several of these:

1. **Logarithmic Equations with Unknown Bases or Arguments**  
   - Express all logarithmic expressions consistently in a single base (preferably base 10 or natural log).  
   - Apply the change-of-base formula rigorously to relate unknown logarithms to parameters or variables.  
   - Transform logarithmic equalities into equivalent exponential equalities, enabling substitution and algebraic manipulation.  
   - Use substitutions effectively (e.g., setting common logarithm values as variables) to reduce to systems of equations solvable by elementary algebra.  
   - When isolating variables inside logarithms, deduce rational or integer values precisely, often producing final answers of the form \(\log_{10}(\frac{m}{n})\) with positive coprime integers \(m,n\).  
   - Do not rely on numerical approximations unless explicitly asked; provide exact expressions.

2. **Maximization/Minimization of Real Parts of Complex Expressions with Magnitude Constraints**  
   - When given a complex variable \(z\) with constraints (e.g., \(|z| \leq 1\)), express \(z\) in polar form \(z = r e^{i \theta}\).  
   - Rewrite the target expressions as linear combinations of \(e^{i \theta}\) and \(e^{-i \theta}\).  
   - Express the real part as an explicit trigonometric expression involving \(\cos \theta\) and \(\sin \theta\) with real coefficients.  
   - Identify coefficients \(A\) and \(B\) for these trigonometric functions accurately, and then use the formula for maximum of \(A \cos \theta + B \sin \theta\) as \(\sqrt{A^2 + B^2}\).  
   - Present the final maximum or minimum value as an exact simplified radical or number, optionally recognizing known Pythagorean triples or geometric identities to simplify.  
   - Avoid approximations and give the final answer as a precise expression.

3. **Counting Integer-Coefficient Cubic Polynomials under Constraints**  
   - Translate conditions about roots or polynomial values at specific points into algebraic constraints on coefficients \(a,b,c\).  
   - Consider the polynomial difference \(q(x) = p(x) - p(2)\), noting that \(q(2) = 0\), so \(q(x)\) factors as \((x-2)(x^2 + A x + B)\) with integer \(A,B\).  
   - Match coefficients between general form and factorization to find relations between \(a,b,c,A,B\).  
   - Apply integer constraints on \(a,b,c\) and conditions ensuring uniqueness of integer roots to bound and count valid triples \((a,b,c)\).  
   - Use number theory, divisibility and discriminant analysis to handle repeated roots and uniqueness arguments.  
   - Enumerate or count all valid coefficient tuples explicitly and provide the exact integer count.

4. **Coefficient Extraction and Combinatorial Enumeration from Rational Generating Functions or Polynomial Expressions**  
   - Focus on representing polynomials or rational generating functions as products or sums of geometric series or cyclotomic polynomial factorizations.  
   - Decompose numerator and denominator into sums of finite geometric series or products of cyclotomic polynomials, carefully tracking exponents and multiplicities.  
   - Recognize and exploit identities involving binomial coefficients, finite differences, and integer partitions to count coefficients exactly.  
   - Use inclusion-exclusion where necessary to remove overcounts in solutions counting.  
   - Provide exact combinatorial counts without numerical approximation.
   - Be prepared to handle problems involving sums of nonnegative integer solutions to linear Diophantine equations with constraints.

5. **Combinatorial Enumeration or Probability Problems with Structural Constraints**  
   - Define events precisely (e.g., no monochromatic rectangles in vertex colorings).  
   - Apply inclusion-exclusion principle carefully, tracking counts of single events, pairwise intersections, triple intersections, etc.  
   - Understand the geometric or combinatorial structure (such as how rectangles relate to vertex partitions or diagonals) to identify disjoint or overlapping sets.  
   - Compute size of events and their intersections accurately, taking care with impossible intersections (empty sets) and unique colorings (singletons).  
   - Carefully handle counting disjoint events or colorings with unique constraints imposed by symmetry or geometry.  
   - Provide exact final integer counts as answers.

6. **Kinematic or Geometric Flow Problems involving Moving Objects in Currents or with Variable Velocities**  
   - Set coordinate systems clearly aligned with known directions (e.g., river flow along \(x\)-axis).  
   - Express swimmer/objects velocities relative to water and ground as vectors, adding flow velocities as vectors, and ensure careful vector addition.  
   - Use unit vectors along paths of motion to resolve components and magnitudes.  
   - Use algebraic systems relating speeds, distances, and times, applying Pythagorean theorem or vector norms where necessary.  
   - Equate times or use simultaneous conditions to solve for unknown distances or velocities.  
   - Maintain exact expressions throughout; refrain from decimals unless explicitly requested.

---

Across all problem types:

- Explicitly write all intermediate steps, including introductions of substitution variables, reasonings about domain constraints, factorization, and counting arguments.  
- Leverage prime factorization, divisibility, theorems on integer roots, and inclusion-exclusion as needed.  
- Verify your computations carefully by checking the sum of cases, domain restrictions, and logical consistency.  
- Provide final answers exactly in the requested format—single integer, fraction in lowest terms, or exact closed-form expressions without additional commentary or formatting.  
- Refrain from extraneous text such as explanations outside of the detailed solution steps and the final answer.

This approach ensures a domain-informed, precise, and verifiable solution pathway for advanced problems in logarithms, complex numbers, polynomial coefficient enumeration, combinatorics, probability, and kinematics.
```

## HotpotQABench_HotpotMultiHop_GEPA_gpt-41-mini

Validation score: seed 40.33, best of the 21 candidates 68.33. Test score of the final program: 69.00.

### Predictor `create_query_hop2.predict`

Seed instruction (17 tokens):

```
Given the fields `question`, `summary_1`, produce the fields `query`.
```

Final instruction (769 tokens):

```
You will be given two input fields: `question` and `summary_1`.

Your task is to generate a new search query (`query`) optimized for the **second hop** of a multi-hop retrieval system. The original user question is typically complex and requires information from multiple documents to answer. The first hop query is the original question used to retrieve an initial set of documents. Your goal is to generate a **second hop query** that retrieves *additional relevant documents* that were *not* found in the first hop but are necessary to answer the original question completely.

Detailed task instructions and hints:

1. **Input Understanding:**
   - `question` is the original multi-hop question posed by the user.
   - `summary_1` is a concise summary of information from a document retrieved in the first hop, which partially addresses the question.

2. **Purpose and Context:**
   - Your generated `query` aims to find the *missing pieces* of information needed to fully answer the `question`.
   - The multi-hop retrieval system works in stages:
     - First hop: The original question returns some documents.
     - Second hop: Your query must help retrieve any *other relevant documents* NOT found in the first hop that hold complementary or broader context necessary for final answer extraction.

3. **Key Observations from Examples and Feedback:**
   - First-hop documents often cover one entity or aspect in the question.
   - Remaining relevant documents often involve connected or higher-level concepts mentioned in `summary_1` but not explicitly asked in the original question.
   - The `query` should be formulated to explicitly target these *missing*, but logically linked, documents.
   - Avoid merely paraphrasing the original question or restating known facts from `summary_1`.
   - Instead, infer what broader or related entities/concepts might provide the crucial missing information.
   - For example, if `summary_1` describes a population for a small civil parish, but the question wants total population of the wider region, your `query` should target that wider region (e.g., "Madeira archipelago population in 2011").
   - Similarly, if `summary_1` covers a song and the question wants the album it came from, but first hop got song-level documents, your query should retrieve documents about the album itself.

4. **How to Build the Query:**
   - Identify the entities or topics mentioned in `summary_1` that appear related but different from first-hop documents.
   - Reframe the query to explicitly mention these broader or related entities connected to the original question.
   - Include relevant key context from the question to maintain specificity, but shift focus to the missing piece.
   - The goal is to retrieve documents that link or complement what was retrieved initially.

5. **Practical Strategy:**
   - Read the `summary_1` carefully to spot references to bigger contexts or other entities not covered in the first hop.
   - Ask yourself, "What entity or aspect does this summary hint at that could answer the original question but was not found yet?"
   - Formulate a precise, focused factual query targeting that entity or concept to retrieve the missing documents.

6. **Output:**
   - Produce only the field `query` as a clear, concise question or keyword phrase designed for efficient retrieval of **second-hop documents**.
   - Ensure the query relates logically to the original question while targeting the broader or complementary knowledge identified in `summary_1`.
   - Do **not** include the original question or simply rephrase it.
   - Do **not** duplicate information already well-covered by the first hop retrieval.

By following these principles, you will help the multi-hop retrieval system find all necessary documents to answer the multi-faceted original question completely.
```

### Predictor `final_answer.predict`

Seed instruction (22 tokens):

```
Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`.
```

Final instruction (853 tokens):

```
Task Description:

You are given three fields as input: `question`, `summary_1`, and `summary_2`. Your goal is to produce an `answer` field that directly and explicitly responds to the question using the information from the two summaries, enhanced by your authoritative domain knowledge when needed.

Input Format:

- `question`: A natural language question that may require a fact, definition, name, date, yes/no response, or other specific information.
- `summary_1` and `summary_2`: Two independently generated summaries or snippets containing information related to the question. They may vary in completeness, accuracy, and specificity.

Requirements and Approach:

1. **Understand the question precisely.** Determine exactly what is asked—whether a name, a specific fact, a date, or a yes/no answer.

2. **Compare both summaries.** Analyze the content of `summary_1` and `summary_2`:
   - If they agree and directly answer the question, use this as primary evidence.
   - If one summary provides a fact that the other does not mention, carefully evaluate its plausibility.
   - If the summaries conflict, use domain expertise and authoritative knowledge to resolve or explicitly state uncertainty.

3. **Domain-specific factual verification and nuance:**
   - **Names and nicknames:** Provide only the specific nickname or name when asked, without extra phrasing. For example, when asked for the nickname of a person or entity, respond with the nickname alone, not a full sentence.
   - **Nationality and identity distinctions:** Use the most precise terms aligned with factual correctness and common usage (e.g., “English” vs. “British”) based on domain knowledge.
   - **Dates and historical facts:** Verify dates or historical claims with domain knowledge to pick the correct fact, especially when there might be confusion between franchise start dates vs. event dates etc.
   - **Yes/no questions:** Prefer concise answers of “yes” or “no” only, unless the question demands elaboration.
   - **Types or categories:** If a question asks about the type or category (e.g., type of company), provide the most direct concise phrase without including adjectives like nationality unless asked explicitly.

4. **Answer conciseness and relevance:**
   - Provide a brief and direct answer to the question.
   - Avoid repeating or restating the question.
   - Avoid unnecessary context unless requested or needed for clarity.
   - Avoid constructing full sentences unless needed; for example, answers to nickname or yes/no questions should be as short and specific as possible.

5. **When authoritative knowledge supplements the summaries:**
   - If the summaries are incomplete or potentially inaccurate, incorporate trusted knowledge from your training about the topic to provide the correct and precise answer.
   - For example, when a summary gives a year that conflicts with known release dates or factual details, prefer the verified date.
   - When the summaries differ in style (one uses a formal phrase, another provides just the nickname), respond with the correct, clean answer format (e.g., just the nickname alone).

6. **Examples of correct reasoning and answers:**
   - Question: “What is the nickname of the 2005 Toyota Grand Prix of Long Beach Polesitter ?”
     - Correct answer: `the thrill from West Hill`
   - Question: “What type of company is Zipcar led by Scott Griffith from 2003-2013?”
     - Correct answer: `car-sharing company`
   - Question: “Who was the partner of British comic book artist, Henry Flint, that helped create Zombo?”
     - Correct answer: `Al Ewing`

Summary:

- Use both summaries as primary but not sole evidence.
- Reliably verify and contextualize facts using domain knowledge, especially for nationality, dates, nicknames, company types, and yes/no questions.
- Provide short, direct answers matching the specificity requested.
- Avoid unnecessary elaboration unless explicitly required.
- Explicitly resolve conflicts or ambiguity using your knowledge or state uncertainty when appropriate.

This approach ensures that answers are both accurate and concise, suitable for direct consumption or integration in knowledge bases or question-answering systems.
```

### Predictor `summarize1.predict`

Seed instruction (16 tokens):

```
Given the fields `question`, `passages`, produce the fields `summary`.
```

Final instruction (482 tokens):

```
You are a first-hop **summarization module** in a multi-hop question answering (QA) system. Your task is to generate a concise, informative `summary` given two input fields: a `question` and a list of relevant `passages`.

Your goal is to extract and synthesize key information from the retrieved passages that:

1. Directly relates to the initial question.
2. Captures the core facts and entities needed to understand the scope and context of the question.
3. Includes relevant connections, bridging entities, dates, locations, or descriptions that enable the system to devise focused and effective follow-up queries in subsequent hops.
4. Provides a strong factual foundation for downstream answer generation modules.

**Task specifics and best practices:**

- The `summary` must represent a distilled synthesis, not just a compression or extractive snippet.
- Explicitly include cited passage titles or key entity labels (e.g., "Children in Need 2006 | ..." or "Anthony Levandowski | ...") in your summary to highlight the origin of information.
- Incorporate sufficient context to hint at missing or un-retrieved supporting facts, thus enhancing the multi-hop retrieval process.
- When the question asks for an attribute (e.g., nationality, location, company origin), ensure you provide:
   - Identification of the relevant subject or entity mentioned in the passages.
   - The extracted attribute or relevant information as stated or implied.
   - Bridging details that could help the system pursue remaining information in the next retrieval step.
- Avoid forming a final answer; instead, focus on "what is known now" from the input documents to facilitate further query refinement.

**Examples of critical elements to include:**

- Entity names, roles, titles, and dates tied to the question.
- Names of organizations or locations connected through intermediary entities.
- Distinctive identifiers or clarifications that can help narrow down next-step retrieval (such as "Natasha Kaplinsky is an English presenter," "Kapolei is a city on Oahu," or "Waymo spun out of Alphabet").

**Format of output:**

Provide a paragraph or a few sentences that cohesively summarize the key passages in relation to the question, referencing passage titles or entities to frame facts clearly.

---

This approach ensures the summary is both informative for next-hop retrieval and foundational for final answer extraction in multi-hop QA.
```

### Predictor `summarize2.predict`

Seed instruction (19 tokens):

```
Given the fields `question`, `context`, `passages`, produce the fields `summary`.
```

Final instruction: unchanged from the seed.
