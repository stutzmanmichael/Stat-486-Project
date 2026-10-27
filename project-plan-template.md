# STAT 486 Project Plan

**Due:** Tuesday, October 27, 2026, at 11:59 PM Mountain Time  
**Team:** [Names]  
**Repository:** [GitHub URL]  
**Date updated:** [Date]

Use these prompts to plan your investigation. Keep the answers focused; revise the plan when evidence or instructor feedback changes your direction. The Oct. 27 checkpoint still requires a new, dated project-plan entry in the root `README.md` of your team's GitHub repository. Link this document there if you use it; this document does not replace that entry.

## Project question and purpose

- What specific question do you want to investigate?
- Why does this question interest your team, and what might someone learn from answering it?
- What claim or expectation could your evidence support, challenge, or leave uncertain?
- What does one observation represent in the data you expect to use?

## Chosen route

Choose one route and explain why it fits your question and available time.

- [ ] **A. Develop something new** in an ML method or system
- [ ] **B. Kaggle competition** with two meaningfully different pipelines
- [ ] **C. Build a dataset** and use it in a reproducible ML pipeline

**Why this route:** [How does the route support the question and the team's intended contribution?]

## Data source and access

- What data will you use? Give the dataset, competition, API, or source name and a direct link if available.
- What does each row or record represent, and which fields are relevant to the question?
- How will the team obtain the data? Note access requirements, collection dates or versions, and any expected size or compute needs.
- What license, competition rules, privacy, ethical, or other use limits apply? Could any data be restricted or identifying?
- What data quality issues might affect the investigation (for example, missingness, duplicates, labels, coverage, or changes over time)?

## Route-specific plan

Complete the subsection for your chosen route. For a different route, remove the unused subsections.

### Route A: New method or system component

- What simple reference method will you implement first?
- What is the one team-designed change you want to test, and what purpose does it serve?
- How will you isolate the effect of that change under comparable evaluation conditions?
- What small or synthetic case can help check that the component works as intended?
- What result or setting might show that the change does not help?

### Route B: Kaggle competition

- Which competition are you considering, and has the instructor approved it?
- Do its rules allow the team's planned data and code use? Are data access and compute feasible by Dec. 10?
- What will pipeline 1 do, from competition data through preparation, local validation, training, and inference?
- What substantive design decision will differ in pipeline 2 (not only a seed or minor hyperparameter)? Why investigate that change?
- What local validation approach and metric will you use? How will you guard against leakage?
- How will you record each pipeline's run command, configuration, commit, valid Kaggle submission, local result, and Kaggle result?

### Route C: New dataset and ML pipeline

- What useful dataset will you create, and what question will the ML analysis investigate?
- What source or sources will you combine or collect? Are collection and redistribution permitted?
- What does one observation represent, and what will the dataset schema contain?
- How will you collect, integrate, clean, or label the data? Which scripts will reproduce these steps?
- How will you record source locations, access or license terms, collection dates or versions, and transformations?
- What quality checks will you run (such as join coverage, duplicates, missingness, label consistency, or source changes)?
- Can you preserve a permitted snapshot or small fixture? If raw data cannot be shared, how can an authorized reader rebuild or verify the cleaned data?
- What reference ML method will use the cleaned dataset, and what evidence will show both dataset fitness and ML usefulness?

## Initial reference and proposed contribution

- What is the simplest credible baseline, reference method, or comparison for your question?
- What will your team design, build, collect, or investigate beyond that reference?
- Which choices, code, data, papers, notebooks, packages, or pretrained models will come from others, and how will you credit them?

## Evaluation and evidence

- What evaluation design fits the question and how the data were collected (for example, a split, cross-validation, control, stability check, or data-quality diagnostic)?
- What metric or diagnostic will you use, and why does it match the question?
- How will you compare your work with a baseline or other reference? What would count as evidence against your initial expectation?
- What informative error, failure, uncertainty, tradeoff, or limitation will you inspect?
- What figures, tables, examples, or other outputs could make the results understandable?

## Risks and backup scope

- What could prevent the plan from working (for example, access, collection, compute, time, data quality, or competition constraints)?
- What smaller version would still let you answer a useful question if a risk occurs?
- What evidence or feedback would lead you to pivot, and how would you document the reason?

## Reproducibility and team contributions

- Where will code, configuration, data instructions, and results live in the repository?
- What is the first small runnable result you aim to produce, and what will a reader need to reproduce it?
- How will team members contribute substantively and review one another's work?
- What support, approval, or early instructor feedback would help you proceed?

## Sources

List initial data sources, papers, competition pages, public notebooks, software, and other materials relevant to the plan. Include links and access or license details when known.

- [Source or material, link, and relevant note]
