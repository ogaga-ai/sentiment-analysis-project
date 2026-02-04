# Project Blog Notes

> Use this document to capture your learning journey, decisions, and insights for writing your first article/blog.

---

## Article Working Title

**"Building a Sentiment Analysis Tool from Scratch: A Beginner's Journey into Responsible AI"**

*(Feel free to change this as you go)*

---

## Key Themes to Cover

- [ ] Why I chose this project
- [ ] The importance of understanding WHO uses the technology
- [ ] Learning by doing vs. reading papers first
- [ ] Data cleaning as a critical skill
- [ ] Model explainability (why it matters)
- [ ] Bias in ML models (what I discovered)
- [ ] Lessons learned and mistakes made

---

## Phase 1: Project Setup & Data Collection

### Date Started: 2026-01-31

### What I Did
- Created organized project folder structure (data/, notebooks/, src/, models/, app/)
- Set up README.md with project overview, target users, features, and ethics section
- Created requirements.txt with all Python dependencies (pandas, scikit-learn, transformers, etc.)
- Created .gitignore to exclude large data files and sensitive info from version control
- Created BLOG_NOTES.md to document the learning journey
- Researched existing sentiment analysis projects on GitHub to understand the landscape
- Identified gaps in existing work (lack of explainability, no bias analysis, poor documentation)

### Why I Did It
- **Organized structure** = Mimics real-world ML projects, easier to maintain and share on GitHub
- **README from day one** = Portfolio-ready, forces clarity on project goals before coding
- **Multiple datasets (Amazon, Yelp, IMDB)** = Model generalizes across domains, not just one type of review
- **Ethics section upfront** = Responsible AI thinking from the start, not an afterthought
- **Researched prior work** = Understand what exists, identify how to differentiate and improve

### What I Learned
- Before writing code, understanding WHO uses the technology and its potential impact is crucial
- Many existing projects lack explainability, bias analysis, and proper documentation — this is an opportunity to stand out
- Reading all research papers before starting isn't necessary — learn by doing, read as needed
- A well-structured project is easier to explain, maintain, and showcase in a portfolio

### Challenges Faced
- Feeling overwhelmed by the volume of research papers and existing projects
- Uncertainty about skill level — "Do I know enough to start this?"
- Realizing that learning ML is iterative, not linear

### Aha Moment
> "It's not just about building the model — we have to ask: Who uses this? What's the impact? Who could be harmed?"

### Key Decisions Made
| Decision | Why |
|----------|-----|
| Used 3 datasets instead of 1 | Better generalization across domains (e-commerce, local business, entertainment) |
| Included ethics section in README | Responsible AI from the start, not bolted on later |
| Learn-by-doing approach | Reading papers first leads to paralysis; build first, read as needed |
| Document everything for a blog | Forces reflection, creates portfolio content, helps others learn |
| User does the coding | Hands-on learning is more effective than watching someone else code |

---

## Phase 2: Data Preprocessing & EDA

### Date Started: 2026-02-03

### What I Did
- Installed required packages (pandas, jupyter, matplotlib already installed; added seaborn)
- Learned Python fundamentals needed for data work: f-strings, raw strings, absolute vs relative paths
- Created a **practice notebook** (separate from the project) to experiment with code
- Loaded the IMDB dataset using `pd.read_csv()` with an absolute path
- Explored the data using `.head()`, `.shape`, `.columns`, `.info()`
- Checked sentiment distribution using `.value_counts()` and percentages
- Examined actual positive and negative review examples
- Measured review lengths using `.apply(len)` and `.describe()`

### Why I Did It
- **Practice notebook first** = Build confidence before touching the real project
- **Understand each line** = No blindly copying code; every command must make sense
- **Explore before cleaning** = Need to see what the raw data looks like before deciding how to clean it
- **Check balance** = Imbalanced datasets (e.g., 90% positive, 10% negative) cause biased models

### What I Learned
- `pd.read_csv()` loads CSV files into DataFrames (like Excel in Python)
- `.head()` shows first 5 rows, `.shape` gives (rows, columns), `.info()` gives summary
- `.value_counts()` counts unique values; add `normalize=True` for proportions
- `.apply(len)` runs a function on every row — useful for creating new columns
- `.describe()` gives min, max, mean, median and percentile statistics
- Windows paths need `r""` (raw strings) or forward slashes to avoid Python escape character conflicts
- Absolute paths = full address from root; relative paths = directions from current location
- `python -m pip install` is safer than `pip install` when multiple Python versions exist
- Regular expressions (`re.sub()`) for pattern-based text cleaning — removing HTML, special characters
- Raw strings (`r''`) needed for regex patterns too, not just file paths — same `\s` escape issue
- Different datasets come in different formats (CSV, compressed .bz2, JSON Lines) — each needs a different loading method
- `lines=True` in `pd.read_json()` for JSON files where each line is a separate object
- `.map()` with a dictionary to convert values (stars → sentiment labels)
- `.dropna(subset=[...])` to remove specific missing values without losing other data
- `pd.concat()` to stack multiple DataFrames together, `ignore_index=True` to reset row numbers
- `plt.subplots(rows, cols)` to create multiple charts side by side
- 58% of IMDB reviews contained HTML tags — cleaning is essential, not optional
- Checking the full dataset matters — `.head()` alone can miss issues affecting thousands of rows

### Challenges Faced
- seaborn initially failed to install — resolved by upgrading pip first
- Didn't understand articles about pandas/EDA — switched to hands-on learning with real data
- Needed explanation of Python concepts (f-strings, raw strings, .apply()) before the code made sense
- SyntaxWarning for regex escape sequences — fixed by adding `r` prefix to regex strings
- Different dataset formats required different loading approaches (CSV vs bz2 vs JSON)
- Yelp star 3 reviews had to be dropped — neutral reviews don't fit a binary classification task

### Aha Moments
> "I don't need to understand everything from articles first. Loading real data and asking questions about each line of code teaches me faster."

> "Looking at just 5 rows tells you almost nothing about 50,000 reviews. Always verify with the full dataset."

> "58% of the data had HTML tags in it. If I hadn't checked, the model would have learned garbage."

### Key Insights from the Data
- **IMDB:** 50,000 reviews, perfectly balanced (50/50 positive/negative)
- **Amazon:** 100,000 reviews loaded from compressed format
- **Yelp:** Started with 100,000, dropped to 88,638 after removing neutral (star 3) reviews — about 11% were neutral
- **Combined total:** 238,638 cleaned reviews from 3 domains
- **Dataset sizes are unequal** — Amazon has 2x more reviews than IMDB, which could bias the model toward Amazon-style language
- **Different review styles:** Amazon = product-focused, IMDB = opinion-heavy, Yelp = service-focused
- **HTML tags found in 58% of IMDB reviews** — major cleaning needed
- Three different labeling methods all converted to the same positive/negative format



---

## Phase 3: Baseline Model

### Date Started:

### What I Did


### Why I Did It


### What I Learned


### Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | |
| Precision | |
| Recall | |
| F1 Score | |

### What Surprised Me



---

## Phase 4: Model Comparison

### Date Started:

### Models Tested
| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | | |
| Naive Bayes | | |
| SVM | | |
| Random Forest | | |

### What I Learned


### Best Model & Why



---

## Phase 5: Deep Learning (DistilBERT)

### Date Started:

### What I Did


### Why Transformers?


### Performance vs Baseline
| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Best Traditional Model | | |
| DistilBERT | | |

### What I Learned



---

## Phase 6: Explainability (SHAP/LIME)

### Date Started:

### What I Did


### Why Explainability Matters


### Interesting Findings
*(Which words drove positive/negative predictions?)*



---

## Phase 7: Bias & Error Analysis

### Date Started:

### What I Tested


### Bias Findings
| Test | Result |
|------|--------|
| Sarcasm detection | |
| Different dialects | |
| Short vs long reviews | |
| Different domains | |

### Who Gets Harmed?
*(Reflect on who might be negatively impacted by model errors)*



### Recommendations for Responsible Use



---

## Phase 8: Deployment (Streamlit)

### Date Started:

### What I Built


### User Experience Decisions


### Demo Link
*(Add link when deployed)*



---

## Phase 9: Final Documentation

### Date Completed:

### GitHub Repo Link


### What I Would Do Differently



---

## Quotable Moments

> *(Capture memorable insights, frustrations, or breakthroughs here for your blog)*

1. "It's not just about building the model — we have to ask: Who uses this? What's the impact?"

2. "Looking at these research papers, it just tells me I don't know enough to begin this project." — The classic beginner's doubt before realizing learning by doing is the way.

3. "Sentiment analysis isn't neutral. The same technology that helps businesses can also enable surveillance."

4. "I don't understand a thing from the articles." — Honesty about not understanding led to a better approach: hands-on coding with real data instead of reading tutorials.

5. "I don't need to read all the papers first. Load the data, write the code, ask questions as I go."

---

## Resources That Helped Me Most

| Resource | Why It Helped |
|----------|---------------|
| [Hugging Face: Getting Started with Sentiment Analysis](https://huggingface.co/blog/sentiment-analysis-python) | Beginner-friendly, modern approach, just 5 lines of code to start |
| [PyCharm Blog: Intro to Sentiment Analysis](https://blog.jetbrains.com/pycharm/2024/12/introduction-to-sentiment-analysis-in-python/) | Python basics, VADER, practical examples |
| [Ethical Considerations in AI-Powered Sentiment Analysis](https://cogentixresearch.com/ethical-considerations-in-ai-powered-sentiment-analysis/) | Opened my eyes to privacy, bias, and surveillance concerns |
| Existing GitHub projects (vinaykanigicherla, ezgigm, etc.) | Showed me what's been done and where the gaps are |

---

## Blog Outline Draft

### Introduction
- Hook: Why sentiment analysis matters
- My background and why I chose this project

### The Problem
- What is sentiment analysis?
- Who uses it and why it matters

### The Ethical Question
- Before building: Who is impacted?
- Responsible AI considerations

### The Build Process
- Phase by phase walkthrough
- Key challenges and solutions

### Results
- Model performance
- What worked, what didn't

### Lessons Learned
- Technical skills gained
- Soft skills (critical thinking, documentation)

### Conclusion
- Call to action for readers
- What's next for me

---

## Where to Publish

| Platform | Audience | Notes |
|----------|----------|-------|
| [Towards Data Science](https://towardsdatascience.com) | Data scientists, ML engineers | High visibility, editor review required |
| [Towards AI](https://pub.towardsai.net) | Similar audience | Easier to get accepted |
| [Analytics Vidhya](https://www.analyticsvidhya.com/blog/) | Beginners & practitioners | Beginner-friendly |
| [DEV.to](https://dev.to) | General developers | No review process, publish instantly |
| Personal blog (GitHub Pages) | Anyone | Full control, no gatekeepers |

---

## Raw Notes

*(Dump any random thoughts, links, or ideas here)*


