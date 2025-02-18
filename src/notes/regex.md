# Regex Cheat Sheet

Resource Link: https://docs.python.org/3/library/re.html

| **Pattern** | **Meaning** | **Example** |
|------------|------------|-------------|
| `.` | Any character (except newline) | `"c.t"` → matches `"cat"`, `"cut"`, `"cot"` |
| `^` | Start of a string | `"^Hello"` → matches `"Hello, world"` but not `"world, Hello"` |
| `$` | End of a string | `"end$"` → matches `"The end"` but not `"end of story"` |
| `*` | Zero or more occurrences | `"go*"` → matches `"g"`, `"go"`, `"goo"`, `"gooo"` |
| `+` | One or more occurrences | `"go+"` → matches `"go"`, `"goo"`, `"gooo"`, but not `"g"` |
| `?` | Zero or one occurrence | `"colou?r"` → matches `"color"` and `"colour"` |
| `{n,m}` | Between n and m occurrences | `\d{2,4}` → matches 2 to 4 digit numbers |
| `\d` | Any digit (`0-9`) | `"User \d+"` → matches `"User 123"`, `"User 42"` |
| `\w` | Any word character (alphanumeric + `_`) | `"\w+"` → matches `"hello"`, `"world123"` |
| `\s` | Any whitespace (space, tab, newline) | `"hello\sworld"` → matches `"hello world"` |
| `\b` | Word boundary | `"\bAI\b"` → matches `" AI "` but not `"AIML"` |

<br>
<br>

---
---
# Best Libraries for Regex in Python

| **Use Case**                      | **Best Library**  |
|------------------------------------|------------------|
| General Regex in Python           | `re`, `regex`   |
| Interactive Regex Testing         | `pythex`        |
| Big Data & Log Analysis           | `pygrep`, `loguru` |
| Large-Scale Text Matching         | `flashtext`     |
| NLP Entity Extraction             | `spaCy`         |
| DataFrame Regex Operations        | `pandas`        |

<br>
<br>

---
---
# Best Tools for Distributed Data & Regex Processing

| **Use Case**                      | **Best Tool**                             |
|------------------------------------|------------------------------------------|
| **Distributed Regex Processing**   | `PySpark`, `Spark NLP`                   |
| **Real-Time Text Analysis**        | `Apache Kafka`, `Flink`                   |
| **Fast Text Search & Indexing**    | `Elasticsearch`, `Solr`                   |
| **Cloud-Based Big Data Storage**   | `HDFS`, `S3`, `Google Cloud Storage`      |

<br>
<br>

---
---

# Free Large-Scale Datasets for Regex & NLP Practice

## **1. Open Government & Web Scraping Datasets**
- 🔗 [Common Crawl](https://commoncrawl.org/) – Petabytes of web crawl data (HTML, metadata, text).
- 🔗 [Enron Email Dataset](https://www.cs.cmu.edu/~enron/) – 500,000+ real email conversations.
- 🔗 [Amazon Open Data Registry](https://registry.opendata.aws/) – Large public datasets on AWS (logs, text, NLP).
- 🔗 [U.S. Government Open Data](https://www.data.gov/) – Public datasets from multiple industries.
- 🔗 [Wikipedia Dumps](https://dumps.wikimedia.org/) – Entire Wikipedia text corpus (~90GB compressed).

## **2. Chatbot & NLP Text Data**
- 🔗 [Twitter Customer Support Dataset](https://www.kaggle.com/thoughtvector/customer-support-on-twitter) – 3M+ chatbot-like tweets.
- 🔗 [Cornell Movie Dialogs](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) – 220k movie dialogue lines for chatbot training.
- 🔗 [Reddit Comments Dataset](https://files.pushshift.io/reddit/comments/) – Large-scale comments (compressed in JSON).
- 🔗 [OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles-v2018.php) – Multi-language conversation dataset (subtitles).

## **3. Log Files & Error Detection**
- 🔗 [NASA Apache Web Server Logs](https://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html) – Gigabytes of real server logs.
- 🔗 [Linux Syslog Datasets](https://www.secrepo.com/) – System logs, security logs for regex testing.
- 🔗 [Microsoft Malware Threat Logs](https://www.kaggle.com/c/malware-classification/data) – Security-related log files.

## **4. Large-Scale Financial & Business Datasets**
- 🔗 [SEC Edgar Filings (10-K Reports)](https://www.sec.gov/dera/data) – Huge financial reports (ideal for regex-based entity extraction).
- 🔗 [IMF Economic Data](https://www.imf.org/en/Data) – Structured CSVs for regex parsing.
- 🔗 [Kaggle’s Free Business Datasets](https://www.kaggle.com/datasets?search=business) – Business transaction logs, text-based datasets.

## **5. Cloud-Based Big Data (S3, HDFS)**
- 🔗 [Google Cloud Public Datasets](https://cloud.google.com/public-datasets/) – NLP, text, and structured data for testing.
- 🔗 [AWS Open Data Registry](https://registry.opendata.aws/) – Free access to TBs of structured & unstructured data.

<br>
<br>

---
---

# Structured Learning Guide: Strengthening Regex & Distributed Data Frameworks

## **Phase 1: Mastering Regex in Python**
- Learn **basic regex patterns** (`.*, \d+, \bword\b, \w+`).
- Practice with **`re` and `regex` libraries** (`re.search()`, `re.findall()`, `re.sub()`).
- Work with **regex-based text extraction** (emails, URLs, phone numbers).
- Apply regex for **log file parsing** (error codes, timestamps, message filtering).
- Explore **advanced regex concepts** (lookaheads, lookbehinds, nested groups).

## **Phase 2: Applying Regex in NLP & Chatbot Analytics**
- Use **regex for text preprocessing** (stopwords removal, tokenization).
- Implement **intent detection using regex rules** (classifying chatbot queries).
- Extract **named entities with regex** (dates, invoice numbers, product names).
- Perform **regex-based text classification** (sentiment analysis from messages).
- Work with **Pandas** for regex filtering (`df['column'].str.extract()`).

## **Phase 3: Big Data Processing with Regex & PySpark**
- Learn **PySpark DataFrames & RDDs** for distributed regex operations.
- Apply **`regexp_extract` and `regexp_replace` in Spark SQL**.
- Use regex to **clean and normalize large-scale chatbot logs**.
- Perform **real-time regex filtering** on big data stored in HDFS.

## **Phase 4: Real-Time Regex with Apache Kafka**
- Set up **Kafka producers & consumers** to stream chatbot messages.
- Apply **real-time regex filtering** (detect escalation keywords).
- Use regex for **log analysis in Kafka Streams**.

## **Phase 5: High-Speed Regex Searching with Elasticsearch**
- Learn **Elasticsearch indexing & querying**.
- Perform **regex-based full-text search** on chatbot logs.
- Optimize regex performance using **Elasticsearch analyzers**.

## **Phase 6: End-to-End Project Integration**
- Build a **real-time chatbot analytics pipeline** (Kafka → PySpark → Elasticsearch).
- Monitor chatbot **misunderstood queries using regex rules**.
- Optimize **customer sentiment detection with regex-based NLP processing**.
- Automate **regex-based alerts for error messages in chatbot logs**.

<br>
<br>

---
---
# Special Characters in Regex
| Special Character | Meaning                                       | Example Match              |
|-------------------|-----------------------------------------------|----------------------------|
| `.`              | Matches any character except a newline        | `"c.t"` → `cat`, `cut`    |
| `^`              | Matches start of a string                     | `"^Hello"` → `"Hello world"` |
| `$`              | Matches end of a string                       | `"world$"` → `"Hello world"` |
| `\d`            | Matches a digit (0-9)                         | `"\d+"` → `"123"`         |
| `\D`            | Matches any non-digit                         | `"\D+"` → `"abc"`         |
| `\w`            | Matches a word character (letters, digits, `_`) | `"\w+"` → `"Python3"`     |
| `\W`            | Matches any non-word character                | `"\W+"` → `" @!"`         |
| `\s`            | Matches a space, tab, or newline              | `"\s+"` → `" "`           |
| `\S`            | Matches any non-space character               | `"\S+"` → `"word"`        |
| `\b`            | Matches a word boundary                       | `"\bword\b"` → `"word"` (not `"sword"`) |

<br>
<br>

---
---
# Quantifiers (Repetition Patterns)
| Quantifier | Meaning                          | Example Match                |
|------------|----------------------------------|------------------------------|
| `*`        | 0 or more times (greedy)        | `"a*"` → `"aaaa"`, `""`     |
| `+`        | 1 or more times                 | `"a+"` → `"a"`, `"aaa"`     |
| `?`        | 0 or 1 time (optional)          | `"colou?r"` → `"color"`, `"colour"` |
| `{n}`      | Exactly `n` times               | `"a{3}"` → `"aaa"`          |
| `{n,}`     | At least `n` times              | `"a{2,}"` → `"aa"`, `"aaaa"` |
| `{n,m}`    | Between `n` and `m` times       | `"a{2,4}"` → `"aa"`, `"aaa"`, `"aaaa"` |

<br>
<br>

---
---
# Groups and Alternatives
| Pattern   | Meaning                                      | Example Match  |
|-----------|----------------------------------------------|--------------- |
| `(abc)`   | Capturing group (matches `"abc"`)           | `"Hello (abc)"` |
| `(?:abc)` | Non-capturing group (matches but does not store) | `"abc"`  |
| `a|b`     | OR operator (matches `"a"` or `"b"`)        | `"apple"`, `"banana"` |

<br>
<br>

---
---
# Escape Sequences
| Character | Meaning in Regex | Escaped Version (Literal Match) | Example Match |
|-----------|----------------|--------------------------------|---------------|
| `.`       | Matches any character | `\.` (Matches a literal `.`) | `"example.com"` |
| `?`       | 0 or 1 occurrence (optional) | `\?` (Matches a literal `?`) | `"Is this correct?"` |
| `+`       | 1 or more occurrences | `\+` (Matches a literal `+`) | `"C++"` |
| `*`       | 0 or more occurrences | `\*` (Matches a literal `*`) | `"5 * 5 = 25"` |
| `[` `]`   | Character set | `\[` `\]` (Matches `[ ]` literally) | `"[value]"` |
| `\`       | Escape character | `\\` (Matches a literal `\`) | `"C:\Users\Admin"` |


<br>
<br>

---
---
# CHaracter Classes
| Pattern    | Matches |
|------------|---------|
| `[abc]`    | Any **one** of `a`, `b`, or `c` |
| `[0-9]`    | Any digit (same as `\d`) |
| `[A-Z]`    | Any **uppercase** letter |
| `[a-z]`    | Any **lowercase** letter |
| `[A-Za-z]` | Any letter (uppercase or lowercase) |
| `[aeiou]`  | Any vowel |
| `[^0-9]`   | Any **non-digit** (negation with `^`) |


<br>
<br>

---
---
# Anchors
| Pattern  | Matches |
|----------|---------|
| `^word`  | "word" **only at the start** of the string |
| `word$`  | "word" **only at the end** of the string |


<br>
<br>

---
---
# Lookarounds
| Lookaround | Syntax        | Meaning |
|------------|--------------|---------|
| **Positive Lookahead** | `(?=pattern)` | Matches if `pattern` **follows**, but does not capture it |
| **Negative Lookahead** | `(?!pattern)` | Matches if `pattern` **does not follow** |
| **Positive Lookbehind** | `(?<=pattern)` | Matches if `pattern` **precedes**, but does not capture it |
| **Negative Lookbehind** | `(?<!pattern)` | Matches if `pattern` **does not precede** |


<br>
<br>

---
---
# Flags
| Flag              | Description |
|-------------------|-------------|
| `re.IGNORECASE` / `re.I` | **Case-insensitive matching** (`"Hello"` = `"hello"`) |
| `re.MULTILINE` / `re.M`  | `^` and `$` work for **each line**, not just the entire string |
| `re.DOTALL` / `re.S`     | `.` matches **newlines (`\n`)** too |
