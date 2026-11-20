# Interview Guide: How to Explain My Resume

How to use this guide:
- Every resume bullet has a **30-second answer**, a **detailed STAR story** (Situation, Task, Action, Result), and **likely follow-up questions**.
- Keep client internals generic. Say "credit ratings data", "organization reference data", "a downstream analytics platform". Do **not** say internal app names (NSR, GSR, DAPS, CSD, VEGA), topic or table names, or URLs.
- Items marked **[confirm]** are reasonable technical explanations that you should check against your own memory before the interview. Don't claim anything you can't defend.

---

## 0. "Tell me about yourself" (60 seconds)

> I'm a Data Engineer at Ness Digital Engineering, working for S&P Global, one of the largest credit ratings and financial data companies. I joined in March 2025 as a fresher. I started on BI reporting in MicroStrategy and then moved to data engineering.
> Since then, I've built PySpark pipelines in Databricks, Kafka-to-Oracle streaming services in Java, and a change data capture pipeline that delivers data to Google BigQuery. I also automated incident management for Databricks jobs across six environments.
> My biggest achievement was a performance fix: a bulk load of over a million records used to take more than a day, and I brought it down to 16 minutes.
> Right now, I'm building a historical data migration pipeline. I'm also a Databricks Certified Data Engineer Professional.

---

## 1. Bulk-load performance: from over 24 hours to 16 minutes (your strongest story)

**30-second answer:**
A Java service consumes organization reference-data events from Kafka and loads them into Oracle through staging tables and PL/SQL procedures. Normal daily traffic was fine, but a bulk load of about 1 million records took more than a day. I found three bottlenecks: one in the processing framework, missing indexes, and redundant steps in the stored procedures. I fixed all three and also redesigned the composite primary keys. The same load now finishes in 16 minutes, about 90 times faster.

**STAR:**
- **Situation:** Upstream systems sometimes send bulk loads (re-publishes and corrections) of about 1M messages. Processing ran for more than 24 hours, which delayed downstream consumers and kept the backlog growing.
- **Task:** Find the root cause and make bulk loads finish within an acceptable window without breaking normal incremental processing.
- **Action:**
  1. **Measured first.** I compared the input and audit log tables with Kafka consumer lag to see where the time was going: consuming, staging, or procedures. **[confirm the exact method you used]**
  2. **Framework level.** The consumer framework was the blocker. It processed messages one at a time, or committed each message separately. **[confirm which one]** I changed it to handle messages in batches, which removed the per-message overhead.
  3. **Missing indexes.** The procedures looked up and merged into tables on columns that had no index, so every message triggered a full table scan. I added indexes on those join and lookup columns.
  4. **Redundant procedure steps.** Some procedures repeated work (duplicate lookups, unnecessary intermediate updates). I removed those steps.
  5. **Composite primary keys.** I analysed the business key of each table and corrected the composite primary keys. This made upserts and deletes match the right rows and made the key index useful for those operations.
  6. Tested with a production-sized volume in a lower environment and compared row counts before and after so that no data was lost.
- **Result:** 1M+ records went from over 24 hours to 16 minutes (more than 90x faster). Incremental processing was unchanged.

**Likely questions:**
- *How did you find the bottleneck?* Measure each stage separately (consume, stage, procedure). Use Oracle execution plans (`EXPLAIN PLAN`) to find full table scans. Check the AWR or SQL monitoring for the top SQL. **[mention only what you actually used]**
- *Why do indexes help, and what do they cost?* They turn full scans into index lookups. The cost is slower writes and extra storage, so I indexed only the columns used in joins and filters.
- *What is a composite primary key?* A primary key made of several columns that together uniquely identify a row, for example `(entity_id, identifier_type, effective_date)`. Column order matters for index use.
- *How did you make sure you didn't lose or duplicate data?* Row counts and checksums, comparing before and after on the same input, and idempotent upserts.

---

## 2. Event-driven change data capture (CDC) pipeline: Kafka → Oracle → GoldenGate → Kafka → BigQuery

**30-second answer:**
Ratings data arrives on a Kafka topic and gets loaded into Oracle tables. When a target table changes, database triggers turn the change into a JSON message and write it to a distribution table. Oracle GoldenGate replicates that table to a Kafka topic. From there, a connector publishes the messages to GCP Pub/Sub and on to BigQuery tables, where analytics teams use them.

**STAR:**
- **Situation:** Downstream analytics on Google Cloud needed every change to the ratings data, not nightly snapshots.
- **Task:** Build the flow from the source Kafka topic all the way to BigQuery.
- **Action:**
  - A Java Kafka consumer loads messages into Oracle staging tables. PL/SQL procedures then validate the data and upsert it into the target tables.
  - **Triggers** on the target tables build a JSON payload for each changed row. Each payload records the operation type (UPSERT or DELETE) and an event type. The triggers batch these payloads in a temporary table and write them to a message distribution table once per statement.
  - **Oracle GoldenGate** replicates new rows from the distribution table into a Kafka topic (log-based CDC).
  - A connector publishes the messages to **GCP Pub/Sub**, and they land in **BigQuery** tables.
  - Acknowledgments come back through Pub/Sub and are stored for auditing. That is project 1 in the Projects section.
- **Result:** The downstream platform receives every change incrementally, and each step has an audit trail. **[add a volume or latency figure if you know one]**

**Likely questions:**
- *Why triggers plus GoldenGate instead of only GoldenGate?* Triggers let us build a business-shaped JSON message from several related tables. GoldenGate then reliably moves those messages out of Oracle using the redo logs, without polling.
- *How do you handle deletes?* The trigger writes a message with operation type DELETE, and the consumer removes or soft-deletes the record downstream.
- *How do you guarantee ordering?* With a partition key per business entity, so all changes for the same entity go to the same Kafka partition. **[confirm]**
- *What is CDC?* Change data capture: reading only the inserts, updates and deletes from a source instead of full reloads.

---

## 3. Historical data migration in Databricks (current work)

**30-second answer:**
I'm building a PySpark pipeline in Databricks that backfills several years of credit ratings history from multiple source systems. It reads the source tables, transforms them into the target data model, validates them, and publishes them to Kafka for downstream systems.

**Points to explain:**
- Why a backfill is different from incremental loading: very high volume, so it must be safe to rerun and must not flood the consumers.
- Techniques you can mention: partitioned reads, Spark joins and window functions to pick the latest version of each record, controlled batch sizes when publishing to Kafka, and reconciling counts between source and target. **[mention only what you use]**

**Likely questions:** *How do you avoid duplicates if the job fails halfway?* Use idempotent keys or upserts, keep checkpoints or batch watermarks, and rerun only the failed batch.

---

## 4. Organization reference data ETL (PySpark → Kafka)

**30-second answer:**
I built Databricks PySpark pipelines that collect organization data from several source tables, clean and join it into one consolidated record per organization, validate it, and publish it as JSON to Kafka for downstream search and other services.

**Points to explain:** joins across sources, deduplication, schema mapping to a target JSON structure, data-quality checks before publishing, and Kafka producer settings such as batch size and keys.

---

## 5. Databricks job-failure incident automation (6 environments)

**30-second answer:**
Before this, someone had to find failed Databricks jobs and create tickets for them by hand. I built a Python notebook that runs on a schedule, checks every job in the workspace through the Databricks Jobs REST API, extracts the real error message, and automatically creates an Azure DevOps ticket. If the same issue is already open, it updates that ticket instead of creating a duplicate. It runs in 6 environments: Dev, SI, QA, UAT, Prod and DR.

**Technical details (from my code):**
- **Discovery:** lists all jobs with the paginated `jobs/list` API (100 per page). For each job, it reads the last 5 runs and keeps those that **ended within a 6-hour lookback window** with result `FAILED` or `TIMEDOUT`.
- **Root-cause extraction:** for multi-task jobs, it checks each failed task. When Databricks shows only the generic "Workload failed, see run output" message, it calls `runs/get-output` to get the real notebook error or stack trace.
- **Noise reduction:**
  - It skips runs cancelled by a user (pattern matching on the message).
  - It groups repeated failures by an error signature (the first 300 characters).
  - Before creating a ticket, it runs a **WIQL query** for an open issue with the same title created in the **last 7 days**. If it finds one, it adds the new error to that issue instead of opening a new one.
- **Priority by environment:** Prod → Priority 1 / Critical. QA and UAT → Priority 2 / High. Others → Priority 3 / Medium. It also adds tags for the environment and the job.
- **Security:** authenticates to Azure DevOps with Microsoft Entra ID (service principal or managed identity), not personal tokens. Configuration is read from a config file stored in S3.
- **Resilience:** if the configured area path is invalid, it retries without it, so tickets are never lost.

**STAR result:** failures are tracked automatically with the actual error and a link to the run. On-call engineers no longer search for failures by hand, and duplicate tickets are avoided. **[add a count if you know one, for example "~N tickets a month"]**

**Likely questions:**
- *Why poll instead of using job-failure webhooks?* One central monitor covers every job, including new ones, without configuring each job, and it can de-duplicate across runs.
- *How do you avoid ticket storms?* The error signature grouping, the 7-day duplicate check, and skipping cancelled runs.

---

## 6. MicroStrategy BI dashboard

**30-second answer:**
This was my first assignment. I built an operational dashboard in MicroStrategy that tracks how long work items take at each stage of a business process. It shows average turnaround days per stage with the change from the previous year, work in progress by year, sector and department, and monthly filed-versus-distributed volumes. Leadership uses it to see where the process slows down.

**What was on it (generic terms):**
- KPI cards: average days for each stage transition, comparing this year with last year (percentage change).
- Breakdowns of work in progress by year, by industry sector, and by department (donut charts).
- A monthly trend of submitted versus completed counts, with month-over-month change.
- Tabs for different business segments, plus an advanced search page.

**Likely questions:** how you modelled the metrics (attributes, metrics, filters in MicroStrategy), where the data came from (Oracle tables or views), and how you checked the numbers against source queries.

---

## 7. Pub/Sub acknowledgment audit service (Java / Spring Boot)

**30-second answer:**
When the downstream platform receives our messages, it sends back acknowledgments through GCP Pub/Sub. I built a Spring Boot service that consumes those acknowledgments and stores them as audit records in Oracle. This closes the loop on delivery tracking: we can report which messages were delivered and which failed.

**Likely questions:** Pub/Sub ack and nack and redelivery, idempotent inserts (so a duplicate acknowledgment doesn't create a second row), and exception handling and logging.

---

## Words to use instead of internal names

| Internal / avoid | Say instead |
|---|---|
| NSR, GSR, canonical model | credit ratings data platform / ratings data model |
| DAPS, Org Search | organization reference data |
| nsr-processor-kf | Java Kafka consumer service / processing framework |
| CSD Timelines Dashboard | operational turnaround dashboard |
| Table, topic, procedure names | staging tables, target tables, stored procedures |

## Numbers to remember
- **1M+ records: more than 24 hours → 16 minutes (more than 90x faster)**
- **6 environments** monitored; **6-hour** lookback window; **7-day** duplicate window
- Experience: **Mar 2025 – present**
