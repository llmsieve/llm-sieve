---
date: 2026-06-10
authors:
  - sieve-engineering
categories:
  - Design
description: >
  mem0's SDK, Zep's managed platform, and Sieve's transparent proxy
  solve agent memory in three different shapes. How to pick.
image: https://llmsieve.dev/assets/social/social-card-three-shapes-of-memory.png
---

# Sieve, mem0, Zep: three shapes of agent memory

*Applies to: Sieve v1.0.x*

If you're shopping for a memory layer for an LLM agent in 2026, three
credible shapes are on the table: an SDK you call from application
code (mem0), a managed platform you push conversations into (Zep),
and a transparent proxy that sits in the traffic path (Sieve). They
get compared as if they were interchangeable. They aren't — and the
differences that matter are architectural, not benchmark decimals.

We build Sieve, so read this knowing where our incentives sit. In
exchange: every claim about the other two links to *their* docs and
repos, fetched and quoted on 2026-06-10, and we'll be plain about
where each of them is the better choice.

<!-- more -->

## The integration contract

The deepest difference between the three is who has to know the
memory layer exists.

**mem0 is an SDK with explicit calls.** Your application calls
`memory.search()` before each LLM turn, makes its own LLM call, then
calls `memory.add()` afterwards. mem0's docs describe the add
pipeline plainly: ["You trigger this pipeline with a single add
call"](https://docs.mem0.ai/core-concepts/memory-operations/add).
There is also an [OpenAI-compatible client
mode](https://docs.mem0.ai/open-source/features/openai_compatibility)
that wraps memory around chat-completion calls — though it's still a
client you adopt in code, not a network-level interposition. mem0
ships Python and TypeScript SDKs, as a
[library, self-hosted server, or managed
cloud](https://github.com/mem0ai/mem0).

**Zep is a platform with a push/pull contract.** You create users and
threads, push each message in with `thread.add_messages()`, and pull
a "context block" out with `thread.get_user_context()`, which your
app then splices into its own prompt — the [quickstart
guide](https://help.getzep.com/quick-start-guide) walks exactly that
loop. The LLM call remains entirely yours. Server-side, Zep builds a
temporal graph of what it ingests.

**Sieve is a proxy.** You change one base URL —
`127.0.0.1:11434` becomes `127.0.0.1:11435` — and keep your code.
Stripping repeated context, learning facts, and injecting relevant
ones back happens in the traffic path; the client doesn't know the
proxy is there. The trade-offs of that choice (and there are real
ones) got their own post: [Why Sieve](2026-06-09-why-sieve.md).

None of these is "right." They encode different beliefs about where
memory belongs: in your code, in a platform, or in the pipe.

## What happens on a turn

**mem0** runs LLM-based extraction over your messages: an LLM call
pulls out facts, conflicts are resolved, and results land in a vector
store. As of its [April 2026 v3
algorithm](https://github.com/mem0ai/mem0), extraction is
"single-pass ADD-only... one LLM call, no UPDATE/DELETE", and ["Mem0
requires an LLM to function, with `gpt-5-mini` from OpenAI as the
default"](https://github.com/mem0ai/mem0). Worth knowing if you read
older comparisons: graph memory was [removed in
v3](https://docs.mem0.ai/migration/oss-v2-to-v3) in favour of
built-in entity linking, so "vector + graph" descriptions of mem0 are
out of date.

**Zep** ingests "episodes" (messages, JSON, text) and maintains a
temporal context graph — facts carry validity intervals, so it can
represent "worked at X until March, then Y." That engine is
[Graphiti](https://github.com/getzep/graphiti), open source under
Apache 2.0, and genuinely interesting work. Running Graphiti yourself
requires a graph database (Neo4j, FalkorDB, or Amazon Neptune) plus
an LLM for ingestion — it "defaults to OpenAI for LLM inference and
embedding," with local servers (Ollama, vLLM) supported.

**Sieve** does two jobs per turn. Outbound, it strips what the model
has already seen — tool schemas, repeated instructions, stale history
— before the payload leaves your machine. In the background it
extracts durable facts using the same LLM endpoint you already
configured and embeds them with a local model (no separate API key,
no extra vendor). On later turns it injects only the facts the turn
needs, and gates absence: a question about something the store has
never seen should produce a refusal, not an invention.

The emphasis differs accordingly. mem0 and Zep are primarily
*recall* systems — their benchmarks measure answer accuracy over long
histories. Sieve treats *payload reduction* as a first-class goal
alongside recall, because we think the per-turn token bill is the
quieter, larger problem — the argument in [The hidden cost of
context](2026-06-10-hidden-cost-of-context.md).

## Where your data lives

This one divides the field cleanly.

**mem0**: your choice. The OSS library defaults to a [local Qdrant
plus SQLite history](https://docs.mem0.ai/open-source/overview), with
~20 vector backends in Python; the hosted platform keeps memories on
their cloud. Self-hosting the server is supported and documented.

**Zep**: the cloud is the product. The self-hostable Community
Edition was [deprecated in April
2025](https://blog.getzep.com/announcing-a-new-direction-for-zeps-open-source-strategy/)
("we've decided to stop maintaining and releasing Zep Community
Edition"), with open-source effort concentrated on Graphiti. Zep
Cloud [operates in AWS
us-west-2](https://help.getzep.com/bring-your-own-key); enterprise
tiers add bring-your-own-key and deploy-in-your-VPC options, and the
platform holds [SOC 2 Type II
certification](https://help.getzep.com/security-compliance) with
HIPAA BAAs for enterprise customers.

**Sieve**: local only, by design. Facts live in a SQLCipher-encrypted
SQLite file under `~/.sieve/`, embeddings are computed locally, there
is no account and no telemetry. If your LLM endpoint is local too,
nothing leaves the machine. There is deliberately no cloud to trust —
which also means no managed offering if you wanted one.

## What they cost

As of 2026-06-10, from the vendors' own pricing pages:

- **mem0**: OSS is Apache 2.0. The [hosted
  platform](https://mem0.ai/pricing) has a free tier (10,000 add
  requests and 1,000 retrieval requests a month) with paid tiers from
  $19/month. Self-hosted, your real cost is the extraction LLM calls
  and the vector store.
- **Zep**: credit-metered ingestion — ["1 credit per Episode up to
  350 bytes... 0 credits for retrieval, storage, threads, users, and
  graph storage"](https://www.getzep.com/pricing/). Free tier is
  1,000 credits/month; the Flex tier is $104/month billed annually
  ($125 month-to-month) including 50,000 credits.
- **Sieve**: Apache 2.0, no hosted tier, no metering. The cost is
  your own compute — the same endpoint that runs your agent runs the
  extraction.

## The numbers everyone quotes

mem0's [research page](https://mem0.ai/research) (updated May 2026)
reports LoCoMo 92.5 and LongMemEval 94.4 at under 7,000 tokens per
retrieval. Zep's [research page](https://www.getzep.com/research/)
reports LoCoMo 94.7% and LongMemEval 90.2% with sub-200ms retrieval.

Notice anything? Each vendor leads the other on one of the same two
benchmarks. That's not cherry-picking by either of them so much as a
property of the genre: different readers, judges, and harnesses
produce different numbers on the same datasets, and every vendor
naturally publishes the configuration that suits their architecture.
Treat all such tables — including any we publish — as claims about a
specific harness, not facts about the product.

Sieve's position: we don't currently publish cross-tool benchmark
numbers, and we'd rather hand you the harness than the table. `sieve
benchmark` runs a baseline-vs-Sieve comparison on *your* hardware
with *your* model in five to ten minutes, and the demo's absence-trap
turn is reproducible on any model you pull. When we do publish
numbers, they link to methodology you can re-run.

## Where each one wins

**Pick mem0 when** memory is a feature of your application logic —
you want programmatic control over what gets remembered for which
user, you're building multi-tenant SaaS, you already live in Python
or TypeScript, and an extraction LLM call per add is acceptable. The
ecosystem is the largest of the three (~58k GitHub stars as of June
2026) and the backend flexibility is real.

**Pick Zep when** you want memory as managed infrastructure — you
have compliance requirements (SOC 2, HIPAA BAA), you want temporal
reasoning over entity relationships, and you're happy with a cloud
service in the loop. Graphiti alone is worth a look (~27k stars) if
you want to run a temporal context graph yourself and don't mind
operating Neo4j and paying for ingestion-time LLM calls.

**Pick Sieve when** the thing you're protecting is the client code
and the data. You can't or won't modify the agent (closed tools,
many heterogeneous clients, or just discipline about coupling); you
want everything on disk encrypted and on your own machine; you care
as much about the size of every outbound payload as about recall; and
single-user-per-store matches your deployment — a personal agent, a
workstation, one proxy per user.

**Sieve is the wrong choice when** you need a multi-tenant memory
backend for a hosted product, graph-shaped queries over entity
relationships, or a vendor to operate it for you. Those are mem0 and
Zep's home turf, and we'd rather you pick them than bend Sieve into a
shape it doesn't have. It's also the youngest project of the three —
v1.0 shipped this month — and doesn't pretend otherwise.

## The rubric, compressed

| | mem0 | Zep | Sieve |
|---|---|---|---|
| Shape | SDK (+optional client wrapper) | Managed platform | Transparent proxy |
| Code changes | add/search calls | push/pull + prompt splice | base URL only |
| Data lives | your infra or their cloud | their cloud (us-west-2)* | your disk, encrypted |
| Extraction LLM | required (default OpenAI) | platform-side / required for Graphiti | your existing endpoint |
| Primary goal | recall accuracy | recall + temporal graph | payload reduction + recall + absence handling |
| License / price | Apache 2.0 / free tier, from $19/mo | Graphiti Apache 2.0 / from $104/mo | Apache 2.0 / free |

*Enterprise BYOK/BYOC options exist.

Memory for agents is young enough that the shapes haven't converged,
and honest comparison beats benchmark arithmetic. Know which contract
you're signing — code, platform, or pipe — and the rest of the
decision mostly makes itself.

---

*This post was drafted with AI assistance and reviewed by the Sieve
maintainer before publication. Competitor facts were fetched from the
linked primary sources on 2026-06-10 and quoted verbatim where shown;
if we've misrepresented either project, [open an
issue](https://github.com/llmsieve/llm-sieve/issues) and we'll
correct it. Sieve is open source under Apache 2.0.*
