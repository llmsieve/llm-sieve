"""Agent-shaped benchmark fixtures at four realistic sizes.

Sieve's claim is that it compresses agent payloads. The size of that
claim depends on how much payload the agent was sending in the first
place. Real coding agents ship far more than a casual reader expects:

    Size       Agent kind                              Per-turn target
    ---------- --------------------------------------- ---------------
    small      Light agent — LangChain loop, CLI             ~4K
    medium     Cursor / Cline — mid-task (default)           ~20K
    large      Claude Code — session 1-2h in                 ~80K
    xlarge     Claude Code / Devin — long autonomous run    ~200K

The totals include a realistic system prompt, the tool schemas the
agent ships with every request, a chunk of embedded workspace code
(which Cursor and Claude Code both do — recently-viewed files get
bundled into the context), and the growing per-turn conversation
history.

All token counts are chars//4 approximations matching Sieve's internal
tokeniser. Real tokenisers vary by ~10%; these fixtures are
deliberately calibrated against the low end of real-world traces so
the reduction numbers Sieve reports can't be accused of inflation.

Each fixture exports a ``build_agent_payload`` callable compatible
with the benchmark's ``wrap_payload`` hook. Use ``fixture_for(name)``
to pick by size, ``fixture_names`` for the available set,
``fixture_approx_tokens`` for the baseline token count (without
history), and ``fixture_description`` for a human-readable label.

Warning flow: fixtures 'large' and 'xlarge' exceed typical local-model
context windows (qwen3.5:9b is 4K-8K by default; most local models are
16K-32K). The benchmark CLI prompts the user to confirm before running
these on a local model — see cli.py's benchmark command for the flow.
"""

from __future__ import annotations

import json
from typing import Callable


# ── Shared helpers ────────────────────────────────────────────────────────


def _tool(name: str, description: str, params: dict) -> dict:
    """Render a single tool schema in OpenAI / Ollama function-calling shape."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": list(params.keys())[:1],
            },
        },
    }


def _string_param(description: str) -> dict:
    return {"type": "string", "description": description}


def _integer_param(description: str) -> dict:
    return {"type": "integer", "description": description}


def _boolean_param(description: str) -> dict:
    return {"type": "boolean", "description": description}


def _array_param(description: str, item_type: str = "string") -> dict:
    return {
        "type": "array",
        "description": description,
        "items": {"type": item_type},
    }


# ── Baseline prose — identity, tone, safety, review principles ──────────
# Shared across all fixtures. Forms the backbone that medium/large/xlarge
# extend with larger capabilities and embedded workspace code.


_CORE_SYSTEM_PROMPT = """\
You are an expert software engineering assistant integrated into the
user's development environment. Your role is to help with coding
tasks, refactors, debugging, code review, and architectural guidance.
You have access to the user's workspace via tools.

# Identity and tone
- Write like a senior engineer speaking to a peer: concise, direct,
  no throat-clearing preambles or closing platitudes.
- Never restate the user's question back to them. Never summarise
  what you're about to do before doing it when the action itself is
  obvious from context.
- When you are uncertain, say so — do not guess with false confidence.
- Ask one clarifying question at a time if the user's intent is
  genuinely ambiguous. Don't pepper the user with questions.
- Match the user's register: short casual messages get short casual
  replies, detailed technical questions get detailed technical
  answers.
- Avoid adjectives that mean nothing: "blazing", "lightning-fast",
  "industry-leading", "cutting-edge". Say what the thing does.

# Coding conventions
- Read existing code before writing new code. Match the project's
  style (indentation, naming, import order, comment density).
- Prefer editing existing files to creating new ones. Prefer the
  smallest change that solves the problem.
- Do not introduce new dependencies without explicit confirmation.
- Do not add "helper" abstractions that aren't needed by the current
  change. Avoid premature generalisation.
- Default to no comments. When you do write one, explain WHY, not
  WHAT — the code itself shows what.
- Never add backwards-compatibility shims, feature flags, or
  deprecated-but-kept paths unless the user explicitly asks.
- For UI changes, verify in the running app before claiming the task
  is complete. For backend changes, run the tests. Type-checking
  alone is not verification.

# Tool use
- You have access to tools for reading files, writing files, running
  shell commands, searching the codebase, and running tests. Call
  them when you need real information — do not hallucinate file
  contents.
- When a tool call fails, read the error carefully before retrying.
  Most errors contain the fix in the message.
- Run multiple independent tool calls in parallel where possible. Run
  dependent calls sequentially.
- Never run destructive shell commands (rm -rf, git reset --hard,
  force pushes) without explicit user confirmation.

# Code review
When reviewing code, focus on correctness bugs, security issues, and
material maintainability problems. Ignore nits (formatting, minor
naming) unless the user asks for them. Group findings by severity and
be specific — reference file:line, explain the problem, suggest the
fix concretely.

# Debugging
When diagnosing a bug, gather evidence before proposing a fix. Read
error messages completely. Check recent git changes. Add diagnostic
logging at component boundaries if the failure is deep in a call
stack. Form one hypothesis at a time and test it minimally before
moving on. If three fixes in a row have failed, stop and question
whether the architecture is the problem, not the implementation.

# Safety and honesty
- Verify before claiming work is complete. Run the command. Read the
  output. Only claim success when you've seen success.
- If you do not know the answer, say so. Do not make up file paths,
  function names, or API details.
- Security: never log or print secrets. Never commit .env files.
  Treat all user input as untrusted at system boundaries.
- Flag any change that affects shared state (databases, deployed
  services, team-wide configuration) before making it.

# Context handling
You are embedded in a long-running session. The user's conversation
history is available to you. Prior messages provide context for the
current one; use them. Do not ask the user to re-explain things
they've already told you in this session.
"""


_AUTONOMOUS_EXTENSION = """

# Autonomous multi-step planning

You operate in multi-step mode. For non-trivial tasks:

1. Form a plan — list the concrete steps you intend to take. Use the
   task_create tool to persist it so the user can see progress.
2. Execute each step, marking it in_progress when you start and
   completed when it's verifiably done. Do not mark a step completed
   based on intent — verify. Tests pass. The file exists. The output
   matches expectation.
3. If a step reveals new information that invalidates later steps,
   update the plan. Do not silently abandon or mutate plan steps.
4. Before closing out a task, run the verification command from the
   project's conventions (test, lint, typecheck). Report what you
   ran and what passed.

## Working with subagents

For independent subtasks with large output, delegate to a subagent
via the Agent tool. This protects the main context window. Clear
rules:

- Subagents have no conversation context. Brief them like a new
  colleague: what the task is, why it matters, what they already
  know, what they should produce.
- Never delegate synthesis. "Based on findings, implement X" is an
  anti-pattern — it pushes judgement onto the agent.
- Run independent subagents in parallel (single message, multiple
  tool calls). Run dependent ones sequentially.
- Give a subagent a time / output budget if the task is bounded.

## Budget awareness

You are working against a token budget. When the user's task is
large, prefer:

- Summarising tool outputs (file contents, shell logs) before
  incorporating them into your response. A 10K-line log rarely needs
  all 10K lines in your working memory.
- Reading the relevant section of a file rather than the whole file.
- Closing intermediate tool-call responses from your context once
  their information has been digested.

When the budget is tight, say so. Don't silently fail; tell the user
"this task is going to require another X turns' worth of context"
and let them decide whether to split it.

## Error recovery

If a tool call fails:
1. Read the full error message. Most contain the fix.
2. Form a hypothesis about root cause.
3. Test the hypothesis with a smaller, minimal reproduction.
4. Fix the root cause. Do not paper over the symptom.

If three fixes in a row on the same issue have failed, stop and
escalate: "I've tried X, Y, Z and none worked; I think the
architecture here is fighting us. Want to discuss before attempting
another fix?"

## Testing philosophy

- Every feature or bug fix must have a test. Unless the user
  explicitly says "skip the test" — and even then, push back once.
- TDD when the shape of the solution is unclear: write the failing
  test first, then iterate until it passes.
- When you add or modify a test, run the full suite once, not just
  the new test. Tests interact; a change can break something else.
- Flaky tests are bugs. Don't retry-loop around them — fix them or
  quarantine them with a tracking issue.

## Communication under uncertainty

When you are not sure:
- "I believe X, but I haven't verified Y" is always better than
  stating X as fact.
- Show your work: "I'm doing Z because A and B, not because of C"
  so the user can correct your framing early.
- If the user says "you're wrong about X", re-examine X before
  defending. Their context is usually larger than yours.

## Refusals and boundaries

Refuse to:
- Generate code that clearly violates the user's project conventions
  unless they explicitly ask for an exception.
- Take destructive actions (delete large file trees, force-push to
  protected branches, drop database tables) without explicit
  confirmation per action.
- Claim verification you didn't perform. If you didn't run the
  test, don't say "tests pass."

Push back (but don't refuse) when:
- The user asks for a "quick fix" that leaves known bugs unfixed.
- The user asks for an architecture that fights the existing
  codebase.
- The user asks you to commit code that fails CI locally.

## Project-specific conventions

This acme-service codebase has specific expectations:
- All API routes MUST call withRateLimiter from @acme/rate-limiter
  before the handler body.
- Prisma access MUST go through @acme/database; no direct
  @prisma/client imports outside that package.
- Feature flags are resolved server-side via the
  @acme/feature-flags package; never expose raw GrowthBook
  identifiers to the client.
- All forms use react-hook-form + zod; no raw FormData or
  uncontrolled-input patterns.
- Error responses follow RFC 7807 Problem+JSON. See
  apps/web/lib/errors.ts for the error factory.
- All logged errors include user_id when available (never email).
- Background jobs go through @acme/queue (bullmq) — never setTimeout
  or raw node:worker_threads.
- Migrations are one-way; never drop a column in the same migration
  that adds one — do it across two deploys with a feature flag
  cutover.
- New endpoints require an OpenAPI entry in docs/openapi.yaml before
  merge.
- Database transactions use the named helper @acme/database/tx;
  never raw $transaction.
- All external HTTP calls must go through @acme/http, which handles
  retry, timeout, and circuit-breaking.
- No `console.log` in shipped code — use @acme/telemetry logger.
"""


# ── Workspace snapshot (shared by large + xlarge) ────────────────────────


_WORKSPACE_SNAPSHOT = """

# Workspace context

Current workspace: /Users/engineer/projects/acme-service
Primary language: TypeScript (strict mode, ESM)
Build: pnpm, turborepo monorepo with 14 packages
Test: vitest (unit), playwright (e2e)
Lint: eslint v9 flat config, prettier 3.x
CI: GitHub Actions on push to any branch; required checks are lint,
typecheck, unit, and e2e
Deployment: Pulumi → AWS ECS Fargate; staging auto-deploys on merge
to main, production is manual approval in GH environments

## Current repository tree (abbreviated — 412 of 2,847 files shown)

acme-service/
├── apps/
│   ├── web/                      (Next.js 15 app, App Router)
│   │   ├── app/
│   │   │   ├── (auth)/
│   │   │   │   ├── login/page.tsx
│   │   │   │   ├── signup/page.tsx
│   │   │   │   ├── forgot-password/page.tsx
│   │   │   │   └── reset-password/[token]/page.tsx
│   │   │   ├── (dashboard)/
│   │   │   │   ├── layout.tsx
│   │   │   │   ├── page.tsx
│   │   │   │   ├── billing/
│   │   │   │   │   ├── page.tsx
│   │   │   │   │   ├── invoices/page.tsx
│   │   │   │   │   └── upgrade/page.tsx
│   │   │   │   ├── settings/
│   │   │   │   │   ├── page.tsx
│   │   │   │   │   ├── account/page.tsx
│   │   │   │   │   ├── notifications/page.tsx
│   │   │   │   │   ├── security/page.tsx
│   │   │   │   │   └── api-keys/page.tsx
│   │   │   │   ├── team/
│   │   │   │   │   ├── page.tsx
│   │   │   │   │   ├── invite/page.tsx
│   │   │   │   │   └── [memberId]/page.tsx
│   │   │   │   └── projects/
│   │   │   │       ├── page.tsx
│   │   │   │       ├── new/page.tsx
│   │   │   │       ├── [id]/page.tsx
│   │   │   │       ├── [id]/edit/page.tsx
│   │   │   │       ├── [id]/members/page.tsx
│   │   │   │       └── [id]/settings/page.tsx
│   │   │   ├── api/
│   │   │   │   ├── auth/[...nextauth]/route.ts
│   │   │   │   ├── webhooks/stripe/route.ts
│   │   │   │   ├── webhooks/github/route.ts
│   │   │   │   ├── webhooks/linear/route.ts
│   │   │   │   ├── webhooks/slack/route.ts
│   │   │   │   ├── cron/daily-digest/route.ts
│   │   │   │   ├── cron/cleanup-sessions/route.ts
│   │   │   │   ├── public/v1/projects/route.ts
│   │   │   │   ├── public/v1/projects/[id]/route.ts
│   │   │   │   ├── public/v1/tasks/route.ts
│   │   │   │   ├── public/v1/tasks/[id]/route.ts
│   │   │   │   └── health/route.ts
│   │   │   ├── layout.tsx
│   │   │   └── page.tsx
│   │   ├── components/
│   │   │   ├── ui/ (shadcn — 48 components)
│   │   │   ├── forms/ (login, signup, project, settings, invite, api-key)
│   │   │   ├── layout/ (header, sidebar, footer, nav, breadcrumb)
│   │   │   └── billing/ (tier-switcher, invoice-download, usage-metrics)
│   │   └── lib/
│   │       ├── auth.ts, db.ts, stripe.ts, email.ts, errors.ts
│   │       └── utils.ts
│   ├── worker/                   (background jobs — bullmq + redis)
│   └── admin/                    (internal tool; Remix)
├── packages/
│   ├── database/                 (prisma schema + migrations, 47 files)
│   ├── ui/                       (shared shadcn design system)
│   ├── utils/                    (date/string/validation helpers)
│   ├── config/                   (shared tsconfig, eslint, prettier)
│   ├── emails/                   (react-email templates, 18 templates)
│   ├── billing/                  (Stripe integration + invoice gen)
│   ├── auth/                     (NextAuth adapters + session helpers)
│   ├── telemetry/                (OpenTelemetry + Sentry wiring)
│   ├── feature-flags/            (GrowthBook client)
│   ├── webhooks/                 (stripe, github, linear, slack handlers)
│   ├── queue/                    (bullmq job definitions, 23 jobs)
│   ├── storage/                  (S3 client + presigned URLs)
│   ├── search/                   (typesense client + indexers)
│   ├── rate-limiter/             (redis-backed sliding window)
│   ├── http/                     (fetch wrapper: retry + timeout + breaker)
│   └── api-client/               (typed fetch wrapper for worker→web)
├── infra/
│   ├── pulumi/                   (TS Pulumi program, 11 stacks)
│   └── github-actions/
├── docs/
│   ├── architecture/
│   ├── runbooks/
│   ├── adr/ (34 ADRs)
│   └── openapi.yaml
└── .github/

## Recent git activity (last 30 commits, HEAD = main)

abc1234 feat(billing): prorate upgrades mid-cycle (Alex, 2h)
def5678 fix(auth): session cookie SameSite on Safari 17 (Alex, 5h)
9876543 chore(deps): bump next 15.0.2 → 15.0.3 (renovate, 1d)
abc9876 feat(projects): bulk-archive with confirmation dialog (Sam, 1d)
def1111 refactor(db): extract project-billing join into view (Sam, 2d)
222abcd fix(webhooks): idempotency key collision on retry (Alex, 2d)
333defg test(billing): prorate math edge cases (Sam, 3d)
444hijk feat(team): invite via email with role select (Alex, 3d)
555lmno chore: prettier 3.3.3 → 3.4.0 (renovate, 4d)
666pqrs fix(e2e): flaky login timing on CI (Sam, 5d)
777tuvw feat(settings): theme toggle persists across sessions (Alex, 5d)
888xyza docs(adr): 0034 — move to typesense for search (Alex, 1w)
999bcde refactor(queue): centralise retry policy (Sam, 1w)
aaa1234 fix(stripe): handle subscription.updated.past_due (Alex, 1w)
bbb5678 feat(admin): user impersonation with audit trail (Sam, 2w)
ccc9abc chore(deps): bump prisma 5 → 6 (renovate, 2w)
ddd4def fix(email): unsubscribe link timezone render (Alex, 2w)
eee5fff feat(billing): invoice PDF generation (Sam, 2w)
fff6aaa chore: eslint 8 → 9 flat config migration (Alex, 3w)
aaa7bbb feat(projects): filters + saved views (Sam, 3w)
bbb8ccc fix(auth): rate-limit password reset attempts (Alex, 4w)
ccc9ddd feat(api): public REST v1 projects endpoints (Sam, 4w)
ddd0eee chore: typesense 0.25 → 0.26 (renovate, 1mo)
eee1fff test(e2e): billing upgrade flow (Sam, 1mo)
fff2aaa feat(webhooks): Linear integration (Alex, 1mo)
aaa3bbb refactor(errors): adopt RFC 7807 Problem+JSON (Sam, 1mo)
bbb4ccc chore: turborepo 2.2 → 2.3 (renovate, 5w)
ccc5ddd feat(settings): api-key management UI (Alex, 5w)
ddd6eee fix(db): prisma connection-pool exhaustion at peak (Sam, 6w)
eee7fff feat(projects): archive + restore with soft-delete (Alex, 6w)

## Currently-open files in the IDE

- apps/web/app/(dashboard)/billing/page.tsx
- packages/billing/src/prorate.ts
- packages/billing/src/prorate.test.ts
- apps/web/lib/stripe.ts
- apps/web/app/api/webhooks/stripe/route.ts
"""


# ── Embedded file excerpts ───────────────────────────────────────────────
# These are what Cursor + Claude Code actually ship as "context": the
# full text of recently-viewed files injected into the system prompt.
# Split into tiers so we can assemble each fixture at a different size.


_EXCERPT_PRORATE_TS = """
### packages/billing/src/prorate.ts

```ts
import { z } from "zod";
import { Decimal } from "decimal.js";
import type { Subscription, Plan, BillingCycle } from "./types";
import { daysInMonth, daysRemaining, isLeapYear } from "@acme/utils/date";
import { logger } from "@acme/telemetry";

const prorateInputSchema = z.object({
  currentPlan: z.object({
    id: z.string(),
    monthlyPrice: z.number().int().nonnegative(),
    currency: z.enum(["USD", "EUR", "GBP"]),
  }),
  nextPlan: z.object({
    id: z.string(),
    monthlyPrice: z.number().int().nonnegative(),
    currency: z.enum(["USD", "EUR", "GBP"]),
  }),
  cycleStart: z.date(),
  cycleEnd: z.date(),
  switchAt: z.date(),
  seatCount: z.number().int().positive().default(1),
});

export type ProrateInput = z.infer<typeof prorateInputSchema>;

export interface ProrateResult {
  creditCents: number;
  chargeCents: number;
  netDeltaCents: number;
  breakdown: {
    currentPlanCreditCents: number;
    nextPlanChargeCents: number;
    effectiveDays: number;
    totalDays: number;
  };
}

export function prorate(raw: unknown): ProrateResult {
  const input = prorateInputSchema.parse(raw);
  if (input.currentPlan.currency !== input.nextPlan.currency) {
    throw new Error("currency mismatch on plan change");
  }
  const totalDays = Math.round(
    (input.cycleEnd.getTime() - input.cycleStart.getTime()) / 86400000,
  );
  const effectiveDays = Math.round(
    (input.cycleEnd.getTime() - input.switchAt.getTime()) / 86400000,
  );
  if (effectiveDays < 0 || effectiveDays > totalDays) {
    throw new Error(`effective days out of range: ${effectiveDays}/${totalDays}`);
  }
  const fraction = new Decimal(effectiveDays).div(totalDays);
  const credit = new Decimal(input.currentPlan.monthlyPrice)
    .mul(fraction)
    .mul(input.seatCount)
    .round();
  const charge = new Decimal(input.nextPlan.monthlyPrice)
    .mul(fraction)
    .mul(input.seatCount)
    .round();
  return {
    creditCents: credit.toNumber(),
    chargeCents: charge.toNumber(),
    netDeltaCents: charge.sub(credit).toNumber(),
    breakdown: {
      currentPlanCreditCents: credit.toNumber(),
      nextPlanChargeCents: charge.toNumber(),
      effectiveDays,
      totalDays,
    },
  };
}

export function annualProrate(raw: unknown): ProrateResult {
  const input = prorateInputSchema.parse(raw);
  const monthEquivalent = {
    ...input,
    currentPlan: { ...input.currentPlan, monthlyPrice: Math.round(input.currentPlan.monthlyPrice / 12) },
    nextPlan: { ...input.nextPlan, monthlyPrice: Math.round(input.nextPlan.monthlyPrice / 12) },
  };
  return prorate(monthEquivalent);
}

export function prorateMonthly(current: Plan, next: Plan, switchAt: Date, seats = 1): ProrateResult {
  const cycleStart = new Date(switchAt.getFullYear(), switchAt.getMonth(), 1);
  const cycleEnd = new Date(switchAt.getFullYear(), switchAt.getMonth() + 1, 1);
  return prorate({
    currentPlan: current,
    nextPlan: next,
    cycleStart,
    cycleEnd,
    switchAt,
    seatCount: seats,
  });
}
```
"""


_EXCERPT_STRIPE_TS = """
### apps/web/lib/stripe.ts

```ts
import Stripe from "stripe";
import { env } from "@acme/config/env";
import { logger } from "@acme/telemetry";
import { db } from "@acme/database";

export const stripe = new Stripe(env.STRIPE_SECRET_KEY, {
  apiVersion: "2024-11-20.acacia",
  typescript: true,
  telemetry: false,
  maxNetworkRetries: 3,
  timeout: 30_000,
  appInfo: { name: "acme-service", version: process.env.APP_VERSION ?? "dev" },
});

export async function ensureCustomer(
  userId: string,
  email: string,
  name?: string,
): Promise<string> {
  const list = await stripe.customers.list({ email, limit: 1 });
  if (list.data[0]) {
    return list.data[0].id;
  }
  const created = await stripe.customers.create({
    email,
    name,
    metadata: { acme_user_id: userId },
  });
  logger.info({ userId, customerId: created.id }, "stripe.customer.created");
  return created.id;
}

export async function createPortalSession(customerId: string, returnUrl: string): Promise<string> {
  const session = await stripe.billingPortal.sessions.create({
    customer: customerId,
    return_url: returnUrl,
  });
  return session.url;
}

export async function createCheckoutSession(opts: {
  customerId: string;
  priceId: string;
  quantity: number;
  successUrl: string;
  cancelUrl: string;
  metadata?: Record<string, string>;
}): Promise<string> {
  const session = await stripe.checkout.sessions.create({
    customer: opts.customerId,
    mode: "subscription",
    line_items: [{ price: opts.priceId, quantity: opts.quantity }],
    success_url: opts.successUrl,
    cancel_url: opts.cancelUrl,
    metadata: opts.metadata,
    allow_promotion_codes: true,
    billing_address_collection: "required",
    tax_id_collection: { enabled: true },
    automatic_tax: { enabled: true },
  });
  if (!session.url) {
    throw new Error("checkout session created without url");
  }
  return session.url;
}

export async function reportUsage(subscriptionItemId: string, quantity: number, timestamp?: number): Promise<void> {
  await stripe.subscriptionItems.createUsageRecord(subscriptionItemId, {
    quantity,
    timestamp: timestamp ?? Math.floor(Date.now() / 1000),
    action: "increment",
  });
}
```
"""


_EXCERPT_BILLING_PAGE = """
### apps/web/app/(dashboard)/billing/page.tsx

```tsx
import { Suspense } from "react";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { stripe, ensureCustomer } from "@/lib/stripe";
import { db } from "@acme/database";
import { formatCurrency, formatDate } from "@acme/utils/format";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { getFeatureFlag } from "@acme/feature-flags";
import { BillingTierSwitcher } from "@/components/billing/tier-switcher";
import { InvoiceDownloadButton } from "@/components/billing/invoice-download";
import { UsageMetrics } from "@/components/billing/usage-metrics";

export const metadata = { title: "Billing — Acme" };
export const dynamic = "force-dynamic";

async function BillingOverview({ userId }: { userId: string }) {
  const subscription = await db.subscription.findFirst({
    where: { userId, status: { in: ["active", "past_due", "trialing"] } },
    include: { plan: true, invoices: { take: 12, orderBy: { createdAt: "desc" } } },
  });
  if (!subscription) {
    return (
      <Card>
        <CardHeader><CardTitle>No active subscription</CardTitle></CardHeader>
        <CardContent><Button asChild><a href="/pricing">Choose a plan</a></Button></CardContent>
      </Card>
    );
  }
  const customer = await ensureCustomer(userId, subscription.user.email);
  const portalFlagOn = await getFeatureFlag("stripe_portal_v2", userId);
  const nextInvoiceUpcoming = await stripe.invoices.retrieveUpcoming({ customer }).catch(() => null);
  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Current plan
            <Badge variant={subscription.status === "active" ? "default" : "destructive"}>
              {subscription.status}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-baseline justify-between">
            <div>
              <div className="text-2xl font-semibold">{subscription.plan.name}</div>
              <div className="text-sm text-muted-foreground">
                {formatCurrency(subscription.plan.monthlyPrice, subscription.plan.currency)} / month
                · {subscription.seatCount} seats
              </div>
            </div>
            <BillingTierSwitcher currentPlanId={subscription.plan.id} />
          </div>
          {nextInvoiceUpcoming && (
            <div className="rounded-md bg-muted p-4 text-sm">
              Next invoice: {formatCurrency(nextInvoiceUpcoming.amount_due, nextInvoiceUpcoming.currency.toUpperCase())}
              on {formatDate(new Date(nextInvoiceUpcoming.period_end * 1000))}
            </div>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Usage this cycle</CardTitle></CardHeader>
        <CardContent><UsageMetrics subscriptionId={subscription.id} /></CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Recent invoices</CardTitle></CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Date</TableHead>
                <TableHead>Amount</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">PDF</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {subscription.invoices.map((invoice) => (
                <TableRow key={invoice.id}>
                  <TableCell>{formatDate(invoice.createdAt)}</TableCell>
                  <TableCell>{formatCurrency(invoice.amountCents, invoice.currency)}</TableCell>
                  <TableCell>
                    <Badge variant={invoice.status === "paid" ? "default" : "destructive"}>
                      {invoice.status}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right"><InvoiceDownloadButton invoiceId={invoice.id} /></TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </>
  );
}

export default async function BillingPage() {
  const session = await auth();
  if (!session?.user?.id) redirect("/login?next=/billing");
  return (
    <div className="container space-y-6 py-6">
      <h1 className="text-2xl font-bold">Billing</h1>
      <Suspense fallback={<div>Loading billing…</div>}>
        <BillingOverview userId={session.user.id} />
      </Suspense>
    </div>
  );
}
```
"""


_EXCERPT_PRORATE_TEST = """
### packages/billing/src/prorate.test.ts

```ts
import { describe, expect, it } from "vitest";
import { prorate, annualProrate, prorateMonthly } from "./prorate";

const basePlan = { id: "pro", monthlyPrice: 9900, currency: "USD" as const };
const upgradedPlan = { id: "team", monthlyPrice: 29900, currency: "USD" as const };
const enterprisePlan = { id: "ent", monthlyPrice: 99900, currency: "USD" as const };

describe("prorate", () => {
  it("computes zero delta when switching to same plan at cycle start", () => {
    const result = prorate({
      currentPlan: basePlan, nextPlan: basePlan,
      cycleStart: new Date("2026-01-01"), cycleEnd: new Date("2026-02-01"),
      switchAt: new Date("2026-01-01"),
    });
    expect(result.netDeltaCents).toBe(0);
  });

  it("charges the differential when upgrading on day 1", () => {
    const result = prorate({
      currentPlan: basePlan, nextPlan: upgradedPlan,
      cycleStart: new Date("2026-01-01"), cycleEnd: new Date("2026-02-01"),
      switchAt: new Date("2026-01-01"),
    });
    expect(result.chargeCents).toBe(29900);
    expect(result.creditCents).toBe(9900);
    expect(result.netDeltaCents).toBe(20000);
  });

  it("prorates half cycle on mid-month upgrade", () => {
    const result = prorate({
      currentPlan: basePlan, nextPlan: upgradedPlan,
      cycleStart: new Date("2026-01-01"), cycleEnd: new Date("2026-02-01"),
      switchAt: new Date("2026-01-16"),
    });
    expect(result.creditCents).toBeCloseTo(5110, -1);
    expect(result.chargeCents).toBeCloseTo(15432, -1);
  });

  it("multiplies by seat count", () => {
    const result = prorate({
      currentPlan: basePlan, nextPlan: upgradedPlan,
      cycleStart: new Date("2026-01-01"), cycleEnd: new Date("2026-02-01"),
      switchAt: new Date("2026-01-01"),
      seatCount: 5,
    });
    expect(result.chargeCents).toBe(29900 * 5);
    expect(result.creditCents).toBe(9900 * 5);
  });

  it("handles cross-currency rejection", () => {
    expect(() => prorate({
      currentPlan: basePlan,
      nextPlan: { ...upgradedPlan, currency: "EUR" as const },
      cycleStart: new Date("2026-01-01"), cycleEnd: new Date("2026-02-01"),
      switchAt: new Date("2026-01-15"),
    })).toThrow("currency mismatch");
  });

  it("rejects switchAt outside the cycle", () => {
    expect(() => prorate({
      currentPlan: basePlan, nextPlan: upgradedPlan,
      cycleStart: new Date("2026-01-01"), cycleEnd: new Date("2026-02-01"),
      switchAt: new Date("2026-02-15"),
    })).toThrow(/effective days out of range/);
  });

  it("prorates February leap year correctly", () => {
    const result = prorate({
      currentPlan: basePlan, nextPlan: upgradedPlan,
      cycleStart: new Date("2024-02-01"), cycleEnd: new Date("2024-03-01"),
      switchAt: new Date("2024-02-15"),
    });
    expect(result.breakdown.totalDays).toBe(29);
    expect(result.breakdown.effectiveDays).toBe(15);
  });

  it("handles same-day downgrade (credit greater than charge)", () => {
    const result = prorate({
      currentPlan: enterprisePlan, nextPlan: basePlan,
      cycleStart: new Date("2026-01-01"), cycleEnd: new Date("2026-02-01"),
      switchAt: new Date("2026-01-15"),
    });
    expect(result.creditCents).toBeGreaterThan(result.chargeCents);
    expect(result.netDeltaCents).toBeLessThan(0);
  });
});

describe("annualProrate", () => {
  it("divides annual price into monthly for computation", () => {
    const annualPlan = { ...basePlan, monthlyPrice: 99_900 };
    const result = annualProrate({
      currentPlan: annualPlan, nextPlan: annualPlan,
      cycleStart: new Date("2026-01-01"), cycleEnd: new Date("2026-02-01"),
      switchAt: new Date("2026-01-01"),
    });
    expect(result.netDeltaCents).toBe(0);
  });
});

describe("prorateMonthly (convenience)", () => {
  it("derives cycle from switchAt's month", () => {
    const result = prorateMonthly(basePlan, upgradedPlan, new Date("2026-03-15"));
    expect(result.breakdown.totalDays).toBe(31);
  });
});
```
"""


_EXCERPT_WEBHOOK_ROUTE = """
### apps/web/app/api/webhooks/stripe/route.ts

```ts
import { NextRequest, NextResponse } from "next/server";
import Stripe from "stripe";
import { headers } from "next/headers";
import { stripe } from "@/lib/stripe";
import { withRateLimiter } from "@acme/rate-limiter";
import { db } from "@acme/database";
import { env } from "@acme/config/env";
import { logger } from "@acme/telemetry";
import { enqueueJob } from "@acme/queue";
import {
  handleInvoicePaid,
  handleInvoiceFailed,
  handleSubscriptionUpdated,
  handleSubscriptionDeleted,
  handleCustomerDeleted,
  handleChargeDisputeCreated,
  handlePaymentMethodAttached,
} from "@acme/webhooks/stripe";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export const POST = withRateLimiter(
  { windowMs: 60_000, max: 500, key: "stripe-webhook" },
  async (req: NextRequest) => {
    const signature = headers().get("stripe-signature");
    if (!signature) {
      return NextResponse.json({ error: "missing signature" }, { status: 400 });
    }
    const body = await req.text();
    let event: Stripe.Event;
    try {
      event = stripe.webhooks.constructEvent(body, signature, env.STRIPE_WEBHOOK_SECRET);
    } catch (err) {
      logger.warn({ err }, "stripe.webhook.signature_invalid");
      return NextResponse.json({ error: "invalid signature" }, { status: 400 });
    }
    const processed = await db.processedWebhook.findUnique({
      where: { stripeEventId: event.id },
    });
    if (processed) {
      return NextResponse.json({ ok: true, idempotent: true });
    }
    try {
      switch (event.type) {
        case "invoice.paid":                       await handleInvoicePaid(event); break;
        case "invoice.payment_failed":             await handleInvoiceFailed(event); break;
        case "customer.subscription.updated":      await handleSubscriptionUpdated(event); break;
        case "customer.subscription.deleted":      await handleSubscriptionDeleted(event); break;
        case "customer.deleted":                   await handleCustomerDeleted(event); break;
        case "charge.dispute.created":             await handleChargeDisputeCreated(event); break;
        case "payment_method.attached":            await handlePaymentMethodAttached(event); break;
        default:
          logger.debug({ type: event.type }, "stripe.webhook.unhandled_type");
      }
      await db.processedWebhook.create({
        data: { stripeEventId: event.id, type: event.type, processedAt: new Date() },
      });
      return NextResponse.json({ ok: true });
    } catch (err) {
      logger.error({ err, eventType: event.type }, "stripe.webhook.handler_failed");
      await enqueueJob("webhook-retry", { eventId: event.id, attempt: 1 });
      return NextResponse.json({ error: "handler failed; retrying" }, { status: 500 });
    }
  },
);
```
"""


_EXCERPT_SCHEMA_PRISMA = """
### packages/database/prisma/schema.prisma (extract: billing models)

```prisma
model Subscription {
  id           String   @id @default(cuid())
  userId       String
  user         User     @relation(fields: [userId], references: [id])
  planId       String
  plan         Plan     @relation(fields: [planId], references: [id])
  seatCount    Int      @default(1)
  status       SubStatus
  cycleStart   DateTime
  cycleEnd     DateTime
  stripeSubId  String?  @unique
  invoices     Invoice[]
  createdAt    DateTime @default(now())
  updatedAt    DateTime @updatedAt
  @@index([userId, status])
  @@index([stripeSubId])
  @@index([cycleEnd])
}

model Plan {
  id            String       @id
  name          String
  monthlyPrice  Int
  currency      String       @default("USD")
  features      String[]
  active        Boolean      @default(true)
  stripePriceId String?      @unique
  subscriptions Subscription[]
  createdAt     DateTime     @default(now())
  updatedAt     DateTime     @updatedAt
}

model Invoice {
  id              String       @id @default(cuid())
  subscriptionId  String
  subscription    Subscription @relation(fields: [subscriptionId], references: [id])
  amountCents     Int
  currency        String       @default("USD")
  status          InvoiceStatus
  stripeInvoiceId String?      @unique
  pdfUrl          String?
  paidAt          DateTime?
  dueDate         DateTime?
  createdAt       DateTime     @default(now())
  @@index([subscriptionId, createdAt(sort: Desc)])
  @@index([status, dueDate])
}

enum SubStatus {
  active
  past_due
  trialing
  canceled
  paused
}

enum InvoiceStatus {
  draft
  open
  paid
  void
  uncollectible
}

model ProcessedWebhook {
  stripeEventId String   @id
  type          String
  processedAt   DateTime
  @@index([type, processedAt])
}

model User {
  id            String         @id @default(cuid())
  email         String         @unique
  name          String?
  companyName   String?
  emailVerified DateTime?
  image         String?
  accounts      Account[]
  sessions      Session[]
  subscriptions Subscription[]
  createdAt     DateTime       @default(now())
  updatedAt     DateTime       @updatedAt
  @@index([email])
}

model Account {
  id                String  @id @default(cuid())
  userId            String
  user              User    @relation(fields: [userId], references: [id])
  type              String
  provider          String
  providerAccountId String
  refresh_token     String? @db.Text
  access_token      String? @db.Text
  expires_at        Int?
  token_type        String?
  scope             String?
  id_token          String? @db.Text
  session_state     String?
  @@unique([provider, providerAccountId])
  @@index([userId])
}

model Session {
  id           String   @id @default(cuid())
  sessionToken String   @unique
  userId       String
  user         User     @relation(fields: [userId], references: [id])
  expires      DateTime
  @@index([userId])
}

model ApiKey {
  id          String    @id @default(cuid())
  userId      String
  user        User      @relation(fields: [userId], references: [id])
  name        String
  keyHash     String    @unique
  keyPrefix   String
  scopes      String[]
  lastUsedAt  DateTime?
  expiresAt   DateTime?
  revokedAt   DateTime?
  createdAt   DateTime  @default(now())
  @@index([userId, revokedAt])
  @@index([keyPrefix])
}

model AuditLog {
  id         String   @id @default(cuid())
  userId     String?
  actorType  String
  action     String
  entityType String
  entityId   String
  metadata   Json?
  ipAddress  String?
  userAgent  String?
  createdAt  DateTime @default(now())
  @@index([userId, createdAt])
  @@index([entityType, entityId, createdAt])
  @@index([action, createdAt])
}
```
"""


_EXCERPT_INVOICE_GEN = """
### packages/billing/src/invoice-generator.ts

```ts
import { Decimal } from "decimal.js";
import PDFDocument from "pdfkit";
import type { Invoice, Subscription, Plan, User } from "@acme/database";
import { formatCurrency, formatDate } from "@acme/utils/format";
import { s3 } from "@acme/storage";
import { logger } from "@acme/telemetry";
import { env } from "@acme/config/env";

interface InvoiceContext {
  invoice: Invoice;
  subscription: Subscription;
  plan: Plan;
  user: User;
  lineItems: Array<{
    description: string;
    unitPriceCents: number;
    quantity: number;
    subtotalCents: number;
  }>;
  taxCents: number;
  totalCents: number;
}

const PAGE_MARGIN = 50;
const LINE_HEIGHT = 18;
const BRAND_COLOR = "#1a2938";
const MUTED_COLOR = "#6b7280";

export async function generateInvoicePDF(ctx: InvoiceContext): Promise<string> {
  const doc = new PDFDocument({
    size: "A4",
    margin: PAGE_MARGIN,
    info: {
      Title: `Invoice ${ctx.invoice.id}`,
      Author: "Acme Inc.",
      Subject: `Invoice for ${ctx.user.email}`,
      CreationDate: ctx.invoice.createdAt,
    },
  });
  const chunks: Buffer[] = [];
  doc.on("data", (chunk) => chunks.push(chunk));
  const done = new Promise<Buffer>((resolve, reject) => {
    doc.on("end", () => resolve(Buffer.concat(chunks)));
    doc.on("error", reject);
  });

  doc.fillColor(BRAND_COLOR).fontSize(24).text("Acme Inc.", PAGE_MARGIN, PAGE_MARGIN);
  doc.fillColor(MUTED_COLOR).fontSize(9).text(
    "123 Example Street\\nSan Francisco, CA 94103\\nUSA\\ninvoices@acme.io\\n+1 (555) 010-0100",
    PAGE_MARGIN, PAGE_MARGIN + 30,
  );
  doc.fillColor(BRAND_COLOR).fontSize(18).text("INVOICE", 400, PAGE_MARGIN,
    { align: "right", width: doc.page.width - 400 - PAGE_MARGIN });
  doc.fillColor(MUTED_COLOR).fontSize(9).text(
    `Invoice: ${ctx.invoice.id}\\nDate issued: ${formatDate(ctx.invoice.createdAt)}\\nDue: on receipt`,
    400, PAGE_MARGIN + 24,
    { align: "right", width: doc.page.width - 400 - PAGE_MARGIN });

  const billToY = PAGE_MARGIN + 130;
  doc.fillColor(BRAND_COLOR).fontSize(10).text("BILL TO", PAGE_MARGIN, billToY);
  doc.fillColor("#111827").fontSize(11).text(
    `${ctx.user.name ?? "(name not set)"}\\n${ctx.user.email}\\n${ctx.user.companyName ? ctx.user.companyName + "\\n" : ""}`,
    PAGE_MARGIN, billToY + 16,
  );

  const tableY = billToY + 100;
  const colX = { desc: PAGE_MARGIN, qty: 300, unit: 360, total: 460 };
  doc.fillColor(BRAND_COLOR).fontSize(9);
  doc.text("DESCRIPTION", colX.desc, tableY);
  doc.text("QTY", colX.qty, tableY, { width: 50, align: "right" });
  doc.text("UNIT PRICE", colX.unit, tableY, { width: 80, align: "right" });
  doc.text("AMOUNT", colX.total, tableY, { width: 80, align: "right" });
  doc.moveTo(PAGE_MARGIN, tableY + 14).lineTo(doc.page.width - PAGE_MARGIN, tableY + 14).stroke();

  let rowY = tableY + 22;
  for (const item of ctx.lineItems) {
    doc.fillColor("#111827").fontSize(10);
    doc.text(item.description, colX.desc, rowY, { width: 240 });
    doc.text(String(item.quantity), colX.qty, rowY, { width: 50, align: "right" });
    doc.text(formatCurrency(item.unitPriceCents, ctx.invoice.currency), colX.unit, rowY, { width: 80, align: "right" });
    doc.text(formatCurrency(item.subtotalCents, ctx.invoice.currency), colX.total, rowY, { width: 80, align: "right" });
    rowY += LINE_HEIGHT;
  }

  rowY += 10;
  doc.moveTo(colX.unit, rowY).lineTo(doc.page.width - PAGE_MARGIN, rowY).stroke();
  rowY += 8;
  doc.fillColor(MUTED_COLOR).fontSize(10).text("Subtotal", colX.unit, rowY, { width: 80, align: "right" });
  const subtotalCents = ctx.lineItems.reduce((sum, li) => sum + li.subtotalCents, 0);
  doc.fillColor("#111827").text(formatCurrency(subtotalCents, ctx.invoice.currency), colX.total, rowY, { width: 80, align: "right" });
  rowY += LINE_HEIGHT;
  if (ctx.taxCents > 0) {
    doc.fillColor(MUTED_COLOR).text("Tax", colX.unit, rowY, { width: 80, align: "right" });
    doc.fillColor("#111827").text(formatCurrency(ctx.taxCents, ctx.invoice.currency), colX.total, rowY, { width: 80, align: "right" });
    rowY += LINE_HEIGHT;
  }
  doc.fillColor(BRAND_COLOR).fontSize(12).text("TOTAL", colX.unit, rowY, { width: 80, align: "right" });
  doc.fillColor(BRAND_COLOR).text(formatCurrency(ctx.totalCents, ctx.invoice.currency), colX.total, rowY, { width: 80, align: "right" });
  rowY += LINE_HEIGHT + 30;

  doc.fillColor(MUTED_COLOR).fontSize(9).text(
    "Thank you for your business.\\nPayment was automatically charged to the card on file.\\nQuestions: billing@acme.io",
    PAGE_MARGIN, rowY,
  );

  doc.end();
  const buffer = await done;
  const key = `invoices/${ctx.user.id}/${ctx.invoice.id}.pdf`;
  await s3.putObject({
    Bucket: env.S3_BUCKET, Key: key, Body: buffer, ContentType: "application/pdf",
    Metadata: { "acme-invoice-id": ctx.invoice.id, "acme-user-id": ctx.user.id },
  });
  const pdfUrl = `https://${env.S3_BUCKET}.s3.amazonaws.com/${key}`;
  logger.info({ invoiceId: ctx.invoice.id, userId: ctx.user.id, pdfUrl }, "invoice.pdf.generated");
  return pdfUrl;
}
```
"""


# ── Tool schemas — three tiers of increasing tool count ─────────────────


SMALL_TOOLS: list[dict] = []

_BASIC_TOOLS: list[dict] = [
    _tool("read_file",
        "Read a file on disk. Returns the full text content. Fails if the "
        "file does not exist. Prefer this over guessing at file contents.",
        {"path": _string_param("Absolute or workspace-relative path."),
         "start_line": _integer_param("Optional 1-indexed start line."),
         "end_line": _integer_param("Optional 1-indexed end line (inclusive).")}),
    _tool("write_file",
        "Write text to a file on disk. Overwrites if it exists. Creates "
        "parent directories as needed.",
        {"path": _string_param("Absolute or workspace-relative path."),
         "content": _string_param("Full file content.")}),
    _tool("run_shell",
        "Execute a shell command in the workspace. Returns stdout, stderr, "
        "exit code. Never use for destructive commands.",
        {"command": _string_param("The shell command to run."),
         "cwd": _string_param("Working directory. Defaults to workspace root."),
         "timeout_s": _integer_param("Timeout in seconds. Default 60, max 600.")}),
]

_CODENAV_TOOLS: list[dict] = [
    _tool("search_codebase",
        "Full-text and regex search across the workspace. Returns file paths "
        "and line numbers. Respects .gitignore.",
        {"query": _string_param("Search pattern. Regex or exact."),
         "path_glob": _string_param("Optional path glob to restrict search."),
         "case_sensitive": _boolean_param("Default false."),
         "max_results": _integer_param("Cap on result count. Default 50.")}),
    _tool("find_references",
        "Find all references to a symbol in the workspace. LSP-backed.",
        {"symbol": _string_param("Symbol name."),
         "path": _string_param("File path containing the definition."),
         "line": _integer_param("Line number of the definition.")}),
    _tool("run_tests",
        "Execute the project's test suite, optionally scoped.",
        {"scope": _string_param("Path, file, or test-name pattern. Empty = all."),
         "watch": _boolean_param("Re-run on file change. Default false."),
         "updateSnapshots": _boolean_param("Update snapshots. Default false.")}),
    _tool("git_diff",
        "Show unified diff for a commit, branch, or file.",
        {"ref": _string_param("Commit hash, branch, or HEAD."),
         "path": _string_param("Optional file path to restrict the diff.")}),
    _tool("apply_diff",
        "Apply a unified-diff patch. Validates before applying.",
        {"patch": _string_param("Unified diff text."),
         "check_only": _boolean_param("Validate only. Default false.")}),
    _tool("git_blame",
        "Show git blame for a file or range.",
        {"path": _string_param("File path."),
         "line_start": _integer_param("Optional start line."),
         "line_end": _integer_param("Optional end line.")}),
    _tool("lsp_hover",
        "Get LSP hover info (type, signature, docs) at a position.",
        {"path": _string_param("File path."),
         "line": _integer_param("1-indexed line."),
         "column": _integer_param("1-indexed column.")}),
    _tool("lsp_goto_definition",
        "Jump to the definition of the symbol at a position.",
        {"path": _string_param("File path."),
         "line": _integer_param("1-indexed line."),
         "column": _integer_param("1-indexed column.")}),
    _tool("lsp_rename",
        "Rename a symbol workspace-wide via LSP.",
        {"path": _string_param("File path."),
         "line": _integer_param("1-indexed line."),
         "column": _integer_param("1-indexed column."),
         "new_name": _string_param("New symbol name.")}),
]

_AUTONOMOUS_TOOLS: list[dict] = [
    _tool("browser_screenshot",
        "Take a screenshot of the running dev server in headless Chromium. "
        "Returns base64 PNG plus rendered DOM.",
        {"url": _string_param("URL or path relative to localhost:3000."),
         "viewport": _string_param("Viewport size: desktop, tablet, mobile."),
         "wait_for_selector": _string_param("CSS selector to wait for.")}),
    _tool("database_query",
        "Run a read-only SQL query against the dev database. SELECT/WITH only.",
        {"sql": _string_param("The SQL query."),
         "params": _array_param("Positional parameters for $1, $2, …", "string"),
         "limit": _integer_param("Row cap. Default 100, max 10000.")}),
    _tool("task_create",
        "Add a task to the current plan.",
        {"title": _string_param("Short imperative title."),
         "description": _string_param("What needs to be done and why."),
         "depends_on": _array_param("IDs of blocking tasks.", "string")}),
    _tool("task_update",
        "Update a task's status or content.",
        {"task_id": _string_param("The task identifier."),
         "status": _string_param("pending | in_progress | completed | blocked | cancelled."),
         "notes": _string_param("Optional progress notes.")}),
    _tool("task_list",
        "List tasks in the current plan, optionally filtered by status.",
        {"status": _string_param("Optional status filter.")}),
    _tool("send_slack",
        "Post a message to the team's Slack channel. Use sparingly.",
        {"channel": _string_param("Channel name (e.g. #engineering)."),
         "message": _string_param("Message body, markdown supported."),
         "thread_ts": _string_param("Optional thread timestamp.")}),
    _tool("github_issue_create",
        "Create a GitHub issue in the current repo.",
        {"title": _string_param("Issue title, imperative mood."),
         "body": _string_param("Markdown body. Include repro for bugs."),
         "labels": _array_param("Label names.", "string"),
         "assignees": _array_param("GitHub usernames.", "string")}),
    _tool("github_pr_create",
        "Create a GitHub pull request from the current branch.",
        {"title": _string_param("PR title."),
         "body": _string_param("PR body, markdown."),
         "base": _string_param("Base branch. Default main."),
         "draft": _boolean_param("Mark as draft. Default false.")}),
    _tool("github_pr_comment",
        "Post a review comment on a pull request.",
        {"pr_number": _integer_param("PR number."),
         "body": _string_param("Comment body."),
         "path": _string_param("Optional file path for inline comment."),
         "line": _integer_param("Optional line number.")}),
    _tool("linear_issue_create",
        "Create a Linear issue.",
        {"title": _string_param("Issue title."),
         "description": _string_param("Issue body."),
         "team": _string_param("Team identifier or name."),
         "priority": _integer_param("0 = none, 1 = urgent, 2 = high, 3 = medium, 4 = low.")}),
    _tool("linear_issue_update",
        "Update a Linear issue's status, assignee, or labels.",
        {"issue_id": _string_param("Linear issue ID."),
         "state": _string_param("Optional new state."),
         "assignee": _string_param("Optional new assignee."),
         "labels": _array_param("Optional new labels.", "string")}),
    _tool("ci_trigger",
        "Trigger a CI workflow run on the current branch.",
        {"workflow": _string_param("Workflow filename or name."),
         "ref": _string_param("Git ref. Default HEAD."),
         "inputs": _string_param("JSON-encoded workflow inputs.")}),
    _tool("log_search",
        "Search production logs via the telemetry backend.",
        {"query": _string_param("Query in the log-search DSL."),
         "from": _string_param("Start time (ISO 8601 or relative like '1h ago')."),
         "to": _string_param("End time."),
         "limit": _integer_param("Max results. Default 100.")}),
    _tool("metric_query",
        "Query a time-series metric from the telemetry backend.",
        {"metric": _string_param("Metric name."),
         "from": _string_param("Start time."),
         "to": _string_param("End time."),
         "agg": _string_param("Aggregation: avg, sum, p50, p95, p99.")}),
    _tool("feature_flag_update",
        "Change a feature flag's rollout percentage.",
        {"flag": _string_param("Flag identifier."),
         "percentage": _integer_param("0-100."),
         "environments": _array_param("Which environments to update.", "string")}),
    _tool("agent_delegate",
        "Delegate a bounded subtask to an agent. Provides isolated context.",
        {"prompt": _string_param("Task description for the subagent."),
         "max_turns": _integer_param("Budget for subagent turns. Default 10."),
         "return_format": _string_param("markdown | json | text.")}),
]


# ── Fixture assembly ────────────────────────────────────────────────────


_ALL_EXCERPTS = [
    _EXCERPT_BILLING_PAGE,
    _EXCERPT_PRORATE_TS,
    _EXCERPT_PRORATE_TEST,
    _EXCERPT_STRIPE_TS,
    _EXCERPT_WEBHOOK_ROUTE,
    _EXCERPT_SCHEMA_PRISMA,
    _EXCERPT_INVOICE_GEN,
]


def _assemble(target_tokens: int, base_parts: list[str]) -> str:
    """Build a system prompt of approximately ``target_tokens`` by padding
    ``base_parts`` with repeated file excerpts until the chars//4
    estimate clears the target.

    Real agents inline the same files multiple times during a session
    (planner re-reads, task_update writes, tool-call responses repeat
    the content). Simulating that directly is the fastest way to a
    realistic size without maintaining enormous literals by hand.
    """
    content = "".join(base_parts)
    # Chars/4 ≈ tokens. Pad in excerpt-sized chunks until we cross the
    # target, then trim by whole excerpts so each fixture ends on a
    # clean file boundary.
    idx = 0
    while len(content) // 4 < target_tokens:
        content += "\n\n## Additional embedded file (re-read during session)\n"
        content += _ALL_EXCERPTS[idx % len(_ALL_EXCERPTS)]
        idx += 1
    return content


SMALL_SYSTEM_PROMPT = """\
You are a helpful command-line assistant. Keep answers concise and
direct. When the user asks for a command, give the command first,
then a one-line explanation. Do not preface answers with "Sure!" or
"Of course!" — get to the point. Stay under ten lines unless the user
asks for more detail.
"""

# Target sizes are calibrated to measured turn-15 tokens (with history):
# small ~3K, medium ~20K, large ~80K, xlarge ~200K. The base padding
# targets below subtract the expected per-turn contribution of tools
# (serialized JSON) and growing history so the final numbers land close.

MEDIUM_SYSTEM_PROMPT = _assemble(
    target_tokens=11_000,
    base_parts=[
        _CORE_SYSTEM_PROMPT,
        _WORKSPACE_SNAPSHOT,
        "\n## Currently-relevant file excerpts\n",
        _EXCERPT_BILLING_PAGE,
        _EXCERPT_PRORATE_TS,
        _EXCERPT_STRIPE_TS,
    ],
)

LARGE_SYSTEM_PROMPT = _assemble(
    target_tokens=60_000,
    base_parts=[
        _CORE_SYSTEM_PROMPT,
        _AUTONOMOUS_EXTENSION,
        _WORKSPACE_SNAPSHOT,
        "\n## Currently-relevant file excerpts\n",
    ],
)

XLARGE_SYSTEM_PROMPT = _assemble(
    target_tokens=170_000,
    base_parts=[
        _CORE_SYSTEM_PROMPT,
        _AUTONOMOUS_EXTENSION,
        _WORKSPACE_SNAPSHOT,
        "\n## Currently-relevant file excerpts\n",
    ],
)

MEDIUM_TOOLS = _BASIC_TOOLS + _CODENAV_TOOLS
LARGE_TOOLS = _BASIC_TOOLS + _CODENAV_TOOLS + _AUTONOMOUS_TOOLS[:8]
XLARGE_TOOLS = _BASIC_TOOLS + _CODENAV_TOOLS + _AUTONOMOUS_TOOLS


# ── Fixture registry ────────────────────────────────────────────────────


def _make_builder(system_prompt: str, tools: list[dict]) -> Callable:
    def _build(
        user_message: str,
        model: str,
        history: list[dict] | None = None,
        stream: bool = False,
    ) -> dict:
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        return payload
    return _build


_FIXTURES: dict[str, tuple[str, list[dict], str]] = {
    "small":  (SMALL_SYSTEM_PROMPT,  SMALL_TOOLS,
               "Light agent — LangChain loop / CLI assistant"),
    "medium": (MEDIUM_SYSTEM_PROMPT, MEDIUM_TOOLS,
               "Cursor / Cline — mid-task with workspace context"),
    "large":  (LARGE_SYSTEM_PROMPT,  LARGE_TOOLS,
               "Claude Code — session 1–2h in, many open files"),
    "xlarge": (XLARGE_SYSTEM_PROMPT, XLARGE_TOOLS,
               "Claude Code / Devin — long autonomous run, 1M-context territory"),
}


def fixture_names() -> list[str]:
    return ["small", "medium", "large", "xlarge"]


def fixture_description(name: str) -> str:
    if name not in _FIXTURES:
        raise KeyError(f"unknown fixture: {name}")
    return _FIXTURES[name][2]


def fixture_approx_tokens(name: str) -> int:
    """Approximate baseline token count for a fixture (system + tools only).

    Chars//4 estimate over the system prompt plus the serialised tool
    schemas. Does not include history or per-turn user message — those
    get added at request time.
    """
    if name not in _FIXTURES:
        raise KeyError(f"unknown fixture: {name}")
    sys_prompt, tools, _ = _FIXTURES[name]
    return (len(sys_prompt) + len(json.dumps(tools))) // 4


def fixture_for(name: str) -> Callable:
    """Return the build_agent_payload callable for the named fixture."""
    if name not in _FIXTURES:
        raise KeyError(
            f"unknown fixture {name!r}; valid: {', '.join(fixture_names())}"
        )
    sys_prompt, tools, _ = _FIXTURES[name]
    return _make_builder(sys_prompt, tools)


# Backwards-compat: existing code paths that import AGENT_SYSTEM_PROMPT /
# AGENT_TOOLS / build_agent_payload get the medium fixture. New code
# should call ``fixture_for`` with an explicit size.
AGENT_SYSTEM_PROMPT = MEDIUM_SYSTEM_PROMPT
AGENT_TOOLS = MEDIUM_TOOLS
build_agent_payload = _make_builder(MEDIUM_SYSTEM_PROMPT, MEDIUM_TOOLS)
