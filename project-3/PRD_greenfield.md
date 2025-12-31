# BMAD Demo NodeJS Project - Product Requirements Document

**Author:** Aditya
**Date:** 2025-11-03
**Version:** 1.0

---

## Executive Summary

NYU Admissions managers rely on the Device Inventory Google Sheet to decide which staff members should receive or return hardware. This PRD defines a demo-ready web experience that ingests the Devices sheet into MongoDB, renders a responsive grid tailored to managerial workflows, and keeps the view current without manual spreadsheet wrangling.

### What Makes This Special

Delivers a snappy, polished spreadsheet UI that loads instantly on first visit, letting admissions managers trust the live roster the moment the page opens and impressing stakeholders with a premium alternative to raw Google Sheets.

---

## Project Classification

- **Product Type:** Internal web application
- **Primary Users:** NYU Admissions managers (Google Workspace accounts)
- **Core Workflow:** Monitor device ownership, status, and readiness for handoff
- **Technical Stack:** Node.js service hosted on Google App Engine, MongoDB Atlas primary store, Google Sheets (Devices tab) as source of truth / backup
- **Project Complexity:** Level 2 demo (single-team, tightly scoped)

### Complexity Notes

The engagement remains a Level 2 demonstration: narrow scope, limited stakeholders, and no production scale expectations. All work is sized for a single squad validating feasibility rather than preparing campus-wide rollout.

### Domain Context

NYU admissions managers operate under institutional policies that tie device custody to sensitive student and staffing workflows. Even in demo form, the experience must reflect correct enforcement of access controls so only authenticated managers can review hardware records, and must demonstrate audit-friendly trails for transfers or deactivations.

---

## Success Criteria

- End-to-end sync: Any edit to the Devices Google Sheet appears in the web UI within 60 seconds during live demos.
- First impression: Landing page renders authenticated manager view in under 2 seconds even on shared campus Wi-Fi.
- Interaction quality: Sorting, filtering, refresh, and anonymization interactions respond within 200 ms so the experience feels like a native spreadsheet.
- Governance confidence: Audit log captures every sync attempt and anonymization toggle change with timestamps to demonstrate accountable access.

### Business Metrics

- Time-to-update (sheet edit → UI refresh) ≤ 60 seconds during demo walkthrough
- UI interaction latency (sort/filter/apply anonymization) ≤ 200 ms client-observed
- Successful login conversion ≥ 95% for invited NYU manager accounts during demo prep sessions
- Sync reliability: ≥ 98% of scheduled 2-minute updates complete without manual retry over demo window

---

## Product Scope

### MVP - Minimum Viable Product

1. Manager-only access: Google OAuth restricted to `@nyu.edu` accounts on the admissions managers list, with failed attempts logged.
2. Automated ingest: On-demand refresh button plus 2-minute scheduled sync pulling the Devices sheet into MongoDB with schema normalization.
3. Inventory grid: Responsive web UI showing device, assignee, status, condition, and last-seen timestamp with sort/filter controls.
4. Governance signals: Sync status banner, anonymization toggle for sensitive columns, and audit log view of recent refreshes/toggles.

### Growth Features (Post-MVP)

1. Bidirectional updates pushing staged MongoDB edits back to Google Sheets after manager approval.
2. Role-based experiences separating demo operators, managers, and read-only observers with customizable permissions.
3. Multi-sheet orchestration supporting additional admissions datasets (e.g., Departures, Accessories) with per-tab pipelines.
4. Health dashboard aggregating sync performance metrics and surfacing proactive alerts when Sheets throttling occurs.

### Vision (Future)

1. Self-service onboarding wizard that lets departments connect new sheets and map columns without engineering support.
2. AI-assisted anomaly detection to flag device condition changes, overdue returns, or conflicting ownership records.
3. Federated data connectors bringing in ServiceNow or asset-tracking feeds alongside Google Sheets.
4. Collaborative comments and approval workflows directly inside the grid with optimistic updates and notifications.

---

## Domain-Specific Requirements

- Uphold NYU information-access standards: restrict device inventory visibility to authenticated admissions managers, log unauthorized attempts, and surface audit-friendly cues for leadership review.
- Reflect staffing workflows: display assignee department and offboarding status to support timely collection decisions when employees depart.
- Preserve historical context: retain last-known device condition and transfer notes so governance teams can reconstruct decisions post-demo.
- Handle FERPA-adjacent sensitivity: ensure any fields that could reveal student interactions (e.g., assigned admissions programs) are masked or omitted during demo playback.

This section shapes all functional and non-functional requirements below.

---

## Innovation & Novel Patterns

None identified for this demo iteration. Focus remains on disciplined execution of the admissions device workflow.

---

## Web app (Node.js + MongoDB + Google Sheets sync) Specific Requirements

Need Node.js backend service to authenticate with Google Sheets API, transform rows for MongoDB persistence, and expose data to front-end via REST/GraphQL; front-end must render spreadsheet grid with inline edit affordances and reflect sync status in real time.

### API Specification

| Endpoint | Method | Purpose | Auth | Key Params | Response Highlights |
| --- | --- | --- | --- | --- | --- |
| `/api/devices` | GET | Serve paginated device roster for grid render | Manager session | `page`, `pageSize`, `sort`, `filters`, `anonymized` | JSON list with device metadata, assignee info, condition, lastSyncedAt |
| `/api/sync` | POST | Trigger on-demand ingest from Google Sheets into MongoDB | Manager session | None | Summary object: `rowsProcessed`, `durationMs`, `status`, `errorCode` (if any) |
| `/api/audit` | GET | Display recent sync/anonymization events | Manager session with lead role | `limit` (default 20) | Chronological events with actor, action, timestamp, outcome |
| `/internal/sync` | POST | Scheduler entry point for 2-minute cron jobs | Service token (App Engine) | None | Text `ok` with runId; logs errors with stack trace reference |

- Error responses follow `{ errorCode, message, details }` contract to simplify demo narration.
- Upstream Sheets errors translated into actionable codes (e.g., `SHEET_NOT_FOUND`, `SHEETS_RATE_LIMIT`, `OAUTH_REVOKED`).

### Authentication & Authorization

- Google OAuth restricted to `@nyu.edu` accounts; admissions manager allowlist stored in configuration collection.
- Successful login sets secure HTTP-only cookie containing signed JWT (1-hour expiry, sliding refresh on activity).
- Middleware enforces session validity on every API request; failures return `401` and append audit entry.
- Lead managers receive elevated claim enabling access to `/api/audit` and future configuration endpoints.

### Platform Support

- Optimized for modern desktop browsers (Chrome ≥ 120, Edge ≥ 120, Safari ≥ 17) at 1280×800 and 1440×900 resolutions.
- Tablet fallback (iPad Safari) delivers read-only view; mobile portrait blocked with explanatory message citing demo constraints.
- Supports dark-mode preference but defaults to NYU-aligned light theme for presentation clarity.

### Device Capabilities

- Requires network latency <150 ms to keep refresh interactions inside the 200 ms goal during demos.
- Browser must support WebSockets and EventSource to enable future live updates; polyfills bundled for legacy support if needed.
- Clipboard access permission requested only when user copies device rows; no additional hardware integrations.

### Multi-Tenancy Architecture

- Single-tenant configuration bound to the Admissions Devices sheet and dedicated MongoDB database.
- Tenant-aware abstractions deferred; configuration UI fields locked to one namespace throughout demo.

### Permissions & Roles

- **Admissions Manager (default):** Access roster UI, trigger sync, toggle anonymization, view audit feed.
- **Lead Manager (subset):** All manager abilities plus access to audit API and configuration preview.
- **Demo Operator (future growth):** Planned role for managing sheet mappings and schedule toggles; out of MVP scope.
- All unauthorized attempts—whether non-`@nyu.edu` domains or unlisted managers—are logged with timestamp and source IP.


---

## User Experience Principles

- Professional, data-forward aesthetic that mirrors NYU brand while signaling modern tooling (e.g., deep purple accents, clear typography).
- Calm-state defaults with high-contrast cues for sync status, anonymization mode, and pending actions so managers instantly know whether the roster is trustworthy.
- Minimal cognitive load: surface only device attributes managers act on during demos (device, assignee, status, condition, last seen).

### Key Interactions

- Landing experience: auto-authenticated managers see the latest roster with a hero banner confirming last sync time.
- Grid interaction: inline sort, filter, column show/hide, and anonymization toggle with optimistic updates to maintain the wow-factor responsiveness.
- Governance lane: dedicated panel showing the last five sync jobs and anonymization state changes with timestamps and operator identity.

---

## Functional Requirements

- **FR-001 Access Control:** Authenticate NYU admissions managers via Google OAuth and deny all other domains.  
  *Acceptance:* Valid manager reaches dashboard post-login; non-whitelisted user receives 403 and event logged with email + timestamp.
- **FR-002 Whitelist Management:** Provide configuration (file or admin page) listing approved manager emails for demo use.  
  *Acceptance:* Updating allowlist reflects on next login attempt without redeploy; audit log records change reference.
- **FR-003 Sheet Registration:** Allow operators to set the target Devices sheet ID and MongoDB collection name prior to demo.  
  *Acceptance:* Configuration persisted in MongoDB `config` collection; `/api/sync` reads values without code change.
- **FR-004 Scheduled Sync:** Run ingest automatically every 2 minutes while demo mode is active.  
  *Acceptance:* Cron job triggers `/internal/sync`; successful runs recorded with duration, rows processed, anonymization state.
- **FR-005 Manual Refresh:** Surface “Refresh Now” control that triggers immediate ingest and provides optimistic status feedback.  
  *Acceptance:* Button call completes within 5 seconds, updates status banner, logs event in audit feed.
- **FR-006 Data Transformation:** Normalize Google Sheets rows into MongoDB documents with consistent typing (dates, enums) and latest change timestamp.  
  *Acceptance:* Sample Sheet change appears as updated document with `lastSyncedAt` and normalized values.
- **FR-007 Device Grid UI:** Present sortable, filterable grid showing device ID, assignee name/email, status, condition, and last sync time.  
  *Acceptance:* Sorting and filtering apply instantly; empty states and error states handled gracefully.
- **FR-008 Sync Status Banner:** Display last successful sync time, running state, and any error codes prominently.  
  *Acceptance:* Banner changes color/state for success (green), running (blue), failure (red) with actionable message.
- **FR-009 Anonymization Toggle:** Allow managers to mask/unmask sensitive columns with deterministic placeholders, reflecting state across grid and logs.  
  *Acceptance:* Toggling updates grid within 200 ms, records event with actor, and reuses same placeholders per device.
- **FR-010 Audit Trail:** Provide panel/API listing the most recent sync runs and anonymization toggles with actor, action, status, and timestamp.  
  *Acceptance:* Managers can view last 20 events; entries persist during demo to demonstrate governance readiness.
- **FR-011 Governance Cues:** Highlight devices flagged for offboarding (if marked in sheet) and pending actions for managers.  
  *Acceptance:* Sheet field `offboardingStatus` surfaces indicator chip in grid and can be filtered.
- **FR-012 Error Handling:** Surface friendly errors when Sheets API is unreachable or credentials expire, with guidance to retry or update config.  
  *Acceptance:* Simulated Sheets outage shows descriptive banner, retains last good data, and logs failure with `errorCode`.

---

## Non-Functional Requirements

### Performance

- First paint and authenticated render under 2 seconds on App Engine standard tier using cached MongoDB read.
- Grid interactions (sort/filter/toggle) under 200 ms measured via performance instrumentation.
- Scheduled sync window completes within 20 seconds for 1,000-row sheet; failures raise alert in audit log.

### Security

- Enforce Google OAuth for `@nyu.edu`; deny other domains at identity provider level when possible.
- Store credentials and secrets (Sheets API, Mongo connection string) in Google Secret Manager; never commit to source.
- Log all authentication failures, sync requests, and anonymization toggles with actor details.
- Ensure anonymized view omits or masks any fields flagged as sensitive in configuration.

### Scalability

- Demo optimized for single-department dataset (~1k rows).  
- MongoDB indexes on `deviceId`, `assignedTo`, and `lastSyncedAt` keep queries performant if dataset doubles during evaluation.  
- Cron schedule configurable but defaults to 2 minutes to avoid Sheets API quota exhaustion.

### Accessibility

- WCAG 2.1 AA color contrast for status indicators and text.  
- Keyboard support for grid navigation (arrow keys, tab focus).  
- Screen reader labels for anonymization toggle, refresh button, and sync status banner.

### Integration

- Google Sheets API (read-only) using service credentials; network errors retried with exponential backoff (3 attempts).  
- MongoDB Atlas cluster configured with IP allowlist from App Engine.  
- Optional webhook to Slack/Email (future growth) remains stubbed but documented for operations handoff.

---

## Implementation Planning

### Epic Breakdown Required

Requirements must be decomposed into epics and bite-sized stories (200k context limit).

**Next Step:** Run `workflow epics-stories` to create the implementation breakdown.

---

## PRD Summary

- Vision: Deliver a premium, manager-only device dashboard that proves the value of BMAD orchestrations by syncing the Devices sheet into a responsive UI.  
- Success: Hit sub-60-second data refreshes, sub-2-second authenticated loads, sub-200 ms interactions, and maintain a complete governance log.  
- Scope: 12 functional requirements and 5 non-functional requirement categories covering security, performance, accessibility, scale, and integrations.  
- Key Considerations: Admissions governance, sensitive data masking, App Engine hosting, MongoDB as operational store with Google Sheets as authoritative source.

## Project Level & Target Scale

- **Project Level:** Level 2 (demo-focused, single squad)  
- **Target Scale:** NYU Admissions manager demo with one Google Sheet source and one MongoDB namespace

## Epic Details

Existing epic plan (see `docs/epics.md`) covers:
- **Epic 1 – Foundation & Access Control:** OAuth lock-down, session middleware, Mongo setup.  
- **Epic 2 – Sheets Ingest Pipeline:** Scheduled + manual sync, data normalization, anonymization layer.  
- **Epic 3 – Spreadsheet UI Experience:** Grid interactions, status messaging, anonymization toggle UX.  
- **Epic 4 – Operational Guardrails:** Admin settings, observability, rollback scripts, demo playbook.

---

## References

- Product Brief: None (context captured through stakeholder discovery in-session)
- Domain Brief: None (leveraging institutional knowledge shared in-session)
- Research: Google Sheets API setup guides; MongoDB Atlas documentation

---

## Next Steps

1. **Epic & Story Breakdown** – Run: `workflow epics-stories`
2. **UX Design** – Run: `workflow ux-design`
3. **Architecture** – Run: `workflow create-architecture`

---

_This PRD captures the essence of the BMAD Demo NodeJS Project — a seamless, manager-focused device roster experience._

_Created through collaborative discovery between Aditya and the AI facilitator._
