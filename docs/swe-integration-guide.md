# PesaFlow ML API — Integration Guide for Backend SWE

## Base URL
```
Production:  https://ml.pesaflow.ai
Local dev:   http://localhost:18743
```

## Authentication
All scoring and label endpoints require a JWT token with the appropriate role. Pass it as:
```
Authorization: Bearer <token>
```
Roles: `ML_ADMIN`, `AML_AUDITOR`, `FRAUD_ANALYST`

---

## 1. Fraud Scoring

Call this when a payment transaction is created or updated.

```
POST /api/v1/ml/fraud/score
```

```json
{
  "transaction_id": "uuid",
  "user_id": "uuid",
  "amount": 5000.00,
  "currency": "KES",
  "transaction_type": "PAYMENT",
  "channel": "MOBILE_APP",
  "merchant_id": "uuid",
  "geo_location": { "lat": -1.2921, "lng": 36.8219 },
  "device_fingerprint": "fp_abc123",
  "ip_address": "196.201.214.100",
  "timestamp": "2026-02-27T10:30:00Z"
}
```

**Response** — Use `risk_level` and `recommended_action` for your UI:
```json
{
  "prediction_id": "uuid",
  "risk_score": 0.32,
  "risk_level": "LOW",
  "recommended_action": "APPROVE",
  "factors": [ { "feature": "amount", "contribution": 0.15, "description": "..." } ],
  "model_metadata": { "scoring_mode": "heuristic", "maturity": "COLD" }
}
```

| risk_level | recommended_action | What to do |
|---|---|---|
| LOW | APPROVE | Auto-approve transaction |
| MEDIUM | REVIEW | Flag for manual review |
| HIGH | BLOCK | Block and create fraud review case |
| CRITICAL | BLOCK | Block immediately, alert fraud team |

**Batch scoring** (up to 100 transactions):
```
POST /api/v1/ml/fraud/score/batch
{ "transactions": [ ...same schema as single... ] }
```

---

## 2. AML Scoring

Call this for wire transfers, large payments, or any transaction that needs AML screening.

```
POST /api/v1/aml/score
```

```json
{
  "transaction_id": "uuid",
  "user_id": "uuid",
  "amount": 500000.00,
  "currency": "KES",
  "transaction_type": "WIRE",
  "channel": "BANK_TRANSFER",
  "sender_account": "1234567890",
  "receiver_account": "0987654321",
  "sender_bank": "KCB",
  "receiver_bank": "Equity",
  "timestamp": "2026-02-27T10:30:00Z"
}
```

**Response:**
```json
{
  "prediction_id": "uuid",
  "risk_score": 0.55,
  "risk_level": "MEDIUM",
  "recommended_action": "MONITOR",
  "factors": [...],
  "model_metadata": { "scoring_mode": "heuristic", "maturity": "COLD" }
}
```

| recommended_action | What to do |
|---|---|
| APPROVE | Proceed normally |
| MONITOR | Log for AML review queue |
| ESCALATE | Create AML case for compliance team |
| BLOCK | Block transaction, file STR |

---

## 3. Merchant Risk Scoring

Call this on merchant onboarding and periodically (daily/weekly) for active merchants.

```
POST /api/v1/ml/merchant/score
```

```json
{
  "merchant_id": "uuid",
  "amount": 10000.00,
  "currency": "KES",
  "transaction_type": "CARD_PAYMENT",
  "channel": "POS",
  "mcc_code": "5411",
  "business_type": "retail",
  "registration_date": "2024-01-15",
  "monthly_volume": 2500000.00,
  "chargeback_rate": 0.01,
  "timestamp": "2026-02-27T10:30:00Z"
}
```

---

## 4. AML Case Management

When AML scoring returns ESCALATE/BLOCK, create a case:

```
POST /api/v1/aml/cases
```

```json
{
  "entity_id": "uuid (user_id)",
  "entity_type": "USER",
  "risk_score": 0.85,
  "risk_factors": ["high_value_transfer", "new_account"],
  "source": "ml_scoring"
}
```

**List cases** (for your compliance dashboard):
```
GET /api/v1/aml/cases?status=OPEN&limit=50&offset=0
```

**Update case** (when analyst reviews):
```
PUT /api/v1/aml/cases/{case_id}
{
  "status": "CLOSED",
  "resolution": "FALSE_POSITIVE",
  "notes": "Verified legitimate business transaction"
}
```

---

## 5. Label Feedback (Critical for ML Improvement)

**This is the most important integration for ML model improvement.** Every time a fraud case is resolved, a chargeback comes in, or an AML investigation concludes, submit a label back to the ML service.

```
POST /api/v1/ml/labels/submit
```

```json
{
  "prediction_id": "uuid (from scoring response)",
  "domain": "fraud",
  "label": 1,
  "label_source": "MANUAL_REVIEW",
  "labeled_by": "uuid (analyst user_id)",
  "reason": "Confirmed fraud — chargeback received"
}
```

- `label`: `1` = fraudulent/suspicious, `0` = legitimate
- `domain`: `fraud`, `aml`, or `merchant`
- `label_source`: One of:
  - `MANUAL_REVIEW` — analyst reviewed and confirmed
  - `CHARGEBACK_FEED` — chargeback received from payment processor
  - `SAR_CONFIRMED` — SAR investigation confirmed suspicious activity
  - `AUTOMATED_RULE` — rule-based system confirmed
  - `PARTNER_FEEDBACK` — partner bank/PSP feedback

**When to submit labels:**
1. Fraud review resolved → submit label with `prediction_id` from original score
2. Chargeback received → submit label=1, source=CHARGEBACK_FEED
3. AML case closed → submit label based on resolution (FALSE_POSITIVE=0, CONFIRMED=1)
4. Merchant suspended for fraud → submit label=1, domain=merchant

**Batch labels** (up to 500):
```
POST /api/v1/ml/labels/submit/batch
{ "labels": [ ...same schema as single... ] }
```

**Check maturity** (how close to ML mode):
```
GET /api/v1/ml/labels/statistics
```

Returns per-domain stats: `total_predictions`, `labeled_count`, `maturity_level` (COLD → WARMING → WARM → HOT).

---

## 6. SHAP Explanations

Get human-readable explanation for any scored transaction:

```
GET /api/v1/ml/fraud/explanation/{prediction_id}
GET /api/v1/aml/explanation/{prediction_id}
GET /api/v1/ml/merchant/explanation/{prediction_id}
```

Returns top contributing features with descriptions — useful for analyst dashboards and regulatory compliance.

---

## Integration Checklist

1. **Payment created** → call `POST /fraud/score`, store `prediction_id` with the transaction
2. **Large transfer** → call `POST /aml/score`, create case if ESCALATE
3. **Merchant onboarding** → call `POST /merchant/score`
4. **Fraud review resolved** → call `POST /labels/submit` with outcome
5. **Chargeback received** → call `POST /labels/submit` (label=1, source=CHARGEBACK_FEED)
6. **AML case closed** → call `POST /labels/submit` with outcome
7. **Dashboard** → use `/labels/statistics` to show ML maturity progress

## Postman Collection

Import `tests/pesaflow-ml.postman_collection.json` into Postman. Set the `base_url` variable to `https://ml.pesaflow.ai` (or `http://localhost:18743` for local dev). The collection has 38 endpoints across 9 folders with test scripts.

## Error Handling

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 201 | Created (AML cases) |
| 400 | Bad request — check payload |
| 404 | Not found (prediction_id/case_id doesn't exist) |
| 422 | Validation error — check field types/formats |
| 500 | Server error — retry with backoff |
| 503 | Service unavailable (DB/Redis down) — retry later |

All IDs are UUIDs. Amounts are floats (KES). Timestamps are ISO 8601.
