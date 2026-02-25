"""Cross-domain label transfer rules with confidence decay.

Defines how labels transfer between domains:
  - Fraud→AML: confirmed fraud → mark AML suspicious (0.8x confidence)
  - Fraud→Merchant: confirmed fraud → merchant risk signal (0.6x confidence)
  - AML→Merchant: confirmed suspicious → merchant risk signal (0.7x confidence)
  - Merchant→Fraud: blocked merchant → transactions from this merchant are riskier (0.5x confidence)

Safety constraint: Max cross-domain labels = 20% of labeled data per target domain.
"""

import uuid
from typing import Any

from loguru import logger


class TransferRule:
    """A single cross-domain label transfer rule."""

    def __init__(
        self,
        source_domain: str,
        target_domain: str,
        source_label: int,
        transferred_label: int,
        confidence_decay: float,
        description: str,
    ):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.source_label = source_label
        self.transferred_label = transferred_label
        self.confidence_decay = confidence_decay
        self.description = description

    def __repr__(self) -> str:
        return f"TransferRule({self.source_domain}→{self.target_domain}, {self.confidence_decay}x)"


# Default cross-domain transfer rules
DEFAULT_TRANSFER_RULES = [
    TransferRule(
        source_domain="fraud",
        target_domain="aml",
        source_label=1,
        transferred_label=1,
        confidence_decay=0.8,
        description="Confirmed fraud → AML suspicious",
    ),
    TransferRule(
        source_domain="fraud",
        target_domain="merchant",
        source_label=1,
        transferred_label=1,
        confidence_decay=0.6,
        description="Confirmed fraud → merchant risk signal",
    ),
    TransferRule(
        source_domain="aml",
        target_domain="merchant",
        source_label=1,
        transferred_label=1,
        confidence_decay=0.7,
        description="Confirmed AML suspicious → merchant risk signal",
    ),
    TransferRule(
        source_domain="merchant",
        target_domain="fraud",
        source_label=1,
        transferred_label=1,
        confidence_decay=0.5,
        description="Blocked merchant → transactions from merchant are riskier",
    ),
]


_DOMAIN_TABLE = {
    "fraud": "ml_predictions",
    "aml": "aml_predictions",
    "merchant": "merchant_risk_predictions",
}

# Cross-domain linkage via entity IDs
_DOMAIN_ENTITY_COL = {
    "fraud": "user_id",
    "aml": "user_id",
    "merchant": "merchant_id",
}


class LabelTransferEngine:
    """Executes cross-domain label transfers with safety constraints."""

    def __init__(
        self,
        db_pool: Any,
        rules: list[TransferRule] | None = None,
        max_transfer_ratio: float = 0.20,
    ):
        self._db = db_pool
        self._rules = rules or DEFAULT_TRANSFER_RULES
        self._max_transfer_ratio = max_transfer_ratio

    async def execute_transfers(self) -> dict:
        """Run all transfer rules and return transfer counts."""
        if not self._db:
            return {}

        results = {}
        for rule in self._rules:
            try:
                count = await self._execute_rule(rule)
                key = f"{rule.source_domain}→{rule.target_domain}"
                results[key] = count
            except Exception as e:
                logger.error(f"Transfer rule {rule} failed: {e}")
                results[f"{rule.source_domain}→{rule.target_domain}"] = 0

        return results

    async def _execute_rule(self, rule: TransferRule) -> int:
        """Execute a single transfer rule."""
        source_table = _DOMAIN_TABLE[rule.source_domain]
        target_table = _DOMAIN_TABLE[rule.target_domain]

        async with self._db.acquire() as conn:
            # Count existing labels in target domain (for ratio check)
            target_labeled = await conn.fetchval(f"SELECT COUNT(*) FROM {target_table} WHERE label IS NOT NULL")

            # Count existing transfers for this rule
            existing_transfers = await conn.fetchval(
                """
                SELECT COUNT(*) FROM ssl_cross_domain_transfers
                WHERE source_domain = $1 AND target_domain = $2
                """,
                rule.source_domain,
                rule.target_domain,
            )

            # Check max transfer ratio
            max_transfers = int(max(10, target_labeled * self._max_transfer_ratio))
            remaining_budget = max_transfers - existing_transfers
            if remaining_budget <= 0:
                logger.info(f"Transfer budget exhausted for {rule.source_domain}→{rule.target_domain}")
                return 0

            # Find source labels that haven't been transferred yet
            source_rows = await conn.fetch(
                f"""
                SELECT s.id as source_id, s.{_DOMAIN_ENTITY_COL[rule.source_domain]} as entity_id,
                       s.label, s.risk_score
                FROM {source_table} s
                WHERE s.label = $1
                  AND NOT EXISTS (
                      SELECT 1 FROM ssl_cross_domain_transfers t
                      WHERE t.source_prediction_id = s.id
                        AND t.target_domain = $2
                  )
                ORDER BY s.labeled_at DESC NULLS LAST
                LIMIT $3
                """,
                rule.source_label,
                rule.target_domain,
                remaining_budget,
            )

            transferred = 0
            for source_row in source_rows:
                entity_id = str(source_row["entity_id"])

                # Find matching prediction in target domain
                target_entity_col = _DOMAIN_ENTITY_COL.get(rule.target_domain, "user_id")
                target_row = await conn.fetchrow(
                    f"""
                    SELECT id FROM {target_table}
                    WHERE {target_entity_col} = $1
                      AND label IS NULL
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    entity_id,
                )

                if not target_row:
                    continue

                # Record transfer
                await conn.execute(
                    """
                    INSERT INTO ssl_cross_domain_transfers
                    (id, source_domain, target_domain, source_prediction_id,
                     target_prediction_id, source_label, transferred_label,
                     confidence_decay, transfer_rule)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    str(uuid.uuid4()),
                    rule.source_domain,
                    rule.target_domain,
                    str(source_row["source_id"]),
                    str(target_row["id"]),
                    rule.source_label,
                    rule.transferred_label,
                    rule.confidence_decay,
                    rule.description,
                )
                transferred += 1

            logger.info(
                f"Transferred {transferred} labels: {rule.source_domain}→{rule.target_domain} "
                f"(decay={rule.confidence_decay})"
            )
            return transferred
