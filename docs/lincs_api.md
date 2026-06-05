# SigCom LINCS API Reference

Reference for querying the SigCom LINCS L1000 signature database
(Ma'ayan Lab, Icahn School of Medicine at Mount Sinai). Used by
`discovery/lincs_api.py` to screen an iMEMORY therapeutic signature against
real L1000 chemical perturbation signatures and return compounds that reverse
the disease signature.

This file is intended both as human documentation and as agent/RAG context.

---

## Base URLs

- **Data API:** `https://maayanlab.cloud/sigcom-lincs/data-api/api/v1`
- **Metadata API:** `https://maayanlab.cloud/sigcom-lincs/metadata-api`

Corpus scale (approx.): 18,000+ signature queries available, 1,696 gene
queries, drawn from the LINCS L1000 program (chemical and genetic
perturbations across many cell lines).

---

## The query workflow (3 steps)

A connectivity query — "which compounds reverse my disease signature?" — is a
three-step process. The iMEMORY therapeutic signature is first decomposed into
an UP gene set (genes elevated in disease) and a DOWN gene set (genes suppressed
in disease). Then:

### Step 1 — Resolve gene symbols to entity UUIDs (Metadata API)

The enrichment endpoint operates on internal entity UUIDs, not gene symbols, so
symbols must be resolved first.

```
POST https://maayanlab.cloud/sigcom-lincs/metadata-api/entities/find
Content-Type: application/json

{
  "filter": {
    "where": { "meta.symbol": { "inq": ["STAT1", "IRF7", "MX1", "TLR2"] } },
    "fields": ["id", "meta.symbol"]
  }
}
```

Response (abbreviated):

```json
[
  { "id": "<entity-uuid-1>", "meta": { "symbol": "STAT1" } },
  { "id": "<entity-uuid-2>", "meta": { "symbol": "IRF7" } }
]
```

Build a `{symbol: uuid}` map from the response.

### Step 2 — Two-sided rank enrichment (Data API)

Submit the up and down entity UUID lists. The two-sided enrichment compares the
query against every signature in the chosen library.

```
POST https://maayanlab.cloud/sigcom-lincs/data-api/api/v1/enrich/ranktwosided
Content-Type: application/json

{
  "up_entities":   ["<up-uuid-1>",   "<up-uuid-2>",   ...],
  "down_entities": ["<down-uuid-1>", "<down-uuid-2>", ...],
  "limit": 100,
  "database": "<signature-library-uuid>"
}
```

Response (abbreviated):

```json
{
  "results": [
    { "uuid": "<signature-uuid>", "zSum": -2.41, "zUp": -1.7, "zDown": -1.6, ... },
    ...
  ]
}
```

**Score interpretation.** The query represents the DISEASE direction (up = up in
disease). A signature (compound) with a **negative `zSum`** is anti-correlated
with the disease signature — it REVERSES the disease and is the therapeutic
candidate. A positive `zSum` mimics the disease (avoid). Rank ascending by
`zSum`; the most negative are the best reversers.

### Step 3 — Resolve signature UUIDs to perturbagen metadata (Metadata API)

The enrichment returns signature UUIDs; resolve them to compound identities.

```
POST https://maayanlab.cloud/sigcom-lincs/metadata-api/signatures/find
Content-Type: application/json

{
  "filter": {
    "where": { "id": { "inq": ["<signature-uuid-1>", "<signature-uuid-2>"] } }
  }
}
```

Response contains, per signature, a `meta` object with perturbagen fields
(commonly `pert_name`, `pert_id`, cell line, dose, time). Use `pert_name` /
`pert_id` as the compound identity, and (where available) the canonical SMILES.

---

## Finding the signature library UUID

The `database` field in Step 2 is a library UUID. List libraries and pick the
chemical-perturbation consensus library (do not hard-code a UUID — it can change;
fetch it):

```
GET https://maayanlab.cloud/sigcom-lincs/metadata-api/libraries
```

Look for the LINCS L1000 chemical perturbation consensus signature library
(name typically contains "L1000" and "Chem Pert" / "Consensus"). Use its `id`
as the `database` value. There are also CRISPR KO and over-expression libraries
for genetic-perturbation queries.

---

## Other useful endpoints

- `POST /data-api/api/v1/enrich/rank` — one-sided rank enrichment (single gene set).
- `POST /data-api/api/v1/enrich/overlap` — set-overlap enrichment (Fisher-style) for
  discrete gene sets rather than ranked signatures.
- `POST /data-api/api/v1/fetch/rank` — fetch the full ranked gene vector for a given
  signature UUID (use to pull a compound's L1000 signature for downstream
  embedding-space recovery scoring).
- `POST /metadata-api/signatures/find` — query signatures by metadata.
- `GET  /metadata-api/signatures/{id}` — fetch a single signature's metadata.

---

## Notes for implementers

- All POST bodies are JSON; set `Content-Type: application/json`.
- The metadata API uses a LoopBack-style `filter` object (`where`, `fields`,
  `inq` for "in list").
- Gene symbols must match the SigCom LINCS entity vocabulary (HGNC symbols).
  Resolve in batches; not every symbol will exist.
- Field names (`zSum`, `up_entities`, `pert_name`) should be verified against a
  live response — the API evolves. `discovery/lincs_api.py` is written to fail
  gracefully and log the raw response shape if fields differ.
- Rate-limit politely; cache resolved gene UUIDs (they are stable).
- Reference: Evangelista et al., "SigCom LINCS: data and metadata search engine
  for a million gene expression signatures," Nucleic Acids Research, 2022.
