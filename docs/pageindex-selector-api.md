# PageIndex Selector API

This document describes the dedicated selector vector ingestion and CRUD APIs.

## Purpose

These endpoints manage the selector collections used for document-level candidate selection:

- `pageindex_selector_chunks`
- `pageindex_selector_doc_headers`

The selector pipeline uses deterministic query expansion, chunk/header retrieval, document-level scoring, reranking, and confidence estimation.

## Endpoints

Base prefix: `/api/v1/retrieval`

### Ingest

- `POST /pageindex/selector/ingest`
- `POST /pageindex/selector/ingest/bulk`

Both routes reuse PageIndex indexing and refresh selector vectors. Re-ingestion deletes stale vectors first.

### Read

- `GET /pageindex/selector/documents/{document_id}`
- `POST /pageindex/selector/documents/list`
- `GET /pageindex/selector/documents/{document_id}/chunks`
- `GET /pageindex/selector/chunks/{chunk_id}`
- `POST /pageindex/selector/search/documents`
- `POST /pageindex/selector/search/chunks`

`/search/documents` returns debug fields:

- `query_variants_used`
- `matched_documents`
- `matched_chunks`
- `score_breakdown`
- `matched_documents_debug`
- `matched_chunks_debug`

### Update

- `POST /pageindex/selector/documents/{document_id}/refresh`
- `PATCH /pageindex/selector/documents/{document_id}/metadata`
- `PUT /pageindex/selector/documents/{document_id}/chunks`

`PUT .../chunks` replaces the document's selector chunks safely by deleting old vectors first.

### Delete

- `DELETE /pageindex/selector/documents/{document_id}`
- `DELETE /pageindex/selector/chunks/{chunk_id}`
- `DELETE /pageindex/selector/collection/clear` (admin only)

## Example evaluation scenario

Use this script:

- `scripts/evaluate_pageindex_selector.py`

Expected uploaded files:

- `Health_Authority_Communication_Example.pdf`
- `Summary_Basis_of_Approval_Example.pdf`
- `FDA_Guidance_Document_Example.pdf`
- `510.pdf`

Example run:

```bash
python scripts/evaluate_pageindex_selector.py \
  --base-url http://localhost:8080 \
  --token <OPENWEBUI_API_TOKEN> \
  --file-id <FILE_ID_1> \
  --file-id <FILE_ID_2> \
  --file-id <FILE_ID_3> \
  --file-id <FILE_ID_4>
```

The script asserts the top-ranked candidate is:

- `Health_Authority_Communication_Example.pdf`
