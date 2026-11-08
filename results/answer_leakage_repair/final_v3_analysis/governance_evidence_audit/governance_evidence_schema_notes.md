# Governance evidence audit schema notes

This is an offline, provider-independent audit of the frozen final-main manifests.

## Source-native fields

- **LongMemEval:** `source_evidence.answer_session_ids`,
  `source_evidence.evidence_session_ids`, `source_evidence.question_type`, and
  `oracle_state.current_status`, `ordered_evidence`, `relevant_session_ids`,
  `temporal_relation`; the base question and expected answer link the task target.
- **Memora:** `source_evidence.memory_evidence`, `forgetting_evidence`,
  `relevant_source_session_ids`, and `governance_capabilities`, together with
  `oracle_state.current_status`, `obsolete_item`, `replacement_item`, and
  `relevant_session_ids`.
- **OAKS:** `source_evidence.document_id`, `chunk_to_answer`,
  `ordered_answer_states`, `transition_chunks`, and `evidence_fields_used`,
  together with the ordered answer states and transitions in `oracle_state`.

`corruption_specification.deterministic`, `operator`, `operation`, and
`source_oracle_fields` in each condition row are checked against the base's
declared `supported_fault_operators`, `fault_support_reasons`, and oracle state.
No model output is consulted. `source_locator` preserves the persisted native
identifier/fingerprint (and OAKS document ID where present).
