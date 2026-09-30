import uuid
import re
import json
import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse


STOP_WORDS = {
    "what", "is", "the", "are", "of", "in", "to", "a", "an", "and", "or",
    "how", "why", "when", "where", "who", "which", "this", "that", "for",
    "with", "on", "at", "by", "from", "as", "be", "was", "were", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "i", "me", "my",
    "we", "us", "our", "you", "your", "he", "she", "it", "they", "them",
    "their", "about", "give", "show", "tell", "list", "all", "any",
}


class HealthReportAgent(ChatAgent):
    """Health report RAG agent with hybrid retrieval + reranking."""

    def load_context(self, context):
        from databricks.sdk import WorkspaceClient
        import mlflow.deployments

        self.w = WorkspaceClient()
        self.deploy_client = mlflow.deployments.get_deploy_client("databricks")

        self.vs_index = "workspace.default.health_report_chunks_index"
        self.chunks_table = "workspace.default.health_report_pdf_chunks"
        self.llm_endpoint = "databricks-meta-llama-3-3-70b-instruct"
        self.warehouse_id = "807f80ae0b4f84a8"
        self.candidate_k = 20
        self.final_k = 8
        self.max_context_chars = 500
        self.max_tokens = 512
        self.lab_data_table = "workspace.default.health_report_lab_data"
        self.content_filter = '{"chunk_type": ["text", "table", "section_header"]}'

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def _vector_search(self, query):
        """Semantic retrieval via Vector Search."""
        results = self.w.vector_search_indexes.query_index(
            index_name=self.vs_index,
            columns=["chunk_id", "chunk_text", "chunk_type", "page", "section_id", "file_name"],
            query_text=query,
            num_results=self.candidate_k,
            filters_json=self.content_filter,
        )
        return results.result.data_array if results and results.result else []

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def _keyword_search(self, query, limit=20):
        """Lexical/keyword retrieval via SQL ILIKE matching on chunks table."""
        words = [w.lower().strip(".,!?;:()[]{}\"')") for w in query.split()]
        keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        if not keywords:
            return []

        like_clauses = " OR ".join(
            f"chunk_text ILIKE '%{re.escape(k).replace(chr(39), chr(39)+chr(39))}%'" for k in keywords[:8]
        )
        sql = (
            f"SELECT chunk_id, chunk_text, chunk_type, page, section_id, file_name "
            f"FROM {self.chunks_table} "
            f"WHERE ({like_clauses}) "
            f"AND chunk_type IN ('text', 'table', 'section_header') "
            f"LIMIT {limit}"
        )
        try:
            result = self.w.statement_execution.execute_statement(
                warehouse_id=self.warehouse_id,
                statement=sql,
                catalog="workspace",
                schema="default",
                wait_timeout="5s",
            )
            if result.result and result.result.data_array:
                return [list(row) for row in result.result.data_array]
        except Exception:
            pass
        return []

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def _hybrid_retrieve(self, query):
        """Hybrid retrieval: merge vector search + keyword search, deduplicate by chunk_id."""
        vector_results = self._vector_search(query)

        merged = {}
        for row in vector_results:
            cid = row[0]
            if cid not in merged:
                merged[cid] = {"data": list(row), "source": "vector", "score": row[-1]}

        # Keyword search via SQL warehouse (disabled for deployment stability)
        try:
            keyword_results = self._keyword_search(query, self.candidate_k)
            for row in keyword_results:
                cid = row[0]
                if cid not in merged:
                    merged[cid] = {"data": list(row) + [0.0], "source": "keyword", "score": 0.0}
                else:
                    merged[cid]["source"] = "both"
        except Exception:
            pass

        return list(merged.values())

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def _structured_lookup(self, query, intent):
        """Query structured lab data table for test values and abnormal flags."""
        if intent == "abnormal_normal":
            sql = (
                f"SELECT test_name, value, unit, reference_range, status, page "
                f"FROM {self.lab_data_table} "
                f"WHERE status ILIKE '%abnormal%' OR status ILIKE '%borderline%' "
                f"ORDER BY page"
            )
        elif intent == "summary":
            sql = (
                f"SELECT test_name, value, unit, reference_range, status, page "
                f"FROM {self.lab_data_table} "
                f"ORDER BY page"
            )
        else:
            words = [w.lower().strip(".,!?;:()[]{}\"')") for w in query.split()]
            keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
            if not keywords:
                return []
            like_clauses = " OR ".join(
                f"test_name ILIKE '%{re.escape(k).replace(chr(39), chr(39)+chr(39))}%'" for k in keywords[:5]
            )
            sql = (
                f"SELECT test_name, value, unit, reference_range, status, page "
                f"FROM {self.lab_data_table} WHERE {like_clauses} ORDER BY page"
            )
        try:
            result = self.w.statement_execution.execute_statement(
                warehouse_id=self.warehouse_id,
                statement=sql,
                catalog="workspace",
                schema="default",
                wait_timeout="5s",
            )
            if result.result and result.result.data_array:
                return [list(row) for row in result.result.data_array]
        except Exception:
            pass
        return []

    def _classify_intent(self, query):
        """Classify user question intent: specific_qa, abnormal_normal, or summary."""
        q = query.lower()
        if any(w in q for w in ["abnormal", "out of range", "out-of-range",
                                 "high", "low", "elevated", "below",
                                 "above normal", "borderline", "not normal"]):
            return "abnormal_normal"
        if any(w in q for w in ["summary", "overview", "summarize", "summarise",
                                 "key findings", "main findings", "overall",
                                 "general", "all tests", "everything",
                                 "health status"]):
            return "summary"
        return "specific_qa"

    def _validate_context(self, query, reranked, structured_results):
        """Check if retrieved context is sufficient to answer the question."""
        if not reranked and not structured_results:
            return False
        return True

    @mlflow.trace(span_type=SpanType.LLM)
    def _rerank(self, query, candidates, top_k=8):
        """LLM-as-reranker: score candidate chunks by relevance to the query."""
        if len(candidates) <= top_k:
            return candidates[:top_k]

        chunk_summaries = []
        for i, cand in enumerate(candidates):
            data = cand["data"]
            text = str(data[1])[:200]
            page = data[3]
            section = data[4]
            chunk_summaries.append(f'{i + 1}. [p{page} {section}] {text}')

        prompt = (
            f"Rank these document chunks by relevance to the query. "
            f"Return ONLY a JSON array of objects with \"index\" (1-based) and \"score\" (0.0-10.0).\n\n"
            f"Query: {query}\n\nChunks:\n{chr(10).join(chunk_summaries)}\n\n"
            f'Format: [{{"index": 1, "score": 9.5}}, ...]'
        )

        try:
            response = self.deploy_client.predict(
                endpoint=self.llm_endpoint,
                inputs={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                    "temperature": 0.0,
                },
            )
            raw = response["choices"][0]["message"]["content"].strip()
            json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                scored = []
                for item in scores:
                    idx = int(item["index"]) - 1
                    if 0 <= idx < len(candidates):
                        candidates[idx]["rerank_score"] = float(item["score"])
                        scored.append(candidates[idx])
                for c in candidates:
                    if "rerank_score" not in c:
                        c["rerank_score"] = 0.0
                        scored.append(c)
                scored.sort(key=lambda x: x["rerank_score"], reverse=True)
                return scored[:top_k]
        except Exception:
            pass

        return candidates[:top_k]

    @mlflow.trace(span_type=SpanType.LLM)
    def _generate(self, question, context_str, intent=None):
        """Generate answer via Foundation Model API with page citations."""
        if intent == "summary":
            system_prompt = (
                "You are a scientist assistant. Provide a comprehensive summary "
                "of the health report based on the available context. Include key "
                "findings, notable test results, and any abnormal values with their "
                "reference ranges. Always cite page numbers using (p<page>) format."
            )
        else:
            system_prompt = (
                "You are a medical report assistant helping a patient understand their health report. "
                "Answer using the provided health report context (document chunks + structured lab data). "
                "When the question is about a lab test or biomarker, structure your answer as:\n"
                "1. What it is: A brief, plain-language explanation of what the test/substance is "
                "and why it's measured (use your general medical knowledge for this part).\n"
                "2. Your report shows: List each matching test value with its unit. Include the "
                "reference range and whether the value is normal/abnormal/borderline.\n"
                "3. In simple terms: A one-sentence plain-language summary of what the test is for "
                "and whether the patient's values are normal.\n\n"
                "Always cite page numbers using (p<page>) format. If the evidence is insufficient "
                "to answer the question, say exactly: \"I don't have enough information to answer "
                "this question based on the report.\" Be thorough yet concise."
            )
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
        response = self.deploy_client.predict(
            endpoint=self.llm_endpoint,
            inputs={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.1,
            },
        )
        return response["choices"][0]["message"]["content"]

    @mlflow.trace(span_type=SpanType.CHAIN)
    def predict(self, messages, context=None, custom_inputs=None):
        """Intent → routing → hybrid retrieval + structured lookup → rerank → validate → generate."""
        user_question = ""
        for msg in reversed(messages):
            if msg.role == "user" and msg.content:
                user_question = msg.content
                break

        if not user_question:
            return ChatAgentResponse(
                messages=[ChatAgentMessage(role="assistant", content="Please ask a question about the health report.", id=str(uuid.uuid4()))]
            )

        mlflow.update_current_trace(metadata={
            "mlflow.trace.user": "health-report-user",
            "mlflow.trace.session": f"session-{uuid.uuid4().hex[:8]}",
        })

        # Step 1: Classify intent (specific_qa / abnormal_normal / summary)
        intent = self._classify_intent(user_question)

        # Step 2: Route to appropriate retrieval paths based on intent
        candidates = self._hybrid_retrieve(user_question)
        structured_results = self._structured_lookup(user_question, intent)

        # Step 3: Validate context sufficiency
        if not self._validate_context(user_question, candidates, structured_results):
            return ChatAgentResponse(
                messages=[ChatAgentMessage(role="assistant", content="I don't have enough information to answer this question based on the report.", id=str(uuid.uuid4()))],
                custom_outputs={"retrieved_chunks": [], "structured_lab_data": [], "query": user_question, "intent": intent}
            )

        # Step 4: Rerank vector candidates to top FINAL_K chunks
        reranked = self._rerank(user_question, candidates, self.final_k) if candidates else []

        # Step 5: Build context from reranked chunks + structured lab data
        context_parts = []
        retrieved_info = []
        for i, cand in enumerate(reranked):
            data = cand["data"]
            chunk_id, chunk_text, chunk_type, page, section_id = data[0], data[1], data[2], data[3], data[4]
            score = cand.get("rerank_score", cand.get("score", 0.0))
            source = cand.get("source", "unknown")
            context_parts.append(f"[S{i + 1}] p{page} {section_id} (source: {source})\n{str(chunk_text)[:self.max_context_chars]}")
            retrieved_info.append({
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "page": page,
                "section_id": section_id,
                "score": round(score, 4),
                "source": source,
                "text_preview": str(chunk_text)[:200],
            })

        structured_info = []
        if structured_results:
            lab_parts = []
            for r in structured_results:
                test_name, value, unit, ref_range, status, page = r[0], r[1], r[2], r[3], r[4], r[5]
                lab_parts.append(f"[LAB] p{page} {test_name}: {value} {unit} (ref: {ref_range}, status: {status})")
                structured_info.append({
                    "test_name": test_name,
                    "value": value,
                    "unit": unit,
                    "reference_range": ref_range,
                    "status": status,
                    "page": page,
                })
            context_parts.append("Structured Lab Data:\n" + "\n".join(lab_parts))

        context_str = "\n\n".join(context_parts)

        # Step 6: Generate answer with citations
        answer = self._generate(user_question, context_str, intent)

        return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=answer, id=str(uuid.uuid4()))],
            custom_outputs={
                "retrieved_chunks": retrieved_info,
                "structured_lab_data": structured_info,
                "query": user_question,
                "intent": intent,
                "candidate_count": len(candidates),
                "final_count": len(reranked),
                "source_pages": list(set(int(r["data"][3]) for r in reranked)) if reranked else [],
            }
        )


import mlflow.models
mlflow.models.set_model(HealthReportAgent())
