import chromadb


class RegulatoryPolicyStore:
    """
    In-memory ChromaDB HNSW vector store grounding explanations and collection scripts
    strictly in vetted RBI regulatory clauses (zero ungrounded parametric recall).
    """

    def __init__(self, collection_name: str = "rbi_compliance_policies"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._seed_default_policies()

    def _seed_default_policies(self) -> None:
        if self.collection.count() > 0:
            return
        docs = [
            "RBI MRM & FREE-AI Mandate: Automated adverse actions require deterministic reason codes, audit trail logging, and customer right to human review.",
            "RBI Digital Lending Fair Practices: Collection outreach is strictly permitted only between 08:00 and 19:00 IST, capped at 3 contacts per week without coercion.",
            "Velocity & Layering Freeze Policy: Accounts exhibiting >3 sigma 1-hour outflow velocity or shared device linkage with confirmed fraud rings undergo temporary debit-freeze pending 2FA KYC verification.",
            "EMI Restructuring Eligibility: Borrowers with survival probability S(30d) < 0.65 and zero prior default history qualify for a 3-month tenure extension offer.",
        ]
        ids = ["POL_MRM_01", "POL_COLL_02", "POL_FRAUD_03", "POL_RESTRUCT_04"]
        metadatas = [
            {"category": "governance"},
            {"category": "collections"},
            {"category": "fraud"},
            {"category": "collections"},
        ]
        self.collection.add(documents=docs, ids=ids, metadatas=metadatas)

    def retrieve_clauses(
        self, query_text: str, n_results: int = 2
    ) -> list[dict[str, str]]:
        res = self.collection.query(query_texts=[query_text], n_results=n_results)
        docs = res.get("documents", [[]])[0]
        ids = res.get("ids", [[]])[0]
        return [{"policy_id": pid, "clause": doc} for pid, doc in zip(ids, docs)]
