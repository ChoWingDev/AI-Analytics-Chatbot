import json
import sqlite3
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GEMINI_API_KEY, GLOSSARY_PATH, DB_PATH

class GlossaryRetriever:
    def __init__(self, glossary_path=GLOSSARY_PATH, db_path=DB_PATH):
        with open(glossary_path, "r", encoding="utf-8") as f:
            self.glossary = json.load(f)

        self.db_path = db_path

        # 初始化 Gemini Embedding
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GEMINI_API_KEY
        )

        # 只建立結構化的指標（Metrics）向量文件，徹底丟棄 Word 物理切片
        all_docs = self.build_metric_documents()

        self.vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            collection_name="metric_glossary"
        )

    def build_metric_documents(self):
        documents = []
        for metric_key, metric_info in self.glossary["metrics"].items():
            if "alias_of" in metric_info:
                continue

            text = f"""
            Metric Key: {metric_key}
            Business Name: {metric_info.get("business_name")}
            Definition: {metric_info.get("definition")}
            Aliases: {metric_info.get("aliases", [])}
            Formula: {metric_info.get("sql_formula")}
            Business Logic: {metric_info.get("business_logic", [])}
            Preferred Source: {metric_info.get("preferred_source")}
            """
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source_type": "metric", "metric_key": metric_key}
                )
            )
        return documents

    def retrieve_metrics(self, question, top_k=2):  # 尋找 SQL 通常前 1~2 個指標就夠，太多會干擾
        docs = self.vectorstore.similarity_search(question, k=top_k + 5)
        metric_keys = []
        for doc in docs:
            if doc.metadata.get("source_type") == "metric":
                metric_keys.append(doc.metadata["metric_key"])
        return metric_keys[:top_k]

    def retrieve_time_periods(self, question):
        # 清理標點符號，防止問號阻礙匹配
        cleaned_question = question.lower().replace("?", "").strip()
        matched = {}
        for period_key, period_sql in self.glossary["time_periods"].items():
            keyword = period_key.replace("_", " ")
            if keyword in cleaned_question or period_key in cleaned_question:
                matched[period_key] = period_sql
        return matched

    def retrieve_metric_details(self, metric_name):
        metric_info = self.glossary["metrics"][metric_name]
        return {
            "metric": metric_name,
            "business_name": metric_info.get("business_name"),
            "definition": metric_info.get("definition"),
            "business_logic": metric_info.get("business_logic", []),
            "preferred_source": metric_info.get("preferred_source"),
            "sql_formula": metric_info.get("sql_formula")
        }

    def get_schema_context(self, preferred_source):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(f"PRAGMA table_info({preferred_source})")
            columns = cursor.fetchall()
            return [row[1] for row in columns]
        except Exception as e:
            return []
        finally:
            conn.close()

    def retrieve(self, question):
        matched_metrics = self.retrieve_metrics(question)
        metric_details = [self.retrieve_metric_details(m) for m in matched_metrics]
        time_periods = self.retrieve_time_periods(question)

        schema_context = {}
        sources = []

        # 按需動態收集需要注入的 Table
        for metric in matched_metrics:
            metric_info = self.glossary["metrics"][metric]
            if metric_info.get("preferred_source"):
                sources.append(metric_info["preferred_source"])
            if metric_info.get("fallback_sources"):
                sources += metric_info["fallback_sources"]

        # 只從 SQLite 動態抓取相關 Table 的真實 Schema
        for source in set(sources):
            cols = self.get_schema_context(source)
            if cols:
                schema_context[source] = cols

        return {
            "metrics": metric_details,
            "time_periods": time_periods,
            "schema": schema_context
        }