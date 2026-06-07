import json
import sqlite3

class GlossaryRetriever:
    def __init__(self, glossary_path, db_path):

        #load glossary from json file
        with open(glossary_path, 'r', encoding="utf-8") as f:
            self.glossary = json.load(f)
        self.db_path = db_path

    def retrieve_metrics(self, question):
        question = question.lower()
        matched_metrics = []

        for metric_key, metric_info in self.glossary["metrics"].items():

            if "alias_of" in metric_info:
                continue
            
            #metric key
            keywords = [metric_key.lower()]

            #business name
            if "business_name" in metric_info:
                keywords.append(
                    metric_info["business_name"].lower()
                )

            #aliases
            if "aliases" in metric_info:
                keywords.extend(
                    [a.lower() for a in metric_info["aliases"]]
                )

            for keyword in keywords:
                if keyword in question:
                    matched_metrics.append(metric_key)
                    break

        return matched_metrics
    
    def retrieve_time_periods(self, question):
        question = question.lower()
        matched = {}
        for period_key, period_sql in self.glossary["time_periods"].items():
            keyword = period_key.replace("_", " ")
            if keyword in question:
                matched[period_key] = period_sql
        return matched
    
    def retrieve_metric_details(self, metric_name):
        metric_info = self.glossary["metrics"][metric_name]
        return {
            "metric": metric_name,
            "business_name": metric_info.get("business_name"),
            "preferred_source": metric_info.get("preferred_source"),
            "preferred_fields": metric_info.get("preferred_source_fields",[]),
            "sql_formula":metric_info.get("sql_formula")
        }
    
    #DB schema
    def get_schema_context(self, preferred_source):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"PRAGMA table_info({preferred_source})"
        )
        columns = cursor.fetchall()
        conn.close()
        return [
            row[1]
            for row in columns
        ]
    
    #combine everything together
    def retrieve(self, question):
        matched_metrics = self.retrieve_metrics(question)
        metric_details = [self.retrieve_metric_details(m) for m in matched_metrics]
        time_periods = self.retrieve_time_periods(question)
        schema_context = {}

        if metric_details:
            metric_info = self.glossary["metrics"][matched_metrics[0]]
            sources = []
            if metric_info.get("preferred_source"):
                sources.append(metric_info["preferred_source"])
            sources += metric_info.get("fallback_sources", [])

            for source in sources:
                try:
                    schema_context[source] = self.get_schema_context(source)
                except Exception:
                    pass

        return {"metrics": metric_details, "time_periods": time_periods, "schema": schema_context}

# Retriever Unit Test 
if __name__=="__main__":
    retriever = GlossaryRetriever("/Users/chowingchan/Desktop/Project/AI-Analytics-Copilot/AI-Analytics-Copilot/config/glossary.json",
                                  "/Users/chowingchan/Desktop/Project/AI-Analytics-Copilot/Competitive-Intelligence-Internal-Analytics-System/data/database/thelook_ecommerce.db")

    test_questions = [
        "What is total revenue in last 30 days?",
        "What is average order value?",
        "What is return rate?",
        "Show me cvr for last 7 days",
        "How many new users did we acquire last month?"
    ]

    for question in test_questions:
        print("\nQuestions:", question)
        print(retriever.retrieve(question))

