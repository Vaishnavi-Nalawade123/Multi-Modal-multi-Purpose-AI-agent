import requests
import json

SCHEMA = """
You are an AI assistant that converts natural language questions into valid Apache Spark SQL queries for a Hive table named youtube_data1.trending_videos.

Table: youtube_data1.trending_videos
Columns:
- video_id (string)
- title (string)
- channel (string)
- category (string)
- region (string)          -- Partition column
- views (bigint)
- likes (bigint)
- comments (bigint)
- tags (string)
- published_date (date)
- fetched_on (date)

Guidelines:
1. Use table name as youtube_data1.trending_videos.
2. Use WHERE clause to filter partition column `region`.
3. Use format: published_date >= 'YYYY-MM-DD'.
4. Only return a valid SQL query, no explanation or markdown.
"""

# Build the LLaMA prompt
def build_prompt(user_question):
    return f"""
{SCHEMA}

User Question: {user_question}
SQL:
"""

# Call Ollama (LLaMA 3.1)
def get_sql_from_ollama(user_question):
    prompt = build_prompt(user_question)
    response = requests.post(
        "http://localhost:11434/api/generate",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False
        })
    )
    if response.status_code == 200:
        result = response.json()["response"]
        return extract_sql(result)
    else:
        print("Error:", response.text)
        return None

def extract_sql(response_text):
    for line in response_text.splitlines():
        if line.strip().lower().startswith(("select", "with", "insert", "update", "delete")):
            return line.strip()
    return response_text.strip()

if __name__ == "__main__":
    question = "List top 10 most liked videos published after 2023 from the US."
    sql = get_sql_from_ollama(question)
    print("Generated SQL:\n", sql)
