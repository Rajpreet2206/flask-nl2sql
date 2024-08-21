import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask import Flask, request, render_template
import sqlite3
# Initialize Flask app
app = Flask(__name__)

# Initialize the tokenizer and model from Hugging Face Transformers library
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained('cssupport/t5-small-awesome-text-to-sql')
model = model.to(device)
model.eval()

def generate_sql(input_prompt):
    # Tokenize the input prompt
    inputs = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    # Decode the output IDs to a string (SQL query in this case)
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_sql

def execute_sql_query(sql_query):
    # Connect to SQLite3 database
    conn = sqlite3.connect('database/nl2sql_models.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM MODEL
    """)
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()  # Fetch all rows of the query result
        columns = [description[0] for description in cursor.description]  # Get column names
        results = [dict(zip(columns, row)) for row in results]  # Convert to list of dictionaries
        conn.commit()  # Commit any changes (for example, if the query is an INSERT, UPDATE, etc.)
    except sqlite3.Error as e:
        results = None
        print(f"SQL Error: {e}")
    finally:
        cursor.close()
        conn.close()

    return results    

@app.route('/', methods=['GET', 'POST'])
def index():
    sql_query = None
    results = None
    error = None

    if request.method == 'POST':
        query = request.form['query']
        
        try:
            # Generate SQL from natural language
            sql_query = generate_sql(query)
            print(f"Generated SQL Query: {sql_query}")
            results=execute_sql_query(sql_query)
            if not results:
                results="No results found or error executing the generated query."

        except Exception as e:
            error = f"An error occurred: {str(e)}"

    return render_template('index2.html', sql_query=sql_query, results=results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
