from flask import Flask, render_template, request, g
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sqlite3

app = Flask(__name__)

# Initialize the T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("cssupport/t5-small-awesome-text-to-sql")
model = AutoModelForSeq2SeqLM.from_pretrained("cssupport/t5-small-awesome-text-to-sql")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect('database/nl2sql_models.db')
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def generate_sql(input_prompt):
    # Tokenize the input prompt
    inputs = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    # Decode the output IDs to a string (SQL query)
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_sql

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'input_prompt' in request.form:
            input_prompt = request.form['input_prompt']
            try:
                generated_sql = generate_sql(input_prompt)
                print(f"Generated SQL query: {generated_sql}")
            except Exception as e:
                print(f"Error generating SQL query: {e}")
                return render_template('index.html', error="Error generating SQL query. Please try again.")

            db = get_db()
            c = db.cursor()
            try:
                c.execute(generated_sql)
                results = c.fetchall()
                
                # Save the query and results to the database
                c.execute("INSERT INTO sql_queries (query, result) VALUES (?, ?)", (generated_sql, str(results)))
                db.commit()
            except sqlite3.OperationalError as e:
                print(f"SQL syntax error: {e}")
                print(f"Generated SQL query: {generated_sql}")
                return render_template('index.html', error="Invalid SQL query. Please try a different prompt.")
            except Exception as e:
                print(f"Error executing SQL query: {e}")
                return render_template('index.html', error="Error executing SQL query. Please try again.")
            
            return render_template('index.html', generated_sql=generated_sql, results=results)
        else:
            return render_template('index.html', error="No input prompt provided.")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)