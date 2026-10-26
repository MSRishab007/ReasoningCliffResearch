import os
import glob
import pandas as pd
import json

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
OUTPUT_HTML = "reasoning_viewer.html"

# Specify the numerical IDs of the questions you want to inspect deeply
TARGET_Q_NUMBERS = ["1", "2", "3", "4", "5"] 
BUDGETS = [50, 100, 200, 400, 800, 3000]

EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "generated_graphs", "dashboard_assets"}

# ---------------------------------------------------------
# 2. Data Collection
# ---------------------------------------------------------
print("Scanning for data...")
all_data = []
models = set()
languages = set()

for item in os.listdir("."):
    if os.path.isdir(item) and item not in EXCLUDE_DIRS:
        csv_files = glob.glob(os.path.join(item, "*.csv"))
        if not csv_files:
            continue
            
        print(f"Found model folder: {item}")
        
        for f in csv_files:
            try:
                temp_df = pd.read_csv(f)
                
                # Standardize question ID
                if 'question_id' not in temp_df.columns:
                     temp_df['question_id'] = temp_df['sample_id'].str.split('_b').str[0]
                     
                # Extract the Question Number (the part after the last underscore, e.g., "1" from "GSM8K_EN_1")
                temp_df['q_num'] = temp_df['question_id'].astype(str).str.split('_').str[-1]
                
                # Filter to only the requested question numbers
                filtered_df = temp_df[temp_df['q_num'].isin(TARGET_Q_NUMBERS)].copy()
                
                if not filtered_df.empty:
                    filtered_df['model'] = item
                    all_data.append(filtered_df)
                    models.add(item)
                    languages.update(filtered_df['language'].unique())
                    
            except Exception as e:
                print(f"  [!] Error reading {f}: {e}")

if not all_data:
    print("No data found for the target question numbers. Check your TARGET_Q_NUMBERS list.")
    exit()

df = pd.concat(all_data, ignore_index=True)

# Clean and standardize columns
if 'max_cutoff_limit' not in df.columns and 'budget' in df.columns:
    df['max_cutoff_limit'] = df['budget']

# Convert 'correct' to boolean safely
if df['correct'].dtype == object:
    df['correct'] = df['correct'].astype(str).str.lower() == 'true'
else:
    df['correct'] = df['correct'].astype(bool)
    
# Fill NaN texts with empty strings to prevent JSON serialization errors
for col in ['full_thinking_text', 'raw_answer_text', 'parsed_answer', 'ground_truth']:
    if col in df.columns:
        df[col] = df[col].fillna("")

# ---------------------------------------------------------
# 3. Restructure Data for the HTML Template
# ---------------------------------------------------------
# Nested Structure: data[model][q_num][language][budget]
structured_data = {}

models = sorted(list(models))
languages = sorted(list(languages))
q_numbers = sorted(list(set(df['q_num'].unique())), key=int)

for m in models:
    structured_data[m] = {}
    for q in q_numbers:
        structured_data[m][q] = {}
        for l in languages:
            structured_data[m][q][l] = {}

for index, row in df.iterrows():
    m = row['model']
    q = row['q_num']
    l = row['language']
    b = int(row['max_cutoff_limit'])
    
    # Restrict to chosen budgets or map the max budget to 3000
    if b not in BUDGETS:
         if b == 3000 or b == df['max_cutoff_limit'].max():
             b = 3000
         else:
             continue

    structured_data[m][q][l][str(b)] = {
        "tokens_used": int(row['tokens_used']),
        "correct": bool(row['correct']),
        "cutoff_status": str(row['cutoff_status']),
        "target_lang_ratio": float(row.get('target_lang_ratio_pct', 0.0)),
        "ground_truth": str(row['ground_truth']),
        "parsed_answer": str(row['parsed_answer']),
        "raw_answer": str(row['raw_answer_text']),
        "thinking": str(row['full_thinking_text']),
        "question": str(row['question'])
    }

json_data = json.dumps(structured_data)

# ---------------------------------------------------------
# 4. Generate HTML Template
# ---------------------------------------------------------
print("Generating HTML file...")

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reasoning Trace Viewer</title>
    <style>
        :root {{
            --bg-color: #f4f4f9; --container-bg: #ffffff;
            --text-main: #333; --text-muted: #666;
            --border: #ddd; --accent-blue: #007bff;
            --success-green: #28a745; --fail-red: #dc3545;
            --trace-bg: #f8f9fa;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color); color: var(--text-main);
            margin: 0; padding: 20px;
        }}
        .header {{ text-align: center; margin-bottom: 20px; }}
        .controls {{
            display: flex; justify-content: center; gap: 15px; margin-bottom: 20px; flex-wrap: wrap;
        }}
        select, button {{
            padding: 10px 15px; font-size: 16px; border: 1px solid var(--border);
            border-radius: 5px; background-color: var(--container-bg); cursor: pointer;
        }}
        select:focus, button:hover {{ border-color: var(--accent-blue); outline: none; }}
        .main-container {{ display: flex; gap: 20px; max-width: 1400px; margin: 0 auto; }}
        .panel {{
            background: var(--container-bg); border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05); padding: 20px; flex: 1;
            display: flex; flex-direction: column; overflow: hidden;
        }}
        .panel h3 {{ margin-top: 0; border-bottom: 2px solid var(--accent-blue); padding-bottom: 10px; }}
        .info-grid {{
            display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
            margin-bottom: 15px; background: #f1f3f5; padding: 15px; border-radius: 5px;
        }}
        .info-item span {{ font-weight: bold; color: var(--text-muted); }}
        .text-box {{
            background: var(--trace-bg); border: 1px solid var(--border); border-radius: 5px;
            padding: 15px; white-space: pre-wrap; font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px; overflow-y: auto; flex-grow: 1; max-height: 600px;
        }}
        .status-correct {{ color: var(--success-green); font-weight: bold; }}
        .status-incorrect {{ color: var(--fail-red); font-weight: bold; }}
        .budget-tabs {{
            display: flex; gap: 5px; margin-bottom: 15px; border-bottom: 1px solid var(--border); padding-bottom: 10px;
        }}
        .budget-tab {{ padding: 5px 15px; background: #eee; border: none; border-radius: 3px; cursor: pointer; }}
        .budget-tab.active {{ background: var(--accent-blue); color: white; font-weight: bold; }}
        .question-box {{
            background: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 15px;
            font-size: 16px; border-left: 4px solid var(--accent-blue);
        }}
    </style>
</head>
<body>

    <div class="header">
        <h1>🔍 Reasoning Trace Viewer</h1>
        <p>Compare how different models build their logic across languages and token budgets.</p>
    </div>

    <div class="controls">
        <select id="modelSelect" onchange="updateView()">
            <option value="">-- Select Model --</option>
            {''.join([f'<option value="{m}">{m}</option>' for m in models])}
        </select>
        
        <select id="qNumSelect" onchange="updateView()">
            <option value="">-- Select Question # --</option>
            {''.join([f'<option value="{q}">Question {q}</option>' for q in q_numbers])}
        </select>
        
        <select id="langSelect" onchange="updateView()">
            <option value="">-- Select Language --</option>
            {''.join([f'<option value="{l}">{l.capitalize()}</option>' for l in languages])}
        </select>
    </div>

    <div class="main-container">
        <!-- Left Panel: Context & Details -->
        <div class="panel" style="flex: 1;">
            <h3>Overview</h3>
            <div id="questionContainer" class="question-box">
                Select a model, question number, and language to view the prompt.
            </div>
            
            <div class="budget-tabs" id="budgetTabs"></div>
            
            <div class="info-grid" id="infoGrid">
                <div class="info-item"><span>Budget Cap:</span> <span id="lblBudget">-</span></div>
                <div class="info-item"><span>Tokens Used:</span> <span id="lblTokens">-</span></div>
                <div class="info-item"><span>Status:</span> <span id="lblStatus">-</span></div>
                <div class="info-item"><span>Target Lang Ratio:</span> <span id="lblRatio">-</span>%</div>
                <div class="info-item" style="grid-column: span 2;">
                    <span>Evaluation:</span> 
                    <span id="lblEval" style="font-size: 18px;">-</span>
                </div>
            </div>
            
            <h4>Final Extracted Answer</h4>
            <div class="text-box" id="answerBox" style="max-height: 120px; flex-grow: 0; background: #fff;">-</div>
        </div>

        <!-- Right Panel: Full Trace -->
        <div class="panel" style="flex: 2;">
            <h3>Latent Reasoning Trace</h3>
            <div class="text-box" id="traceBox">Select parameters to view the internal thought process...</div>
        </div>
    </div>

    <script>
        const evaluationData = {json_data};
        let currentBudget = "3000"; 

        function updateView() {{
            const model = document.getElementById("modelSelect").value;
            const qNum = document.getElementById("qNumSelect").value;
            const lang = document.getElementById("langSelect").value;
            
            if (!model || !qNum || !lang || !evaluationData[model] || !evaluationData[model][qNum] || !evaluationData[model][qNum][lang]) {{
                clearView();
                if (model && qNum && lang) {{
                    document.getElementById("questionContainer").innerText = "⚠️ No data available for this specific combination.";
                }}
                return;
            }}
            
            const questionData = evaluationData[model][qNum][lang];
            const availableBudgets = Object.keys(questionData).sort((a, b) => parseInt(a) - parseInt(b));
            
            if (availableBudgets.length === 0) {{
                clearView();
                document.getElementById("questionContainer").innerText = "⚠️ No budget data available.";
                return;
            }}
            
            if (!availableBudgets.includes(currentBudget)) {{
                currentBudget = availableBudgets[availableBudgets.length - 1];
            }}
            
            let tabsHtml = "";
            availableBudgets.forEach(b => {{
                const isActive = (b === currentBudget) ? "active" : "";
                tabsHtml += `<button class="budget-tab ${{isActive}}" onclick="setBudget('${{b}}')">${{b}} Tokens</button>`;
            }});
            document.getElementById("budgetTabs").innerHTML = tabsHtml;

            renderBudgetData(questionData[currentBudget]);
        }}
        
        function setBudget(budgetStr) {{
            currentBudget = budgetStr;
            updateView();
        }}

        function renderBudgetData(data) {{
            if (!data) return;
            
            document.getElementById("questionContainer").innerText = data.question;
            document.getElementById("lblBudget").innerText = currentBudget;
            document.getElementById("lblTokens").innerText = data.tokens_used;
            document.getElementById("lblStatus").innerText = data.cutoff_status.replace(/_/g, ' ');
            document.getElementById("lblRatio").innerText = data.target_lang_ratio.toFixed(1);
            
            const evalSpan = document.getElementById("lblEval");
            if (data.correct) {{
                evalSpan.innerHTML = `<span class="status-correct">✅ Correct (Target: ${{data.ground_truth}}, Parsed: ${{data.parsed_answer}})</span>`;
            }} else {{
                evalSpan.innerHTML = `<span class="status-incorrect">❌ Incorrect (Target: ${{data.ground_truth}}, Parsed: ${{data.parsed_answer}})</span>`;
            }}
            
            document.getElementById("answerBox").innerText = data.raw_answer || "[Empty Answer Generation]";
            document.getElementById("traceBox").innerText = data.thinking || "[Empty Reasoning Trace]";
        }}
        
        function clearView() {{
            document.getElementById("questionContainer").innerText = "Select a model, question number, and language to view the prompt.";
            document.getElementById("budgetTabs").innerHTML = "";
            document.getElementById("lblBudget").innerText = "-";
            document.getElementById("lblTokens").innerText = "-";
            document.getElementById("lblStatus").innerText = "-";
            document.getElementById("lblRatio").innerText = "-";
            document.getElementById("lblEval").innerText = "-";
            document.getElementById("answerBox").innerText = "-";
            document.getElementById("traceBox").innerText = "Select parameters to view the internal thought process...";
        }}
    </script>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ Success! HTML viewer generated at: {OUTPUT_HTML}")