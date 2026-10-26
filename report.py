import os
import glob
import pandas as pd

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
GRAPHS_DIR = "generated_graphs"
VIEWER_HTML = "reasoning_viewer.html"
OUTPUT_REPORT = "final_evaluation_report.html"

print("Compiling Master Final Report...")

# Ensure the graphs directory exists
if not os.path.exists(GRAPHS_DIR):
    print(f"Error: '{GRAPHS_DIR}' not found. Please run the graph generation script first.")
    exit()

# ---------------------------------------------------------
# 2. HTML Boilerplate & Styling
# ---------------------------------------------------------
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multilingual Mathematical Reasoning Under Token Budgets</title>
    <style>
        :root {{
            --primary: #1e293b;
            --secondary: #334155;
            --accent: #2563eb;
            --bg-light: #f8fafc;
            --border: #e2e8f0;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            color: var(--primary);
            background-color: #cbd5e1; /* Slightly darker background for contrast against white containers */
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        .section-card {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }}
        h1 {{ color: var(--primary); font-size: 2.5em; margin-top: 0; border-bottom: 4px solid var(--accent); padding-bottom: 10px; }}
        h2 {{ color: var(--accent); border-bottom: 2px solid var(--border); padding-bottom: 10px; margin-top: 0; }}
        h3 {{ color: var(--secondary); font-size: 1.5em; }}
        
        p.intro-text {{
            font-size: 1.1em;
            color: #475569;
        }}

        /* Model Gallery Grid */
        .model-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
            align-items: flex-start;
        }}
        .graphs-col {{
            flex: 1;
            min-width: 400px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .graphs-col img {{
            width: 100%;
            border-radius: 8px;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .stats-col {{
            flex: 1;
            min-width: 400px;
            background: var(--bg-light);
            border-radius: 8px;
            padding: 15px;
            border: 1px solid var(--border);
            overflow-x: auto;
        }}

        /* Table Styling */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background-color: var(--secondary);
            color: white;
            position: sticky;
            top: 0;
        }}
        tr:nth-child(even) {{ background-color: #f1f5f9; }}
        tr:hover {{ background-color: #e2e8f0; }}

        /* Interactive Viewer Iframe */
        .viewer-iframe {{
            width: 100%;
            height: 900px;
            border: 1px solid var(--border);
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        
        <!-- 1. Introduction Section -->
        <div class="section-card">
            <h1>Multilingual Mathematical Reasoning Under Token Budgets</h1>
            <h2>Executive Summary & Methodology</h2>
            <p class="intro-text">
                <strong>Problem Statement:</strong> Large language models (LLMs) often exhibit varying reasoning capabilities across different languages. Furthermore, non-Latin scripts, such as Indic languages, suffer from severe tokenization inefficiencies (the "Tokenizer Tax"). This project investigates how hard computational limits—enforced via reasoning token budgets—disproportionately affect mathematical reasoning accuracy across English and 10 Indic languages.
            </p>
            <p class="intro-text">
                <strong>Methodology:</strong> We evaluate models across translated versions of the GSM8K dataset. Models are allocated strict reasoning token budgets ranging from 50 to 800 tokens. Using a robust interval-probing methodology, we track exact-match accuracy, the rate of natural sequence completion (natural finish rate), and the persistence of the native script in the latent reasoning trace. The results help isolate whether reasoning failures are due to a lack of model intelligence or a lack of sufficient token runway.
            </p>
        </div>
"""

# ---------------------------------------------------------
# 3. Model Gallery Section (Side-by-Side)
# ---------------------------------------------------------
html_content += """
        <div class="section-card">
            <h2>Model Performance Gallery</h2>
            <p>The visualizations below plot the "Accuracy Cliff" and completion rates. The tables on the right provide the exact aggregated percentages for deep-dive analysis.</p>
"""

# Iterate through each model folder in generated_graphs
model_folders = sorted([d for d in os.listdir(GRAPHS_DIR) if os.path.isdir(os.path.join(GRAPHS_DIR, d))])

if not model_folders:
    html_content += "<p><em>No models found in the generated_graphs directory.</em></p>"

for model in model_folders:
    model_path = os.path.join(GRAPHS_DIR, model)
    
    # Locate assets
    accuracy_img = os.path.join(model_path, f"{model}_accuracy.png")
    finish_img = os.path.join(model_path, f"{model}_finish_rate.png")
    script_img = os.path.join(model_path, f"{model}_native_script.png")
    stats_csv = os.path.join(model_path, f"{model}_summary_stats.csv")
    
    # Build Stats Table HTML
    table_html = "<p><em>Statistical summary not found for this model.</em></p>"
    if os.path.exists(stats_csv):
        df_stats = pd.read_csv(stats_csv)
        # Rename columns for prettier presentation
        df_stats = df_stats.rename(columns={
            'language': 'Language',
            'max_cutoff_limit': 'Token Budget',
            'correct_pct': 'Accuracy (%)',
            'finished_naturally_pct': 'Finished Naturally (%)',
            'target_lang_ratio_pct': 'Native Script Ratio (%)'
        })
        table_html = df_stats.to_html(index=False, border=0, classes='data-table')

    # Ensure we only show image tags if the image actually exists
    img_tags = ""
    if os.path.exists(accuracy_img):
        img_tags += f'<img src="{accuracy_img}" alt="Accuracy Graph - {model}">'
    if os.path.exists(finish_img):
        img_tags += f'<img src="{finish_img}" alt="Finish Rate Graph - {model}">'
    if os.path.exists(script_img):
        img_tags += f'<img src="{script_img}" alt="Native Script Graph - {model}">'

    # Append Model Block
    html_content += f"""
            <div style="border: 2px solid var(--border); border-radius: 8px; padding: 20px; margin-top: 30px; background: white;">
                <h3>Model: {model}</h3>
                <div class="model-row">
                    <!-- Left Column: Graphs -->
                    <div class="graphs-col">
                        {img_tags}
                    </div>
                    
                    <!-- Right Column: Stats Table -->
                    <div class="stats-col">
                        <h4>Aggregated Results Data</h4>
                        {table_html}
                    </div>
                </div>
            </div>
    """

html_content += """
        </div>
"""

# ---------------------------------------------------------
# 4. Interactive Viewer Embedding
# ---------------------------------------------------------
# We use an iframe to embed the previously generated viewer safely.
if os.path.exists(VIEWER_HTML):
    html_content += f"""
        <div class="section-card">
            <h2>Interactive Trace Inspector</h2>
            <p>Explore the exact step-by-step reasoning outputs generated by the models before they hit their token limits.</p>
            <iframe src="{VIEWER_HTML}" class="viewer-iframe" title="Reasoning Viewer"></iframe>
        </div>
    """
else:
    html_content += f"""
        <div class="section-card">
            <h2>Interactive Trace Inspector</h2>
            <p style="color: red;"><em>Warning: '{VIEWER_HTML}' was not found in the directory. Please run the HTML viewer generation script to enable this section.</em></p>
        </div>
    """

# Close HTML tags
html_content += """
    </div>
</body>
</html>
"""

# ---------------------------------------------------------
# 5. Save the Final Report
# ---------------------------------------------------------
with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ Master Report Successfully Generated: '{OUTPUT_REPORT}'")
print(f"Open '{OUTPUT_REPORT}' in any web browser to view your complete presentation.")