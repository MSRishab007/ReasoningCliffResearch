import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os

# Create output directory
output_dir = "Experiment/graphs"
os.makedirs(output_dir, exist_ok=True)

# Load datasets
summary_df = pd.read_csv("Experiment/results/qwen3_4b_nf4_bf16_summary_by_technique_v2.csv")
results_df = pd.read_csv("Experiment/results/qwen3_4b_nf4_bf16_gsm8k_hi_results_v2.csv")

# Set theme
sns.set_theme(style="whitegrid")

# Sort summary by Technique ID (E1, E2, etc.) safely
def extract_num(t_id):
    if pd.isna(t_id):
        return 999
    match = re.search(r'(\d+)', str(t_id))
    return int(match.group(1)) if match else 999

summary_df['sort_key'] = summary_df['technique_id'].apply(extract_num)
summary_df = summary_df.sort_values('sort_key')
technique_labels = summary_df['technique_id'] + " (" + summary_df['technique_name'] + ")"

# 1. Accuracy Graph
plt.figure(figsize=(14, 7))
ax = sns.barplot(x='technique_id', y='accuracy_percent', data=summary_df, palette="viridis")
plt.title('Accuracy by Technique (Qwen3-4B NF4/BF16 on GSM8K Hindi)', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xlabel('Technique ID', fontsize=12)
plt.ylim(0, 100)

for i, p in enumerate(ax.patches):
    ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                xytext=(0, 5), 
                textcoords='offset points')

for i, row in summary_df.iterrows():
    plt.plot([], [], ' ', label=f"{row['technique_id']}: {row['technique_name']}")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Techniques")

plt.tight_layout()
plt.savefig(f"{output_dir}/accuracy_by_technique.png", bbox_inches='tight', dpi=300)
plt.close()

# 2. Time Taken Graph
plt.figure(figsize=(14, 7))
ax = sns.barplot(x='technique_id', y='avg_technique_inference_time_seconds', data=summary_df, palette="plasma")
plt.title('Average Inference Time per Question by Technique', fontsize=16)
plt.ylabel('Average Time (Seconds)', fontsize=12)
plt.xlabel('Technique ID', fontsize=12)

for i, p in enumerate(ax.patches):
    ax.annotate(f'{p.get_height():.1f}s', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                xytext=(0, 5), 
                textcoords='offset points')

for i, row in summary_df.iterrows():
    plt.plot([], [], ' ', label=f"{row['technique_id']}: {row['technique_name']}")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Techniques")

plt.tight_layout()
plt.savefig(f"{output_dir}/inference_time_by_technique.png", bbox_inches='tight', dpi=300)
plt.close()

# 3. Hindi Percentage Graph
def calculate_hindi_percentage(text):
    if not isinstance(text, str) or len(text) == 0:
        return np.nan
    text_clean = re.sub(r'[\s\d\W_]', '', text) 
    if len(text_clean) == 0:
        return 0.0
    devanagari_chars = re.findall(r'[\u0900-\u097F]', text_clean)
    return (len(devanagari_chars) / len(text_clean)) * 100

results_df['hindi_thinking_pct'] = results_df['thinking_text'].apply(calculate_hindi_percentage)
results_df['hindi_answering_pct'] = results_df['final_response'].apply(calculate_hindi_percentage)

lang_usage_df = results_df.groupby(['technique_id']).agg({
    'hindi_thinking_pct': 'mean',
    'hindi_answering_pct': 'mean'
}).reset_index()

summary_with_lang = pd.merge(summary_df, lang_usage_df, on='technique_id', how='left')

plt.figure(figsize=(14, 7))
melted_df = summary_with_lang.melt(
    id_vars=['technique_id'], 
    value_vars=['hindi_thinking_pct', 'hindi_answering_pct'],
    var_name='Phase', 
    value_name='Hindi_Percentage'
)
melted_df['Phase'] = melted_df['Phase'].map({
    'hindi_thinking_pct': 'Thinking Phase',
    'hindi_answering_pct': 'Answering Phase'
})

ax = sns.barplot(x='technique_id', y='Hindi_Percentage', hue='Phase', data=melted_df, palette="muted")
plt.title('Hindi Script (Devanagari) Usage by Technique Phase', fontsize=16)
plt.ylabel('Average Devanagari Characters (%)', fontsize=12)
plt.xlabel('Technique ID', fontsize=12)
plt.ylim(0, 100)

for i, row in summary_df.iterrows():
    plt.plot([], [], ' ', label=f"{row['technique_id']}: {row['technique_name']}")
legend1 = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Techniques")
plt.gca().add_artist(legend1)

plt.tight_layout()
plt.savefig(f"{output_dir}/hindi_usage_by_technique.png", bbox_inches='tight', dpi=300)
plt.close()

# 4. Numeral System Usage Stacked Bar Chart
plt.figure(figsize=(14, 7))
numeral_cols = ['devanagari_outputs', 'hindu_arabic_outputs', 'mixed_outputs', 'unparsed_outputs']
numeral_data = summary_df[['technique_id'] + numeral_cols].set_index('technique_id')
numeral_pct = numeral_data.div(numeral_data.sum(axis=1), axis=0) * 100

ax = numeral_pct.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='Set2')
plt.title('Numeral System Distribution in Final Outputs', fontsize=16)
plt.ylabel('Percentage of Outputs (%)', fontsize=12)
plt.xlabel('Technique ID', fontsize=12)
plt.legend(title='Numeral System', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add legend for technique IDs
import matplotlib.patches as mpatches
handles, labels = ax.get_legend_handles_labels()
empty_patch = mpatches.Patch(color='white', label='')
handles.append(empty_patch)
labels.append("--- Techniques ---")

for i, row in summary_df.iterrows():
    handles.append(mpatches.Patch(color='white', label=f"{row['technique_id']}: {row['technique_name']}"))
    labels.append(f"{row['technique_id']}: {row['technique_name']}")

ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(f"{output_dir}/numeral_systems_by_technique.png", bbox_inches='tight', dpi=300)
plt.close()

print(f"Graphs successfully saved to {output_dir}/ directory.")