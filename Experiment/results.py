import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
import matplotlib.patches as mpatches

# Create output directory
output_dir = "Experiment/graphs_short_labels"
os.makedirs(output_dir, exist_ok=True)

# Load datasets
summary_df = pd.read_csv("Experiment/Results/qwen3_4b_nf4_bf16_summary_by_technique_v2.csv")
results_df = pd.read_csv("Experiment/Results/qwen3_4b_nf4_bf16_gsm8k_hi_results_v2.csv")

# Set theme
sns.set_theme(style="whitegrid")

# Define mapping based on README.md
technique_mapping = {
    'question_only': 'E1',
    'direct': 'E2',
    'native': 'E3',
    'cot': 'E4',
    'structured': 'E5',
    'english_pivot': 'E6',
    'plan_solve': 'E7',
    'self_consistency': 'E8',
    'mspr': 'E9',
    'mspr_verifier': 'E10',
    'answer_trigger': 'E11',
    'answer_trigger_forced': 'E12'
}

summary_df['short_label'] = summary_df['technique_id'].map(technique_mapping)

# Sort numerically by E-number
summary_df['sort_key'] = summary_df['short_label'].str.extract(r'E(\d+)').astype(int)
summary_df = summary_df.sort_values('sort_key')

# Map for legends
legend_labels = summary_df['short_label'] + ": " + summary_df['technique_name']

# 1. Accuracy Graph
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='short_label', y='accuracy_percent', data=summary_df, palette="viridis")
plt.title('Accuracy by Technique (Qwen3-4B NF4/BF16 on GSM8K Hindi)', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xlabel('Experiment ID', fontsize=12)
plt.ylim(0, 100)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

handles = [mpatches.Patch(color='white', label=label) for label in legend_labels]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Techniques", frameon=False)
plt.tight_layout()
plt.savefig(f"{output_dir}/accuracy_by_technique_short.png", bbox_inches='tight', dpi=300)
plt.close()

# 2. Time Taken Graph
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='short_label', y='avg_technique_inference_time_seconds', data=summary_df, palette="plasma")
plt.title('Average Inference Time per Question by Technique', fontsize=16)
plt.ylabel('Average Time (Seconds)', fontsize=12)
plt.xlabel('Experiment ID', fontsize=12)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}s', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Techniques", frameon=False)
plt.tight_layout()
plt.savefig(f"{output_dir}/inference_time_by_technique_short.png", bbox_inches='tight', dpi=300)
plt.close()

# 3. Hindi Percentage Graph
def calculate_hindi_percentage(text):
    if not isinstance(text, str) or len(text) == 0: return np.nan
    text_clean = re.sub(r'[\s\d\W_]', '', text) 
    if len(text_clean) == 0: return 0.0
    devanagari_chars = re.findall(r'[\u0900-\u097F]', text_clean)
    return (len(devanagari_chars) / len(text_clean)) * 100

results_df['short_label'] = results_df['technique_id'].map(technique_mapping)
results_df['hindi_thinking_pct'] = results_df['thinking_text'].apply(calculate_hindi_percentage)
results_df['hindi_answering_pct'] = results_df['final_response'].apply(calculate_hindi_percentage)

lang_usage_df = results_df.groupby('short_label').agg({'hindi_thinking_pct': 'mean', 'hindi_answering_pct': 'mean'}).reset_index()
summary_with_lang = pd.merge(summary_df, lang_usage_df, on='short_label', how='left')

plt.figure(figsize=(12, 6))
melted_df = summary_with_lang.melt(id_vars=['short_label'], value_vars=['hindi_thinking_pct', 'hindi_answering_pct'], var_name='Phase', value_name='Hindi_Percentage')
melted_df['Phase'] = melted_df['Phase'].map({'hindi_thinking_pct': 'Thinking Phase', 'hindi_answering_pct': 'Answering Phase'})

ax = sns.barplot(x='short_label', y='Hindi_Percentage', hue='Phase', data=melted_df, palette="muted")
plt.title('Hindi Script (Devanagari) Usage by Technique Phase', fontsize=16)
plt.ylabel('Average Devanagari Characters (%)', fontsize=12)
plt.xlabel('Experiment ID', fontsize=12)
plt.ylim(0, 100)

phase_handles, phase_labels = ax.get_legend_handles_labels()
phase_handles.append(mpatches.Patch(color='white', label=''))
phase_labels.append("--- Techniques ---")
for label in legend_labels:
    phase_handles.append(mpatches.Patch(color='white', label=label))
    phase_labels.append(label)

ax.legend(handles=phase_handles, labels=phase_labels, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig(f"{output_dir}/hindi_usage_by_technique_short.png", bbox_inches='tight', dpi=300)
plt.close()

# 4. Numeral System Usage
plt.figure(figsize=(12, 6))
numeral_cols = ['devanagari_outputs', 'hindu_arabic_outputs', 'mixed_outputs', 'unparsed_outputs']
numeral_data = summary_df[['short_label'] + numeral_cols].set_index('short_label')
numeral_pct = numeral_data.div(numeral_data.sum(axis=1), axis=0) * 100

ax = numeral_pct.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set2')
plt.title('Numeral System Distribution in Final Outputs', fontsize=16)
plt.ylabel('Percentage of Outputs (%)', fontsize=12)
plt.xlabel('Experiment ID', fontsize=12)
plt.xticks(rotation=0)

num_handles, num_labels = ax.get_legend_handles_labels()
num_handles.append(mpatches.Patch(color='white', label=''))
num_labels.append("--- Techniques ---")
for label in legend_labels:
    num_handles.append(mpatches.Patch(color='white', label=label))
    num_labels.append(label)

ax.legend(handles=num_handles, labels=num_labels, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig(f"{output_dir}/numeral_systems_by_technique_short.png", bbox_inches='tight', dpi=300)
plt.close()

print(f"Graphs successfully generated with E1-E12 short labels in '{output_dir}'.")
