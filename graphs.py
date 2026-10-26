import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
OUTPUT_DIR = "generated_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
EXCLUDE_DIRS = {OUTPUT_DIR, "__pycache__", ".git", ".ipynb_checkpoints"}

print("Scanning directory for model folders...\n")

for item in os.listdir("."):
    if os.path.isdir(item) and item not in EXCLUDE_DIRS:
        csv_files = glob.glob(os.path.join(item, "*.csv"))
        
        if not csv_files:
            continue
            
        print(f"Processing model folder: '{item}' ({len(csv_files)} CSV files found)")
        
        df_list = []
        for f in csv_files:
            try:
                df_list.append(pd.read_csv(f))
            except Exception as e:
                print(f"  [!] Error reading {f}: {e}")
                
        if not df_list:
            continue
            
        df = pd.concat(df_list, ignore_index=True)
        
        # 1. Standardize and calculate core metrics
        if 'correct' in df.columns:
            if df['correct'].dtype == object:
                df['correct_pct'] = (df['correct'].astype(str).str.lower() == 'true').astype(float) * 100
            else:
                df['correct_pct'] = df['correct'].astype(bool).astype(float) * 100
                
        if 'cutoff_status' in df.columns:
            # Strip whitespace just in case of formatting anomalies
            df['finished_naturally_pct'] = (df['cutoff_status'].astype(str).str.strip() == 'Finished_Thinking_Naturally').astype(float) * 100

        # 2. Explicitly aggregate the data
        agg_dict = {
            'correct_pct': 'mean',
            'finished_naturally_pct': 'mean'
        }
        if 'target_lang_ratio_pct' in df.columns:
            agg_dict['target_lang_ratio_pct'] = 'mean'
            
        # Group by language and budget to get the exact plot points
        summary_df = df.groupby(['language', 'max_cutoff_limit']).agg(agg_dict).reset_index()
        summary_df = summary_df.round(2) # Round to 2 decimal places for clean reading

        # 3. Setup output directory
        safe_model_name = item.replace('/', '_').replace('\\', '_')
        model_out_dir = os.path.join(OUTPUT_DIR, safe_model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        
        # 4. Save the aggregated numbers to a CSV file
        summary_csv_path = os.path.join(model_out_dir, f"{safe_model_name}_summary_stats.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"  -> Saved text summary: {summary_csv_path}")
        
        # 5. Generate Graphs using the explicitly aggregated data
        
        # A. Accuracy Graph
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=summary_df, x='max_cutoff_limit', y='correct_pct', hue='language', marker='o')
        plt.title(f'Accuracy Cliff vs. Token Budget\nModel: {item}')
        plt.ylabel('Exact-Match Accuracy (%)')
        plt.xlabel('Token Budget')
        plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(model_out_dir, f'{safe_model_name}_accuracy.png'), dpi=300)
        plt.close()

        # B. Natural Finish Rate Graph
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=summary_df, x='max_cutoff_limit', y='finished_naturally_pct', hue='language', marker='^')
        plt.title(f'Natural Finish Rate vs. Token Budget\nModel: {item}')
        plt.ylabel('Natural Finish Rate (%)')
        plt.xlabel('Token Budget')
        plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(model_out_dir, f'{safe_model_name}_finish_rate.png'), dpi=300)
        plt.close()

        # C. Native Script / Target Lang Ratio Graph
        if 'target_lang_ratio_pct' in summary_df.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=summary_df, x='max_cutoff_limit', y='target_lang_ratio_pct', hue='language', marker='s')
            plt.title(f'Target Script Usage vs. Token Budget\nModel: {item}')
            plt.ylabel('Target Language Script (%)')
            plt.xlabel('Token Budget')
            plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(model_out_dir, f'{safe_model_name}_native_script.png'), dpi=300)
            plt.close()

        print("  -> Saved all graphs successfully.\n")

print(f"✅ All summaries and plots are organized inside the '{OUTPUT_DIR}' directory.")
