#!/usr/bin/env python3
"""
Create a visualization highlighting key findings from 2010-2020 decennial analysis
for all 77 Chicago Community Areas.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('decennial_2010_2020_comparison_ca.csv')

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Key Findings: Chicago Community Areas 2010-2020', 
             fontsize=18, fontweight='bold', y=0.98)

# Color scheme
growth_color = '#2ecc71'
decline_color = '#e74c3c'
neutral_color = '#95a5a6'

# ========== Panel 1: Top 10 Population Growth & Decline ==========
ax1 = axes[0, 0]
top_growth = df.nlargest(10, 'pop_total_change')[['ca_num', 'pop_total_change']].copy()
top_decline = df.nsmallest(10, 'pop_total_change')[['ca_num', 'pop_total_change']].copy()

combined = pd.concat([top_growth, top_decline]).sort_values('pop_total_change')
colors = [growth_color if x > 0 else decline_color for x in combined['pop_total_change']]

y_pos = np.arange(len(combined))
ax1.barh(y_pos, combined['pop_total_change'], color=colors, alpha=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"CA {int(x)}" for x in combined['ca_num']], fontsize=9)
ax1.set_xlabel('Population Change', fontweight='bold')
ax1.set_title('Top 10 Growth & Decline Areas', fontweight='bold', pad=10)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(combined.iterrows()):
    val = row['pop_total_change']
    ax1.text(val + (2000 if val > 0 else -2000), i, f'{int(val):,}', 
             va='center', ha='left' if val > 0 else 'right', fontsize=8)

# ========== Panel 2: Hispanic Share Change Distribution ==========
ax2 = axes[0, 1]
hisp_change = df['hispanic_pct_change'].dropna()
ax2.hist(hisp_change, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
ax2.axvline(x=hisp_change.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {hisp_change.mean():.1f}pp')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_xlabel('Hispanic Share Change (percentage points)', fontweight='bold')
ax2.set_ylabel('Number of CAs', fontweight='bold')
ax2.set_title('Distribution of Hispanic Share Change', fontweight='bold', pad=10)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add text annotation
increasing = (hisp_change > 0).sum()
decreasing = (hisp_change < 0).sum()
ax2.text(0.95, 0.95, f'Increasing: {increasing} CAs\nDecreasing: {decreasing} CAs', 
         transform=ax2.transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

# ========== Panel 3: White Share vs Black Share Change ==========
ax3 = axes[1, 0]
scatter = ax3.scatter(df['white_pct_change'], df['black_pct_change'], 
                     c=df['pop_total_change'], cmap='RdYlGn', 
                     s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax3.set_xlabel('White Share Change (pp)', fontweight='bold')
ax3.set_ylabel('Black Share Change (pp)', fontweight='bold')
ax3.set_title('Racial Composition Shifts', fontweight='bold', pad=10)
ax3.grid(alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Population Change', rotation=270, labelpad=20, fontweight='bold')

# Add quadrant labels
ax3.text(0.05, 0.95, 'White↓ Black↑', transform=ax3.transAxes, 
         ha='left', va='top', fontsize=9, style='italic', alpha=0.7)
ax3.text(0.95, 0.95, 'White↑ Black↑', transform=ax3.transAxes, 
         ha='right', va='top', fontsize=9, style='italic', alpha=0.7)
ax3.text(0.05, 0.05, 'White↓ Black↓', transform=ax3.transAxes, 
         ha='left', va='bottom', fontsize=9, style='italic', alpha=0.7)
ax3.text(0.95, 0.05, 'White↑ Black↓', transform=ax3.transAxes, 
         ha='right', va='bottom', fontsize=9, style='italic', alpha=0.7)

# ========== Panel 4: Key Statistics Summary ==========
ax4 = axes[1, 1]
ax4.axis('off')

# Calculate key statistics
total_pop_2010 = df['pop_total_2010'].sum()
total_pop_2020 = df['pop_total_2020'].sum()
pop_change = total_pop_2020 - total_pop_2010
pop_change_pct = (pop_change / total_pop_2010) * 100

growing_cas = (df['pop_total_change'] > 0).sum()
declining_cas = (df['pop_total_change'] < 0).sum()

avg_white_change = df['white_pct_change'].mean()
avg_black_change = df['black_pct_change'].mean()
avg_hispanic_change = df['hispanic_pct_change'].mean()
avg_asian_change = df['asian_pct_change'].mean()

# Create text summary
summary_text = f"""
KEY FINDINGS (2010-2020)

OVERALL POPULATION:
• Total 2010: {total_pop_2010:,.0f}
• Total 2020: {total_pop_2020:,.0f}
• Change: {pop_change:+,.0f} ({pop_change_pct:+.2f}%)

COMMUNITY AREA TRENDS:
• Growing: {growing_cas} CAs
• Declining: {declining_cas} CAs
• Stable: {77 - growing_cas - declining_cas} CAs

RACIAL COMPOSITION SHIFTS:
• White share: {avg_white_change:.2f}pp (avg)
• Black share: {avg_black_change:.2f}pp (avg)
• Hispanic share: {avg_hispanic_change:.2f}pp (avg)
• Asian share: {avg_asian_change:.2f}pp (avg)

GROWTH HOTSPOTS:
• CA {int(df.loc[df['pop_total_change'].idxmax(), 'ca_num'])}: +{int(df['pop_total_change'].max()):,}
• CA {int(df.nlargest(2, 'pop_total_change').iloc[1]['ca_num'])}: +{int(df.nlargest(2, 'pop_total_change').iloc[1]['pop_total_change']):,}
• CA {int(df.nlargest(3, 'pop_total_change').iloc[2]['ca_num'])}: +{int(df.nlargest(3, 'pop_total_change').iloc[2]['pop_total_change']):,}

DECLINE AREAS:
• CA {int(df.loc[df['pop_total_change'].idxmin(), 'ca_num'])}: {int(df['pop_total_change'].min()):,}
• CA {int(df.nsmallest(2, 'pop_total_change').iloc[1]['ca_num'])}: {int(df.nsmallest(2, 'pop_total_change').iloc[1]['pop_total_change']):,}
• CA {int(df.nsmallest(3, 'pop_total_change').iloc[2]['ca_num'])}: {int(df.nsmallest(3, 'pop_total_change').iloc[2]['pop_total_change']):,}
"""

ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, 
         ha='left', va='top', fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9, pad=1))

plt.tight_layout()
plt.savefig('decennial_key_findings_2010_2020.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Created decennial_key_findings_2010_2020.png")
print(f"   Total population change: {pop_change:+,.0f} ({pop_change_pct:+.2f}%)")
print(f"   Growing CAs: {growing_cas}, Declining CAs: {declining_cas}")
