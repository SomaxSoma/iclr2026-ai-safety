import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('safety_results.csv')
safety = df[df['is_safety'] == True]

print(f"Total ICLR 2026 papers: {len(df)}")
print(f"AI Safety papers: {len(safety)}")
print(f"Percentage: {round(len(safety)/len(df)*100, 1)}%")
print()
print("Breakdown by subdomain:")
subdomain_counts = safety['subdomain'].value_counts()
print(subdomain_counts)
print()
print("High confidence only:")
high = safety[safety['confidence'] == 'high']
print(f"High confidence safety papers: {len(high)}")
print(high['subdomain'].value_counts())

# Bar chart
plt.figure(figsize=(12, 6))
subdomain_counts.plot(kind='barh', color='steelblue')
plt.xlabel('Number of Papers')
plt.title('AI Safety Research at ICLR 2026 — Subdomain Breakdown')
plt.tight_layout()
plt.savefig('ai_safety_iclr2026.png')
print()
print("Chart saved to ai_safety_iclr2026.png")
