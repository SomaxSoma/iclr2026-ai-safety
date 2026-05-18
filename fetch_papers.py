import openreview
import pandas as pd
import json

client = openreview.api.OpenReviewClient(
    baseurl = 'https://api2.openreview.net'
)
print("Fetching ICLR 2026 papers...")

submissions = client.get_all_notes(
    invitation = 'ICLR.cc/2026/Conference/-/Submission'
)

papers = []
for paper in submissions :
    venue = paper.content.get('venueid', {}).get('value', '')
    if 'Rejected' not in venue and 'Withdrawn' not in venue and venue != '':
        papers.append({
        'id' : paper.id,
        'title' : paper.content.get('title', {}).get('value', ''),
        'abstract' : paper.content.get('abstract', {}).get('value', ''),
        'keywords' : paper.content.get('keywords', {}).get('value', ''),
        
        })
print(f"Fetched {len(papers)} papers")

df = pd.DataFrame(papers)
df.to_csv('iclr2026_papers.csv', index=False)
print("Saved to iclr2026_papers.csv")

