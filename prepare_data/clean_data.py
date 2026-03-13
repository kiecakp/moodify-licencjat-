import csv
import os

INPUT_CSV = 'test/training_data_watki.csv'
OUTPUT_CSV = 'test/cleaned_training_data_watki.csv'

def clean_data():
  if not os.path.exists(INPUT_CSV):
    print(f"File {INPUT_CSV} does not exist.")
    return

  with open(INPUT_CSV, mode='r', newline='', encoding='utf-8') as f:
    reader = list(csv.reader(f))
    header = reader[0]
    rows = reader[1:]

  cleaned_rows = []
  removed_count = 0

  for row in rows:
    lyrics_en = row[header.index('LYRICS_EN')].strip()
    if lyrics_en:
      cleaned_rows.append(row)
    else:
      removed_count += 1

  with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(cleaned_rows)

  print("🧹 Cleanup complete")
  print(f"   ❌ Rows removed (empty LYRICS_EN): {removed_count}")
  print(f"   📁 Clean dataset saved as: {OUTPUT_CSV}")
  print(f"   ✅ Final size: {len(cleaned_rows)} rows")

if __name__ == "__main__":
  clean_data()