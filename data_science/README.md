📝 SAMO – Data Science Track
SAMO is a journaling tool that helps users track their emotions by analyzing their text or voice entries.  The goal is to provide a "Memory Lane" dashboard that visualizes emotional patterns over time.

This repository focuses specifically on the data science work that powers SAMO's core functionality.

📊 The Data Science Workflow
Exploratory Data Analysis (EDA)
I began by diving into the GoEmotions dataset. The analysis involved:

Examining the frequency, distribution, and co-occurrence of emotion labels.

Analyzing statistics related to journal entry length to better understand the data we'd be processing.

Model Development
The core of my work involved fine-tuning and adapting a pretrained model for emotion detection. I used HuggingFace's bhadresh-savani/bert-base-go-emotion model and made a few key adjustments to improve its performance on our specific use case:

I implemented a sentence-level chunking strategy to effectively handle long journal entries, as the model has a token limit.

A keyword-based fallback/boosting system was added to ensure reliable emotion detection even when the model's confidence was low.

Output & Evaluation
The final step was structuring the output and evaluating the model's performance.

Structured JSON Output: For each journal entry, I generated a clean JSON object containing detected emotions, confidence scores, and macro-level emotion groupings. I also developed a weekly aggregation system to track "streaks" and create the data for the Memory Lane dashboard.

Evaluation: I tested the pipeline against a small, manually labeled dataset and computed F1 scores (both micro and macro) to measure the accuracy of the predictions.

Deliverables
This track's work resulted in the following key outputs:

Processed Data: A journals_with_emotions.csv file, with emotion data appended to each entry.

User Data: Example JSON outputs for the dashboard, like memory_lane_<user>.json.

Metrics: A metrics.json file summarizing the evaluation results.


