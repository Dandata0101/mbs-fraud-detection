import pandas as pd
import os
from rich.table import Table
from rich.console import Console
from rich.box import ROUNDED
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import openpyxl
print('')
print('')

# Set up the console for rich printing
console = Console()
# Define file path
current_directory = os.getcwd()

# Function to print unique senders from filtered emails
def print_unique_senders(emails_df):
    console.print(f"Unique Senders in Filtered Emails:", style="bold underline")
    unique_senders = emails_df['From'].dropna().unique()
    for sender in unique_senders:
        console.print(f"- {sender}")

# Function to display flagged email count
def display_flagged_email_count(df):
    console.print("Flagged Emails Count:", style="bold underline")
    flag_table = Table(show_header=True, header_style="bold magenta")
    flag_table.add_column("Flag", style="dim")
    flag_table.add_column("Count", justify="right")
    flag_counts = df['flag'].value_counts().items()
    for flag, count in flag_counts:
        flag_table.add_row(str(flag), str(count))
    console.print(flag_table)

def display_email_table(emails_df):
    console = Console()  # Ensure a Console instance is created to use it for printing
    console.print("Filtered Email(s):", style="bold underline")
    email_table = Table(show_header=True, header_style="bold magenta", box=ROUNDED)
    
    # Increase the width of each column here. Adjust these values as needed based on your display.
    email_table.add_column("Message-ID", style="dim", width=120, overflow="fold")
    email_table.add_column("From", width=120, overflow="fold")
    email_table.add_column("To", width=120, overflow="fold")
    email_table.add_column("Date", style="dim", width=120)
    email_table.add_column("Content", overflow="fold", width=150)  
    
    for _, row in emails_df.head().iterrows():
        email_table.add_row(
            str(row['Message-ID']),
            row['From'],
            row['To'],
            str(row['Date']),
            row['clean_content'][:300] + "..."  # Increase the initial content snippet length if desired
        )
    
    console.print(email_table)


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

# You might need to download 'punkt', 'stopwords', and 'averaged_perceptron_tagger' if you haven't already
nltk.download(current_directory+'punkt',download_dir=current_directory+'/nltk')
nltk.download('stopwords',download_dir=current_directory+'/nltk')
nltk.download('averaged_perceptron_tagger',download_dir=current_directory+'/nltk')

def generate_display_wordcloud(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Convert to lower case
    tokens = [word.lower() for word in tokens if word.isalpha()]
    
    # Filter out stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if not word in stop_words]
    
    # Perform POS tagging
    tagged_tokens = pos_tag(filtered_tokens)
    
    # Filter only nouns and verbs
    nouns_verbs = [word for word, tag in tagged_tokens if tag.startswith('NN') or tag.startswith('VB')]
    
    # Combine filtered words for word cloud
    filtered_text = " ".join(nouns_verbs)
    
    # Generate and display word cloud
    console.print("Generating and displaying word cloud for filtered emails", style="bold underline")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Emails Containing Nouns and Verbs')
    plt.show()


