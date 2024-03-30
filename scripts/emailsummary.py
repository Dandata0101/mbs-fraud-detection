from IPython.display import HTML, display
import os
import openai
import pandas as pd

def summarize_text_column_with_openai_chat(text_column, openai_api_key):
    # Set the OpenAI API key
    openai.api_key = openai_api_key
    
    # Concatenate all text entries into one large text for summarization, with null checks
    full_text = ' '.join([str(text) for text in text_column if pd.notnull(text)])
    
    # Limit the size of the full_text to avoid exceeding token limits
    full_text = full_text[:2000]  # Example: limit to the first 2000 characters

    # Use the chat completions endpoint for generating a summary with the chat model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Specify the chat model
        messages=[
            {"role": "system", "content": "Your task is to summarize the following text."},
            {"role": "user", "content": full_text}
        ]
    )
    
    # Extract the summary text from the response
    summary = response['choices'][0]['message']['content'].strip()
    
    # Use display to show the summary in HTML format
    display(HTML(f"""
    <table style="width: 100%; border: 1px solid black; font-size: 20px;">
        <tr>
            <th style="text-align: left; padding-right: 20px;">Summary</th>
        </tr>
        <tr>
            <td style="text-align: left; padding-right: 20px;">{summary}</td>
        </tr>
    </table>
    """));
    
    # Return nothing to avoid the last output being 'None'
    return
